"""本地一键管线桥（前端"选择文件"→ 转写 → 自动装载）。

前端（浏览器 dev / 网页版）无法本地跑 Python 管线；本服务在本地起一个
HTTP 桥（默认 127.0.0.1:8420，stdlib 零依赖）：

  GET  /health              探活（前端用它判断桥是否启动）
  POST /transcribe          音频 body（文件名放 x-filename 头，URL 编码）
                           → {"job": "<id>"}；单 GPU 串行，忙时 409
  GET  /progress/<id>       {"stage","pct","label","elapsed"}（阶段内按预估
                           时长线性插值，转写按 i/n 实进度）
  GET  /files/<id>/<path>   产物（index.json / notation/** / *.mid）

用法：
  env/python.exe pipeline_server.py            # 前端"选择文件"前先起它

进度口径：bpm 0-4% / separate 4-45% / transcribe 45-92%（实进度 i/n）/
notation 92-99% / done 100%。分离预估 ~170s（4060 全曲），超时不越界
（钉在阶段上界-1），避免假 100%。
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

SE = Path(__file__).resolve().parent
sys.path.insert(0, str(SE))
JOBS_ROOT = SE / "output" / "web_jobs"
PORT = 8420

# 阶段 → (pct 起点, pct 终点, 预估秒[阶段内插值用])
STAGE_SPANS = {
    "bpm": (0.0, 4.0, 5.0),
    "separate": (4.0, 45.0, 170.0),
    "transcribe": (45.0, 92.0, 120.0),
    "notation": (92.0, 99.0, 25.0),
}

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()          # 进度读写锁
_GPU_LOCK = threading.Lock()      # 单任务串行（GPU 只有一块）


def _now() -> float:
    return time.time()


def _persist(jid: str) -> None:
    """任务状态落盘（web_jobs/<id>/job.json）：服务重启后可恢复已完成任务、
    内容去重可秒回（2026-08-28：用户 5 分钟转写完成后前端解码失败，结果
    不能因为重试/重启再付一遍 GPU 成本）。"""
    with _LOCK:
        j = _JOBS.get(jid)
        if not j:
            return
        snap = {k: j.get(k) for k in ("stage", "pct", "label", "md5", "t0")}
    try:
        (JOBS_ROOT / jid / "job.json").write_text(
            json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _load_persisted() -> None:
    """启动时恢复历史任务记录（中断的标 error）。"""
    if not JOBS_ROOT.exists():
        return
    for d in JOBS_ROOT.iterdir():
        f = d / "job.json"
        if not d.is_dir() or not f.exists():
            continue
        try:
            snap = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stage = snap.get("stage")
        if stage not in ("done", "error"):
            stage, snap["label"] = "error", "服务重启，任务中断"
        with _LOCK:
            _JOBS[d.name] = {
                "stage": stage, "pct": 100.0 if stage == "done" else 0.0,
                "label": snap.get("label", ""), "md5": snap.get("md5", ""),
                "t0": float(snap.get("t0", _now())), "stage_t0": _now(),
                "sub_i": 0, "sub_n": 0, "dir": str(d)}


def _set(job_id: str, **kw) -> None:
    with _LOCK:
        _JOBS[job_id].update(kw)


def _set_stage(job_id: str, stage: str, label: str = "") -> None:
    with _LOCK:
        j = _JOBS[job_id]
        j.update({"stage": stage, "label": label or stage,
                  "stage_t0": _now(), "sub_i": 0, "sub_n": 0})
    print(f"[job {job_id}] {stage} {label}", flush=True)


def _tick_pct_locked(j: dict) -> None:
    """阶段内按耗时/预估线性插值；转写有 i/n 实进度时按条目推进。
    调用方必须已持有 _LOCK（不可在此再抢——非重入锁会自死锁，2026-08-28
    e2e 第一版就死在这）。"""
    stage = j.get("stage")
    if stage not in STAGE_SPANS:
        return
    lo, hi, est = STAGE_SPANS[stage]
    if stage == "transcribe" and j.get("sub_n", 0) > 0:
        frac = min(1.0, j["sub_i"] / j["sub_n"])
    else:
        elapsed = _now() - j.get("stage_t0", _now())
        frac = min(1.0, elapsed / est)
    j["pct"] = round(lo + (hi - lo) * frac, 1)


def _tick_pct(job_id: str) -> None:
    with _LOCK:
        j = _JOBS.get(job_id)
        if j is not None:
            _tick_pct_locked(j)


def _on_stage(job_id: str):
    def cb(stage: str, label: str = "") -> None:
        _set_stage(job_id, stage, label)
    return cb


def _run_job(job_id: str, audio_path: str, out_dir: Path) -> None:
    try:
        with _GPU_LOCK:
            _set_stage(job_id, "bpm", "BPM 检测")
            from run_one import assemble_demo, detect_bpm
            bpm = detect_bpm(audio_path)
            print(f"[job {job_id}] bpm={bpm}", flush=True)

            os.environ["MUSE_MULTI_INSTRUMENT"] = "1"
            os.environ["MUSE_MULTI_SEPARATION"] = "versep_guitar"
            from src.config import Config  # noqa: F401 （env 就位后 import 生效）
            from src.multi_instrument import run_multi_instrument

            # 转写实进度：钩子细分 sub_i/sub_n（multi 的钩子按 run 条目报）
            base_cb = _on_stage(job_id)

            def cb(stage: str, label: str = "") -> None:
                base_cb(stage, label)  # 先落阶段（_set_stage 会清 sub_*）
                m = re.match(r"多乐器转写 (\d+)/(\d+)", label)
                if m:  # 进入第 i 个 run = 前 i-1 个已完成
                    _set(job_id, sub_i=int(m.group(1)) - 1,
                         sub_n=int(m.group(2)))

            ok = run_multi_instrument(audio_path, str(out_dir), bpm, on_stage=cb)
            if not ok or not (out_dir / "notation" / "notation.json").exists():
                raise RuntimeError("管线未产出记谱结果（无准入轨道或分离失败）")

            _set_stage(job_id, "notation", "demo 打包")
            n = json.loads((out_dir / "notation" / "notation.json")
                           .read_text(encoding="utf-8"))
            assemble_demo(out_dir, audio_path, float(n.get("bpm", bpm)))
            with _LOCK:
                _JOBS[job_id].update(
                    {"stage": "done", "pct": 100.0, "label": "完成",
                     "stage_t0": _now()})
            _persist(job_id)
            print(f"[job {job_id}] done -> {out_dir}", flush=True)
    except Exception as e:  # noqa: BLE001（桥上任何失败都要回报给前端）
        import traceback
        traceback.print_exc()
        with _LOCK:
            _JOBS[job_id].update(
                {"stage": "error", "label": f"{e}", "stage_t0": _now()})
    finally:
        _persist(job_id)
        with _LOCK:
            _JOBS[job_id]["end"] = _now()


def _pct_thread() -> None:
    """500ms 插值心跳（分离等无实进度阶段也走得动）。"""
    while True:
        time.sleep(0.5)
        with _LOCK:
            ids = [jid for jid, j in _JOBS.items()
                   if j.get("stage") in STAGE_SPANS]
        for jid in ids:
            _tick_pct(jid)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "x-filename")

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # CORS 预检
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/health":
            self._json(200, {"ok": True})
            return
        m = re.match(r"^/progress/([0-9a-f-]+)$", u.path)
        if m:
            jid = m.group(1)
            with _LOCK:
                j = _JOBS.get(jid)
                if not j:
                    self._json(404, {"stage": "error", "label": "job 不存在",
                                     "pct": 0, "elapsed": 0})
                    return
                _tick_pct_locked(j)  # 已持有 _LOCK
                out = {"stage": j.get("stage", "error"),
                       "pct": float(j.get("pct", 0)),
                       "label": j.get("label", ""),
                       "elapsed": round(_now() - j.get("t0", _now()), 1)}
            self._json(200, out)
            return
        m = re.match(r"^/files/([0-9a-f-]+)/(.+)$", u.path)
        if m:
            jid, rel = m.group(1), unquote(m.group(2))
            base = (JOBS_ROOT / jid).resolve()
            target = (base / rel).resolve()
            if not str(target).startswith(str(base)) or not target.is_file():
                self._json(404, {"error": "not found"})
                return
            data = target.read_bytes()
            ctype = ("application/json" if target.suffix == ".json"
                     else "audio/midi" if target.suffix == ".mid"
                     else "application/octet-stream")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(404, {"error": "bad path"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/transcribe":
            self._json(404, {"error": "bad path"})
            return
        # 单任务串行（GPU 只有一块）：忙时 409
        with _LOCK:
            running = [j for j in _JOBS.values()
                       if j.get("stage") not in ("done", "error")]
        if running:
            self._json(409, {"error": "已有任务在处理（单 GPU 串行），稍后再试"})
            return
        fname = unquote(self.headers.get("x-filename", "upload.flac"))
        ext = Path(fname).suffix or ".flac"
        if ext.lower() not in (".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac"):
            self._json(400, {"error": f"不支持的音频格式 {ext}"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self._json(400, {"error": "空音频"})
            return
        jid = uuid.uuid4().hex[:12]
        out_dir = JOBS_ROOT / jid
        out_dir.mkdir(parents=True, exist_ok=True)
        # 保留原文件名（demo 显示名 = 音频 stem；剔除路径分隔符防逃逸）
        stem = re.sub(r'[\\/:*?"<>|]+', "_", Path(fname).stem) or "upload"
        audio_path = out_dir / f"{stem}{ext}"
        remaining = length
        with open(audio_path, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        if remaining > 0:
            self._json(400, {"error": "上传不完整"})
            return
        # 歌词增强层（人声专项 v2）：前端同选 .lrc → x-lyric-b64 头（urlsafe
        # base64，http.server 单头行上限 64KB，常规 LRC 2-5KB 富余）；落盘为
        # 同名 .lrc，multi_instrument 按约定自动发现
        lrc_b64 = self.headers.get("x-lyric-b64", "").strip()
        lrc_bytes = b""
        if lrc_b64:
            import base64
            try:
                lrc_bytes = base64.urlsafe_b64decode(
                    lrc_b64 + "=" * (-len(lrc_b64) % 4))
            except Exception:
                lrc_bytes = b""
        lrc_path = None
        if lrc_bytes:
            lrc_path = out_dir / f"{stem}.lrc"
            lrc_path.write_bytes(lrc_bytes)
        # 内容去重：同一文件（md5）已完成 → 秒回旧 job，不重付 GPU 成本。
        # 去重键掺入管线源码指纹：代码升级后旧产物自动失效（2026-08-29
        # 吞音案：08-28 中午的中间版本产物 90 音，用户重传同名文件一直
        # 秒回旧结果，无从触发重跑）。歌词内容也入键（换歌词=换产物）。
        import hashlib
        pipe_stamp = hashlib.md5(
            ";".join(f"{p.name}:{p.stat().st_mtime_ns}"
                     for p in sorted((SE / "src").glob("*.py"))
                     ).encode()).hexdigest()[:10]
        md5 = (pipe_stamp + hashlib.md5(audio_path.read_bytes()).hexdigest()
               + (hashlib.md5(lrc_bytes).hexdigest() if lrc_bytes else ""))
        with _LOCK:
            done_same = [
                j for j, v in _JOBS.items()
                if v.get("md5") == md5 and v.get("stage") == "done"]
        if done_same:
            print(f"[job] {fname} 内容命中已完成任务 {done_same[0]}，复用",
                  flush=True)
            self._json(200, {"job": done_same[0], "reused": True})
            return
        with _LOCK:
            _JOBS[jid] = {"stage": "bpm", "pct": 0.5, "label": "已入队",
                          "t0": _now(), "stage_t0": _now(), "md5": md5,
                          "sub_i": 0, "sub_n": 0, "dir": str(out_dir)}
        _persist(jid)
        threading.Thread(target=_run_job, args=(jid, str(audio_path), out_dir),
                         daemon=True).start()
        print(f"[job {jid}] uploaded {fname} ({length} bytes)", flush=True)
        self._json(200, {"job": jid})

    def log_message(self, fmt, *args):  # 静默默认访问日志（阶段日志走 print）
        pass


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    _load_persisted()
    threading.Thread(target=_pct_thread, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[pipeline-server] http://127.0.0.1:{PORT} （Ctrl+C 退出）", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
