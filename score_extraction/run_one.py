"""一键流 CLI（P6 MVP，Phase-1 结项工程产物）。

单命令：音频 → VER-SEP 3.0b 分离 + ia-amt 转写 → notes.json/谱（量化）
→ score_mid（谱面派生 MIDI）→ 可选组装前端 demo 包（--demo）。

用法：
  python run_one.py <audio> [--out DIR] [--demo] [--sep off|versep_guitar|versep_demucs]
  python run_one.py --demo-sync <管线输出目录>…  # 重装 demo 包并同步（可多目录=曲库）

输出目录结构（= 前端可直接加载的 demo 目录）：
  <out>/notes.json  <out>/*.mid(原始)  <out>/notation/{notation.json,solo/,score_mid/}
  demo 组装另写：index.json（→score_mid 清单）、info.json、音频副本。
  --demo-sync 多目录时 frontend/public/demo 变多曲库（每曲一子目录 +
  根 index.json v2 清单，前端出现切歌下拉）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SE = Path(__file__).resolve().parent
sys.path.insert(0, str(SE))

FRONTEND_DEMO = SE.parent / "frontend" / "public" / "demo"


def detect_bpm(audio: str):
    from src.bpm_detect import detect_bpm as _det
    from src.config import Config
    bpm = _det(audio) or Config().DEFAULT_BPM
    if bpm > 120:
        bpm = round(bpm / 2, 1)
        print(f"[run_one] bpm>120 → 减半启发式: {bpm}")
    return bpm


def _write_browser_audio(src: Path, dst: Path) -> None:
    """demo/试用装音频副本走 soundfile 重编码为干净 PCM_16 FLAC。

    根因（2026-08-30 浏览器实测）：部分源 flac（夏日/小丑，元数据含大
    PICTURE 块等）Chrome decodeAudioData 直接抛 "Unable to decode audio
    data"——44.1k/16bit 与能解的 canon 完全同规格，位深/采样率无关。
    重编码产出规范布局文件（实测可解）；源文件永不改动。非 flac 原样拷贝。
    """
    if src.suffix.lower() != ".flac":
        shutil.copyfile(src, dst)
        return
    try:
        import soundfile as sf
        y, sr = sf.read(str(src), dtype="float32", always_2d=True)
        sf.write(str(dst), y, sr, subtype="PCM_16", format="FLAC")
    except Exception as e:  # noqa: BLE001
        print(f"[run_one] !! 音频重编码失败（{e}），回退原样拷贝")
        shutil.copyfile(src, dst)


def assemble_demo(out_dir: Path, audio: str, bpm: float) -> None:
    """组装前端 demo 包：index.json 指向 score_mid（谱面同源），info.json，音频副本。"""
    notation_p = out_dir / "notation" / "notation.json"
    if not notation_p.exists():
        print("[run_one] 无 notation.json，跳过 demo 组装")
        return
    notation = json.loads(notation_p.read_text(encoding="utf-8"))
    # notation.json 的 score_mids 相对 notation/ 目录；demo 根是 out_dir，
    # 前端 fetch(/demo/<name>) 需要 notation/ 前缀（2026-08-27 卡"解析 MIDI"
    # 的根因：无前缀路径被 Vite SPA 回退成 index.html，Midi 解析抛异常）
    mids = [f"notation/{m}" for m in (notation.get("score_mids") or [])]
    if not mids:
        print("[run_one] score_mid 为空（无准入轨道？），demo 组装中止")
        return
    # 处理前对照（曲内 A/B 切换用）：根目录原始转写 mid 中、谱面准入类的
    # 子集——对照只隔离记谱层差异（时值/量化），不掺稀疏轨删除效应
    admitted = {Path(m).stem for m in mids}
    raw_mids = sorted(p.name for p in out_dir.glob("*.mid")
                      if p.stem in admitted)
    # 显示名用音频文件名（比目录 id 可读；notes.json 的 song 字段不可靠）
    (out_dir / "index.json").write_text(
        json.dumps({"mids": mids, "raw_mids": raw_mids,
                    "audio": Path(audio).name,
                    "name": Path(audio).stem}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "info.json").write_text(
        json.dumps({"bpm": bpm, "time_signature": notation.get("time_signature", "4/4"),
                    "key": notation.get("key"),
                    "source": "MUSE one-click pipeline (VER-SEP 3.0b + ia-amt + notation v2)"},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    audio_dst = out_dir / Path(audio).name
    if not audio_dst.exists():
        _write_browser_audio(Path(audio), audio_dst)
    print(f"[run_one] demo 包组装完成：{len(mids)} 条 score_mid 轨 + index/info")


def sync_to_frontend(out_dirs: list[Path]) -> None:
    """管线输出目录（≥1 个）→ frontend/public/demo 多曲库。

    每曲一个子目录（index/info/notation/音频，路径相对子目录根，与单曲
    结构一致）；根 index.json 写 v2 清单 {version:2, songs:[{id,name,dir,
    audio,mids}]}，前端据此渲染切歌下拉。单曲时结构相同（清单长度 1）。
    """
    FRONTEND_DEMO.mkdir(parents=True, exist_ok=True)
    for child in FRONTEND_DEMO.iterdir():  # demo 目录是 gitignored 派生物
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    songs: list[dict] = []
    for out_dir in out_dirs:
        idx_p = out_dir / "index.json"
        if not idx_p.exists():
            print(f"[run_one] !! {out_dir} 无 index.json（demo 未组装？），跳过")
            continue
        idx = json.loads(idx_p.read_text(encoding="utf-8"))
        sid = out_dir.name
        dst = FRONTEND_DEMO / sid
        dst.mkdir()
        for item in ("index.json", "info.json", "notes.json"):
            if (out_dir / item).exists():
                shutil.copyfile(out_dir / item, dst / item)
        if (out_dir / "notation").exists():
            shutil.copytree(out_dir / "notation", dst / "notation")
        for f in out_dir.glob("*.mid"):  # 转写原始 mid（对照用）
            shutil.copyfile(f, dst / f.name)
        audio = idx.get("audio")
        if audio and (out_dir / audio).exists():
            _write_browser_audio(out_dir / audio, dst / audio)
        songs.append({"id": sid, "name": idx.get("name", sid), "dir": sid,
                      "audio": audio, "mids": idx.get("mids", []),
                      "raw_mids": idx.get("raw_mids", [])})
    (FRONTEND_DEMO / "index.json").write_text(
        json.dumps({"version": 2, "songs": songs}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[run_one] 已同步 {len(songs)} 首到 {FRONTEND_DEMO}（前端刷新即见）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio", nargs="?", help="输入音频（flac/wav/mp3）")
    ap.add_argument("--out", default=None, help="输出目录（默认 output/one_click/<音频名>）")
    ap.add_argument("--demo", action="store_true", help="组装 demo 包并同步 frontend/public/demo")
    ap.add_argument("--sep", default="versep_guitar",
                    choices=["off", "versep_guitar", "versep_demucs"])
    ap.add_argument("--demo-sync", metavar="DIR", nargs="+",
                    help="只重装+同步 demo 包（≥1 个管线输出目录；多个=曲库），不重跑管线")
    args = ap.parse_args()

    if args.demo_sync:
        for d in args.demo_sync:
            d = Path(d)
            if not (d / "notation" / "notation.json").exists():
                # 无记谱产物（如 raw 对照包）：跳过重组装，沿用已有 index.json
                if not (d / "index.json").exists():
                    print(f"[run_one] !! {d} 无 notation.json 且无 index.json，跳过")
                continue
            n = json.loads((d / "notation" / "notation.json").read_text(encoding="utf-8"))
            audio_in_dir = next(iter(list(d.glob("*.flac")) + list(d.glob("*.wav"))), None)
            if audio_in_dir is None:
                print(f"[run_one] !! {d} 无音频副本，跳过")
                continue
            assemble_demo(d, str(audio_in_dir), float(n["bpm"]))
        sync_to_frontend([Path(d) for d in args.demo_sync])
        return 0
    if not args.audio:
        ap.error("需要 <audio> 或 --demo-sync <DIR>")

    audio = str(Path(args.audio).resolve())
    out_dir = Path(args.out) if args.out else SE / "output" / "one_click" / \
        Path(audio).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # env 必须在 BPM 检测之前设置：detect_bpm 的 import 链会实例化
    # src.config 的模块级单例，晚设的 MUSE_MULTI_SEPARATION 永远不生效
    # （潜伏 bug：--sep off/demucs 一直被静默吞掉，2026-08-28 A/B 才暴露）
    os.environ["MUSE_MULTI_INSTRUMENT"] = "1"
    os.environ["MUSE_MULTI_SEPARATION"] = args.sep

    print(f"[run_one] 1/3 BPM 检测：{audio}")
    bpm = detect_bpm(audio)
    print(f"[run_one] bpm = {bpm}")

    print("[run_one] 2/3 多乐器管线（分离+转写+清洗+记谱）…")
    from src.config import Config  # noqa: F401  （env 已就位后 import 生效）
    from src.multi_instrument import run_multi_instrument
    ok = run_multi_instrument(audio, str(out_dir), bpm)
    if not ok:
        print("[run_one] 管线失败（见上日志）")
        return 1
    if not (out_dir / "notation" / "notation.json").exists():
        print("[run_one] 记谱层未产出（无准入轨道？）")
        return 1

    print("[run_one] 3/3 demo 包组装")
    # info.json 用记谱层精修后的 bpm（rubato 曲目与谱面/播放一致）
    try:
        _n = json.loads((out_dir / "notation" / "notation.json")
                        .read_text(encoding="utf-8"))
        bpm = float(_n.get("bpm", bpm))
    except Exception:
        pass
    assemble_demo(out_dir, audio, bpm)
    if args.demo:
        sync_to_frontend([out_dir])
    print(f"[run_one] 完成 → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
