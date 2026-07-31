"""eval/eval.py — AMT 评测闭环主入口.

用法:
  python eval/eval.py --model ours  --n 5   # 自训 OaF, 5 首冒烟
  python eval/eval.py --model basic --n 40  # basic-pitch, 40 首
  python eval/eval.py --model both  --n 40  # 双基线对比

对每首曲目:
  MIDI → 渲染合成音频 → 模型转录 (onset/frame probs) → 后处理 → 音符列表
  → 与 GT 计算 帧级F1 / 音符级F1 / Note-with-offset F1
聚合 → 写 JSON 报告到 eval/reports/
"""
import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.dataset import load, sample_midis
from eval.metrics import evaluate

logger = logging.getLogger(__name__)

# 每个模型各自的时间分辨率 (帧索引 → 秒)
HOP = {"ours": 512, "basic": 256}
SR = 22050


# ---------------------------------------------------------------------------
# 转录前端: 显式选择模型, 绕过 transcriber 的自动选择
# ---------------------------------------------------------------------------

def _write_tmp_wav(audio: np.ndarray, path: str, sr: int = SR) -> str:
    import scipy.io.wavfile as wavfile
    wavfile.write(path, sr, (audio * 32767).clip(-32768, 32767).astype(np.int16))
    return path


def _transcribe_ours(audio: np.ndarray, tmp_path: str) -> dict:
    """用自训 OnsetsAndFrames 转录."""
    from src.transcriber import _load_model, _ours_inference
    model = _load_model()
    if model is None:
        raise RuntimeError("VER2.0_Bootstrap.pth not found — cannot use ours model")
    _write_tmp_wav(audio, tmp_path)
    return _ours_inference(model, tmp_path)


def _transcribe_basic(audio: np.ndarray, tmp_path: str) -> dict:
    """用 basic-pitch 转录."""
    from src.transcriber import _basic_pitch_inference
    _write_tmp_wav(audio, tmp_path)
    return _basic_pitch_inference(tmp_path)


def _postprocess(result: dict, audio: np.ndarray) -> tuple:
    """帧/音符级后处理, 返回 (est_frame_probs, est_notes)."""
    from src.frame_post import process_frames
    from src.note_post import refine_notes

    hop = result.get("hop_length", 512)
    sr = result.get("sr", SR)

    candidates = process_frames(
        result["onset_probs"], result["frame_probs"],
        hop_length=hop, sr=sr,
    )
    notes = refine_notes(candidates, audio, sr=sr, hop_length=hop)
    return result["frame_probs"], notes


def run_one(mid_path: str, model_name: str, tmp_path: str,
            use_cache: bool = True) -> dict:
    """评测一首曲目, 返回指标 dict."""
    gt = load(mid_path) if use_cache else load(mid_path, force_render=True)
    audio = gt["audio"]

    if model_name == "ours":
        result = _transcribe_ours(audio, tmp_path)
    else:
        result = _transcribe_basic(audio, tmp_path)

    est_frame_probs, est_notes = _postprocess(result, audio)
    hop = result.get("hop_length", HOP.get(model_name, 512))

    metrics = evaluate(gt, est_frame_probs, est_notes, hop_length=hop, sr=SR)
    metrics["name"] = os.path.basename(mid_path)
    metrics["n_gt_notes"] = int(len(gt["intervals"]))
    metrics["n_est_notes"] = int(len(est_notes))
    return metrics


# ---------------------------------------------------------------------------
# 评测主体
# ---------------------------------------------------------------------------

def evaluate_model(model_name: str, midis: list[str], out_dir: str) -> dict:
    """对指定模型跑完整评测, 返回聚合结果."""
    results = []
    tmp_path = os.path.join(out_dir, "_tmp.wav")

    for i, mid in enumerate(midis, 1):
        try:
            r = run_one(mid, model_name, tmp_path)
            logger.info(
                f"[{i}/{len(midis)}] {r['name'][:40]:42s} "
                f"frame={r['frame_f1']:.3f} note={r['note_f1']:.3f} "
                f"offset={r['offset_f1']:.3f} "
                f"(gt={r['n_gt_notes']}, est={r['n_est_notes']})")
            results.append(r)
        except Exception as exc:
            logger.warning(f"[{i}/{len(midis)}] {os.path.basename(mid)} FAILED: {exc}")

    if not results:
        return {"model": model_name, "error": "all songs failed"}

    def _agg(key):
        vals = [r[key] for r in results]
        return float(np.mean(vals))

    return {
        "model": model_name,
        "n_songs": len(results),
        "frame_f1": _agg("frame_f1"),
        "frame_precision": _agg("frame_precision"),
        "frame_recall": _agg("frame_recall"),
        "note_f1": _agg("note_f1"),
        "note_precision": _agg("note_precision"),
        "note_recall": _agg("note_recall"),
        "offset_f1": _agg("offset_f1"),
        "offset_precision": _agg("offset_precision"),
        "offset_recall": _agg("offset_recall"),
        "per_song": results,
    }


def main():
    ap = argparse.ArgumentParser(description="AMT 评测闭环")
    ap.add_argument("--model", choices=["ours", "basic", "both"], default="ours")
    ap.add_argument("--n", type=int, default=40, help="评测曲目数")
    ap.add_argument("--seed", type=int, default=42, help="采样 seed")
    ap.add_argument("--out-dir", default=None, help="报告输出目录 (默认 eval/reports)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(out_dir, exist_ok=True)

    midis = sample_midis(n=args.n, seed=args.seed)
    logger.info(f"采样 {len(midis)} 首 GiantMIDI (seed={args.seed})")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    report = {
        "timestamp": timestamp,
        "dataset": "GiantMIDI-PIano (synthetic FluidSynth)",
        "n_requested": args.n,
        "seed": args.seed,
        "results": [],
    }

    models = ["ours", "basic"] if args.model == "both" else [args.model]
    for m in models:
        logger.info(f"\n=== 评测模型: {m} ===")
        agg = evaluate_model(m, midis, out_dir)
        report["results"].append(agg)

    report_path = os.path.join(out_dir, f"report_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\n报告已写入: {report_path}")

    # 终端摘要
    for agg in report["results"]:
        logger.info(
            f"  {agg['model']}: frame_f1={agg.get('frame_f1', float('nan')):.3f} "
            f"note_f1={agg.get('note_f1', float('nan')):.3f} "
            f"offset_f1={agg.get('offset_f1', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
