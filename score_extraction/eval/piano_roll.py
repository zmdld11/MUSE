"""eval/piano_roll.py — 钢琴卷帘可视化 + 音符清单导出 + 人工修正闭环.

真实录音的 MIDI 输出难以直接听辨每个音符, 提供:
  1. 钢琴卷帘图 (PNG): 时间 × 音高, 每个音符一根条, 可叠加 GT 参考
  2. 音符清单 CSV: onset/offset/pitch/confidence/onset_prob, 供人工逐条核对
  3. 修正重生成: 用户删除 CSV 行 → 重新生成 MIDI

用法:
  python -m eval.piano_roll view --midi output/夜の向日葵 - 松本文紀/piano_stabilized.mid \
      --out output/夜の向日葵 - 松本文紀/piano_roll.png [--gt output/himawari_reference_E/himawari_reference_E.mid] [--shift 0.85]
  python -m eval.piano_roll export --midi <x.mid> --out <x_notes.csv>
  python -m eval.piano_roll rebuild --csv <x_notes.csv> --out <fixed.mid> --bpm 82
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def load_notes(midi_path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append({
                "onset": float(n.start), "offset": float(n.end),
                "pitch": int(n.pitch),
            })
    notes.sort(key=lambda n: n["onset"])
    return notes


def view(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    notes = load_notes(args.midi)
    fig, ax = plt.subplots(figsize=(max(20, len(notes) * 0.08), 10))

    # est 层: 蓝色条
    for n in notes:
        ax.add_patch(Rectangle((n["onset"], n["pitch"] - 0.35),
                               n["offset"] - n["onset"], 0.7,
                               facecolor="#4C72B0", edgecolor="none", alpha=0.85))

    # GT 层 (可选): 红色轮廓
    if args.gt:
        gt = load_notes(args.gt)
        for n in gt:
            on = n["onset"] + (args.shift or 0.0)
            ax.add_patch(Rectangle((on, n["pitch"] - 0.35),
                                   n["offset"] - n["onset"], 0.7,
                                   facecolor="none", edgecolor="#C44E52",
                                   linewidth=0.8, alpha=0.9))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_ylim(20, 110)
    ax.set_title(f"{os.path.basename(args.midi)}  ({len(notes)} notes)"
                 + ("  — 红框 = GT 参考" if args.gt else ""))
    # 音高刻度标音名
    ticks = np.arange(24, 108, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t} {note_name(t)}" for t in ticks])
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"卷帘图已保存: {args.out}  ({len(notes)} 个音符)")


def export(args):
    notes = load_notes(args.midi)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "onset_s", "offset_s", "dur_s", "pitch_midi", "note_name"])
        for i, n in enumerate(notes):
            w.writerow([i, f"{n['onset']:.3f}", f"{n['offset']:.3f}",
                        f"{n['offset'] - n['onset']:.3f}", n["pitch"],
                        note_name(n["pitch"])])
    print(f"音符清单已导出: {args.out}  ({len(notes)} 行)")
    print("人工修正: 删除不想要的音符行, 然后跑 rebuild 重新生成 MIDI")


def rebuild(args):
    import pretty_midi
    rows = []
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    pm = pretty_midi.PrettyMIDI(initial_tempo=args.bpm)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for r in rows:
        note = pretty_midi.Note(
            velocity=70,
            pitch=int(r["pitch_midi"]),
            start=float(r["onset_s"]),
            end=float(r["offset_s"]),
        )
        inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(args.out)
    print(f"已生成: {args.out}  ({len(rows)} 音符)")


def main():
    ap = argparse.ArgumentParser(description="钢琴卷帘 + 音符清单 + 人工修正")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_view = sub.add_parser("view", help="生成钢琴卷帘 PNG")
    p_view.add_argument("--midi", required=True)
    p_view.add_argument("--out", required=True)
    p_view.add_argument("--gt", default=None, help="叠加 GT 参考 (红框)")
    p_view.add_argument("--shift", type=float, default=None, help="GT 时间偏移")
    p_view.set_defaults(func=view)

    p_exp = sub.add_parser("export", help="导出音符清单 CSV")
    p_exp.add_argument("--midi", required=True)
    p_exp.add_argument("--out", required=True)
    p_exp.set_defaults(func=export)

    p_reb = sub.add_parser("rebuild", help="从修正后的 CSV 重新生成 MIDI")
    p_reb.add_argument("--csv", required=True)
    p_reb.add_argument("--out", required=True)
    p_reb.add_argument("--bpm", type=float, default=82.0)
    p_reb.set_defaults(func=rebuild)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
