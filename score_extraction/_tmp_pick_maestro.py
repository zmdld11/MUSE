"""复现 gen_f2_piano.py 的选曲（只依赖 csv + seed），输出待下载文件清单。"""
import csv
import glob
import random

MAESTRO = r"D:/program_project/MUSE/score_extraction/data/maestro/maestro-v3.0.0"
OTHER_GLOB = r"D:/program_project/MUSE/data/moisesdb_guitar_pilot_v1/train/*/Other.wav"
SEED, N, MAX_SEC = 20260822, 40, 90.0

rows = list(csv.DictReader(open(MAESTRO + "/maestro-v3.0.0.csv", encoding="utf-8")))
test = [r for r in rows if r["split"] == "test" and float(r["duration"]) >= MAX_SEC]
rng = random.Random(SEED)
rng.shuffle(test)
test = test[:N]
others = sorted(glob.glob(OTHER_GLOB))
print(f"test pool {len([r for r in rows if r['split']=='test'])} -> >=90s {len([r for r in rows if r['split']=='test' and float(r['duration'])>=MAX_SEC])}, picked {len(test)}, accompaniment pool {len(others)}")

with open(MAESTRO + "/_fetch_list.txt", "w", encoding="utf-8") as f:
    for r in test:
        f.write(r["audio_filename"] + "\n")
        f.write(r["midi_filename"] + "\n")
for r in test[:5]:
    print(r["audio_filename"], r["duration"])
