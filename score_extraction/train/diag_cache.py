"""诊断 30s 缓存: 找出未缓存曲目, 排查卡死点."""
import os
import sys
import time

sys.path.insert(0, "d:/program_project/MUSE/score_extraction")

from train.maestro_dataset import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CSV,
    DEFAULT_MAESTRO_DIR,
    _cache_path,
    load_maestro_rows,
)

rows = [r for r in load_maestro_rows(DEFAULT_CSV) if r["split"] == "train"]
missing = []
for r in rows:
    cp = _cache_path(
        os.path.join(DEFAULT_MAESTRO_DIR, r["midi_filename"]),
        os.path.join(DEFAULT_MAESTRO_DIR, r["audio_filename"]),
        30,
        DEFAULT_CACHE_DIR,
    )
    if not os.path.exists(cp):
        missing.append(r)

print(f"未缓存: {len(missing)} 首")
for r in missing[:15]:
    wav = os.path.join(DEFAULT_MAESTRO_DIR, r["audio_filename"])
    sz = os.path.getsize(wav) if os.path.exists(wav) else -1
    print(f"  {r['year']} dur={float(r['duration']):.0f}s wav={sz/1e6:.0f}MB {r['midi_filename'][:70]}")
