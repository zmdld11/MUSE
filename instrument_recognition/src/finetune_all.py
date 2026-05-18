"""批量运行 Stage 2 真实混音微调"""
import subprocess, time, os, sys

os.chdir(r"D:\program_project\MUSE\instrument_recognition")
python = sys.executable

instruments = [
    'acoustic_guitar', 'cello', 'drum_set', 'electric_bass',
    'electric_guitar', 'flute', 'piano', 'singer',
    'synthesizer', 'violin',
]

for inst in instruments:
    print(f"\n{'='*60}")
    print(f"Stage 2 微调: {inst}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(
        [python, "-m", "src.bfinetune", "--instrument", inst, "--epochs", "15"],
    )
    if result.returncode != 0:
        print(f"  [{inst}] 失败!")
    else:
        print(f"  [{inst}] 完成，耗时 {time.time()-start:.0f}s")
