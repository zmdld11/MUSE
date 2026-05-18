"""批量训练全部 10 个乐器的二分类器"""
import subprocess, sys, time, os

os.chdir(r"D:\program_project\MUSE\instrument_recognition")
python = r"D:\program_project\MUSE\instrument_recognition\env\python.exe"

instruments = [
    'acoustic_guitar', 'cello', 'drum_set', 'electric_bass',
    'electric_guitar', 'flute', 'piano', 'singer',
    'synthesizer', 'violin',
]

for inst in instruments:
    print(f"\n{'='*50}")
    print(f"训练: {inst}")
    print(f"{'='*50}")
    start = time.time()
    result = subprocess.run(
        [python, "-m", "src.btrain", "--instrument", inst, "--epochs", "30"],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  [{inst}] 失败! returncode={result.returncode}")
    else:
        print(f"  [{inst}] 完成，耗时 {time.time()-start:.0f}s")
