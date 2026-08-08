"""用 VER4.0 跑卡农 + 夜之向日葵, 输出到带版本号的文件夹.

用法: python run_ver4_songs.py
"""
import os
import sys

os.environ["MUSE_MODEL_PATH"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model", "VER4.0_best.pth")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

MUSIC = r"D:\program_project\MUSE\music"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

run_pipeline(
    os.path.join(MUSIC, "Variations On The Canon By Pachelbel - George Winston.flac"),
    os.path.join(OUT, "canon_ver4"),
)
run_pipeline(
    os.path.join(MUSIC, "夜の向日葵 - 松本文紀.flac"),
    os.path.join(OUT, "himawari_ver4"),
)
print("BOTH SONGS DONE", flush=True)
