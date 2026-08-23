import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** 黑键判断（MIDI 音高 → 是否钢琴黑键） */
export function isBlackKey(pitch: number): boolean {
  const pc = pitch % 12;
  return pc === 1 || pc === 3 || pc === 6 || pc === 8 || pc === 10;
}
