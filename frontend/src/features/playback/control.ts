/**
 * 播放控制门面：按音源（MIDI 合成 / 原曲音频 / 叠加）把操作路由到引擎。
 * 叠加模式 = 双引擎同位播放（时间轴以原曲为准，无原曲时以 MIDI 为准）。
 * UI 层只 import 这里的函数，不直接碰引擎。
 */
import { audioEngine } from "./audioEngine";
import { midiEngine } from "./midiEngine";
import { usePlayerStore } from "@/entities/project/store";

export const SPEED_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

export interface PlayerEngine {
  readonly duration: number;
  readonly currentTime: number;
  readonly isPlaying: boolean;
  onEnded: (() => void) | null;
  play(): void;
  pause(): void;
  seek(t: number): void;
  setRate(r: number): void;
}

/** 当前音源实际要驱动的引擎集合（mix 按项目内容退化） */
function activeEngines(): PlayerEngine[] {
  const s = usePlayerStore.getState();
  const p = s.project;
  if (s.playbackSource === "mix") {
    const list: PlayerEngine[] = [];
    if (p && p.tracks.length > 0) list.push(midiEngine);
    if (p?.hasAudio) list.push(audioEngine);
    return list;
  }
  return [s.playbackSource === "midi" ? midiEngine : audioEngine];
}

/** 时间真源：有原曲以原曲为准（叠加模式双引擎以它对表） */
export function activeEngine(): PlayerEngine {
  const engs = activeEngines();
  return engs.find((e) => e === audioEngine) ?? engs[0];
}

let midiLoadPromise: Promise<void> | null = null;

/** 换曲后使 MIDI 音色缓存失效 */
export function resetMidiCache(): void {
  midiLoadPromise = null;
}

async function ensureMidiLoaded(): Promise<void> {
  if (!midiLoadPromise) {
    const s = usePlayerStore.getState();
    if (!s.project || s.project.tracks.length === 0) {
      throw new Error("没有可播放的 MIDI 轨道");
    }
    midiLoadPromise = midiEngine
      .load(s.project.tracks, s.project.duration, (msg) => s.setLoading(msg))
      .catch((e: unknown) => {
        midiLoadPromise = null;
        throw e;
      });
  }
  return midiLoadPromise;
}

/** 叠加模式降增益防削波；独奏模式恢复 */
function applyMixGain(): void {
  const mix = usePlayerStore.getState().playbackSource === "mix";
  audioEngine.setMasterGain(mix ? 0.8 : 1);
  midiEngine.setMasterGain(mix ? 0.85 : 1);
}

async function prepareEngines(): Promise<boolean> {
  const s = usePlayerStore.getState();
  const engs = activeEngines();
  try {
    if (engs.includes(midiEngine) && !midiEngine.loaded) {
      s.setLoading("准备 MIDI 音色…");
      await ensureMidiLoaded();
    }
  } catch (e) {
    usePlayerStore.getState().setLoading(null);
    window.alert(`音色加载失败：${e instanceof Error ? e.message : e}`);
    return false;
  } finally {
    usePlayerStore.getState().setLoading(null);
  }
  return true;
}

export async function togglePlay(): Promise<void> {
  const engs = activeEngines();
  if (engs.length === 0) return;
  if (!(await prepareEngines())) return;

  applyMixGain();
  const anyPlaying = engs.some((e) => e.isPlaying);
  if (anyPlaying) {
    for (const e of engs) e.pause();
    usePlayerStore.getState().setPlaying(false);
    usePlayerStore.getState().setCurrentTime(activeEngine().currentTime);
  } else {
    const ref = activeEngine();
    const restart =
      ref.duration > 0 && ref.currentTime >= ref.duration - 0.05;
    for (const e of engs) {
      if (restart) e.seek(0); // 播完后再按 = 从头播
      e.play();
    }
    usePlayerStore.getState().setPlaying(true);
    usePlayerStore.getState().setCurrentTime(ref.currentTime);
  }
}

/** 任一引擎播完：全部停下（叠加模式双引擎几乎同时到尾） */
export function handleEngineEnded(): void {
  const s = usePlayerStore.getState();
  for (const e of activeEngines()) e.pause();
  s.setPlaying(false);
  s.setCurrentTime(activeEngine().duration);
}

/** 音源切换（保持播放位置；原在播则无缝续播） */
export async function setPlaybackSource(
  src: "midi" | "audio" | "mix",
): Promise<void> {
  const s = usePlayerStore.getState();
  if (s.playbackSource === src) return;
  // mix 退化：没有原曲按 midi 处理，没有 MIDI 轨按 audio 处理
  let target = src;
  if (src === "mix") {
    if (!s.project?.hasAudio) target = "midi";
    else if (!s.project?.tracks.length) target = "audio";
  } else if (src === "audio" && !s.project?.hasAudio) {
    return;
  } else if (src === "midi" && !s.project?.tracks.length) {
    return;
  }
  if (s.playbackSource === target) return;

  const wasPlaying = activeEngines().some((e) => e.isPlaying);
  const pos = activeEngine().currentTime;
  for (const e of activeEngines()) e.pause();
  s.setPlaying(false);
  s.setPlaybackSource(target);

  if (!(await prepareEngines())) return;

  applyMixGain();
  const engs = activeEngines();
  for (const e of engs) e.seek(pos);
  if (wasPlaying && engs.length > 0) {
    for (const e of engs) e.play();
    usePlayerStore.getState().setPlaying(true);
  }
  usePlayerStore.getState().setCurrentTime(activeEngine().currentTime);
}

function seekAll(t: number): void {
  for (const e of activeEngines()) e.seek(t);
  usePlayerStore.getState().setCurrentTime(activeEngine().currentTime);
}

export function seekBy(delta: number): void {
  const eng = activeEngine();
  seekAll(eng.currentTime + delta);
}

export function seekTo(t: number): void {
  seekAll(t);
}

export function setPlaybackRate(r: number): void {
  audioEngine.setRate(r);
  midiEngine.setRate(r);
  usePlayerStore.getState().setRate(r);
}
