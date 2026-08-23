import { Midi } from "@tonejs/midi";
import type { Note, NotationMeta, Project, Track } from "@/entities/project/types";
import { usePlayerStore } from "@/entities/project/store";
import { audioEngine } from "@/features/playback/audioEngine";
import {
  CLASS_FAMILY,
  GM_NAMES,
  filenameHint,
  gmFamily,
} from "@/shared/theme/instrumentColors";
import {
  isAudioFilename,
  isTauri,
  pickDirectory,
  tauriReadBytes,
  tauriScanDir,
} from "./fileAccess";
import { resetMidiCache } from "@/features/playback/control";

interface RawProject {
  name: string;
  audioBytes?: ArrayBuffer;
  mids: { name: string; bytes: ArrayBuffer }[];
  infoText?: string;
  notation?: NotationMeta;
}

/** MIDI 字节 → 轨道列表（GM 音色 + 文件名提示双路判定族与颜色） */
function tracksFromMidi(midName: string, bytes: ArrayBuffer): Track[] {
  const midi = new Midi(bytes);
  const hint = filenameHint(midName);
  const tracks: Track[] = [];
  midi.tracks.forEach((tr, i) => {
    if (tr.notes.length === 0) return;
    const program = tr.instrument?.number ?? 0;
    const isDrum = tr.channel === 9;
    const family = isDrum ? "perc" : gmFamily(program);
    const name = isDrum ? "鼓组" : (GM_NAMES[program] ?? `音轨 ${i + 1}`);
    // 管线产物（guitar.mid）以文件名提示为准；多音轨混排时仅首轨覆盖
    const override = hint && tracks.length === 0 ? hint : null;
    const notes: Note[] = tr.notes.map((n) => ({
      onset: n.time,
      offset: n.time + Math.max(n.duration, 0.02),
      pitch: n.midi,
      velocity: n.velocity,
    }));
    tracks.push({
      id: `${midName}:${i}`,
      name: override?.name ?? name,
      family: override?.family ?? family,
      program,
      isDrum,
      notes,
      muted: false,
      solo: false,
    });
  });
  return tracks;
}

/** notes.json（阶段二）的 instrument_class 直接映射族；现在先占位，schema 落地后接入 */
export function familyFromInstrumentClass(cls: string) {
  return CLASS_FAMILY[cls] ?? "fx";
}

/** notation.json（记谱层产物）→ 摘要元信息；谱面文件按约定路径拼接 */
export function parseNotationMeta(
  text: string,
  base: { kind: "url" | "fs"; baseUrl: string },
): NotationMeta {
  const n = JSON.parse(text) as {
    key?: string;
    analysis?: { summary?: string };
    tracks?: {
      instrument_class: string;
      events: { bar: number; onset_sec: number }[];
    }[];
  };
  const tracks = (n.tracks ?? []).map((t) => ({
    instrumentClass: t.instrument_class,
    minBar: t.events.length ? t.events[0].bar : 0, // events 已按 onset 排序
    firstOnsetSec: t.events.length ? t.events[0].onset_sec : 0,
    noteCount: t.events.length,
  }));
  return { ...base, key: n.key, analysis: n.analysis, tracks };
}

async function buildAndLoad(raw: RawProject): Promise<Project> {
  const store = usePlayerStore.getState();
  store.setLoading("解析 MIDI…");
  const tracks = raw.mids.flatMap((m) => tracksFromMidi(m.name, m.bytes));
  let bpm: number | undefined;
  try {
    if (raw.infoText) {
      const info = JSON.parse(raw.infoText);
      if (typeof info.bpm === "number") bpm = info.bpm;
    }
  } catch {
    /* info.json 缺失或损坏不致命 */
  }

  let duration = tracks.reduce(
    (mx, t) => Math.max(mx, ...t.notes.map((n) => n.offset)),
    0,
  );
  let hasAudio = false;
  if (raw.audioBytes) {
    store.setLoading("解码音频…");
    duration = await audioEngine.load(raw.audioBytes);
    hasAudio = true;
  }

  const project: Project = {
    name: raw.name,
    bpm,
    duration,
    tracks,
    hasAudio,
    notation: raw.notation,
  };
  store.setProject(project);
  resetMidiCache(); // 换曲后音色缓存失效（轨道集合变了）
  store.setLoading(null);
  return project;
}

/** Tauri：选择目录 → 扫描 → 加载（管线输出目录或纯音频文件夹） */
export async function openDirectoryProject(): Promise<void> {
  const dir = await pickDirectory();
  if (!dir) return;
  const scan = await tauriScanDir(dir);
  if (!scan.audio && scan.mids.length === 0) {
    usePlayerStore.getState().setLoading(null);
    throw new Error("目录里没找到音频或 MIDI 文件");
  }
  const name = dir.split(/[\\/]/).pop() ?? dir;
  const raw: RawProject = { name, mids: [], };
  if (scan.audio) raw.audioBytes = await tauriReadBytes(scan.audio);
  for (const m of scan.mids) {
    raw.mids.push({ name: m.split(/[\\/]/).pop() ?? m, bytes: await tauriReadBytes(m) });
  }
  if (scan.info) {
    const infoBytes = await tauriReadBytes(scan.info);
    raw.infoText = new TextDecoder().decode(infoBytes);
  }
  // 记谱层产物（管线 notation/ 目录，缺失不致命）
  try {
    const nBytes = await tauriReadBytes(`${dir}/notation/notation.json`);
    raw.notation = parseNotationMeta(new TextDecoder().decode(nBytes), {
      kind: "fs",
      baseUrl: dir,
    });
  } catch {
    /* 无记谱输出 */
  }
  await buildAndLoad(raw);
}

/** Web（浏览器开发/将来的网页版）：从 File 列表加载 */
export async function loadWebFiles(files: File[]): Promise<void> {
  const audio = files.find((f) => isAudioFilename(f.name));
  const mids = files.filter((f) => f.name.toLowerCase().endsWith(".mid"));
  const info = files.find((f) => f.name === "info.json");
  const raw: RawProject = {
    name: audio?.name.replace(/\.[^.]+$/, "") ?? (mids[0]?.name.replace(/\.[^.]+$/, "") ?? "未命名"),
    mids: [],
  };
  if (audio) raw.audioBytes = await audio.arrayBuffer();
  for (const m of mids) raw.mids.push({ name: m.name, bytes: await m.arrayBuffer() });
  if (info) raw.infoText = await info.text();
  await buildAndLoad(raw);
}

/** 演示曲目（public/demo，Tauri 与浏览器都能加载；index.json 清单优先，多乐器轨） */
export async function loadDemoProject(): Promise<void> {
  let midNames = ["guitar.mid"];
  try {
    const r = await fetch("/demo/index.json");
    if (r.ok) {
      const idx = (await r.json()) as { mids?: string[] };
      if (Array.isArray(idx.mids) && idx.mids.length > 0) midNames = idx.mids;
    }
  } catch {
    /* 清单缺失时退回单轨 */
  }
  const [audioR, infoR, ...midRs] = await Promise.all([
    fetch("/demo/05_kyomu_vocal.flac"),
    fetch("/demo/info.json").catch(() => null),
    ...midNames.map((name) => fetch(`/demo/${name}`).catch(() => null)),
  ]);
  if (!audioR.ok) throw new Error("演示数据缺失（public/demo）");
  const mids = midNames
    .map((name, i) => ({ name, res: midRs[i] }))
    .filter((m) => m.res && m.res.ok);
  if (mids.length === 0) throw new Error("演示 MIDI 缺失（public/demo）");
  // 记谱层产物（public/demo/notation）
  let notation: NotationMeta | undefined;
  try {
    const nr = await fetch("/demo/notation/notation.json");
    if (nr.ok) {
      notation = parseNotationMeta(await nr.text(), { kind: "url", baseUrl: "/demo" });
    }
  } catch {
    /* 无记谱输出 */
  }
  await buildAndLoad({
    name: "虚無の先で愛を見つける（演示）",
    audioBytes: await audioR.arrayBuffer(),
    mids: await Promise.all(
      mids.map(async (m) => ({ name: m.name, bytes: await m.res!.arrayBuffer() })),
    ),
    infoText: infoR?.ok ? await infoR.text() : undefined,
    notation,
  });
}

export { isTauri };
