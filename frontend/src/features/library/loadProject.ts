import { Midi } from "@tonejs/midi";
import type { Note, NotationMeta, Project, Track } from "@/entities/project/types";
import { usePlayerStore } from "@/entities/project/store";
import { audioEngine } from "@/features/playback/audioEngine";
import { midiEngine } from "@/features/playback/midiEngine";
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

/** MIDI 字节 → 轨道列表（GM 音色 + 文件名提示双路判定族与颜色）；
 * 单个文件解析失败跳过不致命（曾因 fetch 到 SPA 回退 HTML 抛异常卡死
 * loading——2026-08-27） */
function tracksFromMidi(midName: string, bytes: ArrayBuffer): Track[] {
  let midi: Midi;
  try {
    midi = new Midi(bytes);
  } catch (e) {
    console.warn(`[load] MIDI 解析失败，跳过 ${midName}`, e);
    return [];
  }
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
    time_map?: [number, number][];
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
  return {
    ...base,
    key: n.key,
    analysis: n.analysis,
    tracks,
    timeMap: Array.isArray(n.time_map) ? n.time_map : undefined,
  };
}

async function buildAndLoad(raw: RawProject): Promise<Project> {
  const store = usePlayerStore.getState();
  // 换曲先停所有引擎并卸载旧曲 MIDI 音色：audioEngine.load 只停音频源；
  // midiEngine 旧曲音色驻留会让 loaded=true 跳过重载，且旧轨 id 对不上
  // 被可见性开关全静音（2026-08-27 两连修："切歌放旧曲"/"没加载音色"）
  audioEngine.pause();
  audioEngine.seek(0);
  midiEngine.unload();
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

/** 演示曲目（public/demo；v2 多曲清单 {version:2,songs:[…]} 每曲一子目录，
 * 旧单曲格式回退根目录加载。songId 省略 = 清单第一首） */
export async function loadDemoProject(songId?: string): Promise<void> {
  let base = "/demo";
  let midNames = ["guitar.mid"];
  let audioName = "05_kyomu_vocal.flac";
  let songName = "虚無の先で愛を見つける（演示）";
  try {
    const r = await fetch("/demo/index.json");
    if (r.ok) {
      const idx = (await r.json()) as {
        version?: number;
        songs?: { id: string; name?: string; dir?: string; audio?: string;
                  mids?: string[] }[];
        mids?: string[];
        audio?: string;
        name?: string;
      };
      if (Array.isArray(idx.songs) && idx.songs.length > 0) {
        const pick =
          idx.songs.find((s) => s.id === songId) ?? idx.songs[0];
        base = `/demo/${pick.dir ?? pick.id}`;
        if (Array.isArray(pick.mids) && pick.mids.length > 0) midNames = pick.mids;
        if (typeof pick.audio === "string" && pick.audio) audioName = pick.audio;
        if (typeof pick.name === "string" && pick.name) songName = pick.name;
        usePlayerStore.getState().setDemoSongs(
          idx.songs.map((s) => ({ id: s.id, name: s.name ?? s.id })),
          pick.id,
        );
      } else {
        if (Array.isArray(idx.mids) && idx.mids.length > 0) midNames = idx.mids;
        if (typeof idx.audio === "string" && idx.audio) audioName = idx.audio;
        if (typeof idx.name === "string" && idx.name) songName = idx.name;
        usePlayerStore.getState().setDemoSongs([], null);
      }
    }
  } catch {
    /* 清单缺失时退回单轨 */
  }
  const [audioR, infoR, ...midRs] = await Promise.all([
    fetch(`${base}/${audioName}`),
    fetch(`${base}/info.json`).catch(() => null),
    ...midNames.map((name) => fetch(`${base}/${name}`).catch(() => null)),
  ]);
  if (!audioR.ok) throw new Error("演示数据缺失（public/demo）");
  const mids = midNames
    .map((name, i) => ({ name, res: midRs[i] }))
    .filter((m) => m.res && m.res.ok);
  if (mids.length === 0) throw new Error("演示 MIDI 缺失（public/demo）");
  // 记谱层产物（每曲 notation/ 子目录）
  let notation: NotationMeta | undefined;
  try {
    const nr = await fetch(`${base}/notation/notation.json`);
    if (nr.ok) {
      notation = parseNotationMeta(await nr.text(), { kind: "url", baseUrl: base });
    }
  } catch {
    /* 无记谱输出 */
  }
  await buildAndLoad({
    name: songName,
    audioBytes: await audioR.arrayBuffer(),
    mids: await Promise.all(
      mids.map(async (m) => ({ name: m.name, bytes: await m.res!.arrayBuffer() })),
    ),
    infoText: infoR?.ok ? await infoR.text() : undefined,
    notation,
  });
}

export { isTauri };
