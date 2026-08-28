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
import {
  activeEngine,
  resetMidiCache,
  resumePlaybackAt,
} from "@/features/playback/control";

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
    try {
      duration = await audioEngine.load(raw.audioBytes);
      hasAudio = true;
    } catch (e) {
      // 原曲解码失败不致命：降级为 MIDI 播放（hasAudio=false），转写/谱面
      // 结果保住——调用方可用 project.hasAudio 判断是否走兜底（2026-08-28
      // 用户实测：管线 5 分钟跑完，最后一步解码 flac 抛"Unable to decode
      // audio data"，整个结果被丢）
      console.warn("[load] 原曲音频解码失败，降级为 MIDI 播放", e);
      audioEngine.unload();
    }
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
 * 旧单曲格式回退根目录加载。songId 省略 = 清单第一首；source 选择记谱后/
 * 处理前 MIDI 清单；resume 供曲内 A/B 切换恢复播放位置与播放态） */
export async function loadDemoProject(
  songId?: string,
  source: "score" | "raw" = "score",
  resume?: { pos: number; wasPlaying: boolean },
): Promise<void> {
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
                  mids?: string[]; raw_mids?: string[] }[];
        mids?: string[];
        audio?: string;
        name?: string;
      };
      if (Array.isArray(idx.songs) && idx.songs.length > 0) {
        const pick =
          idx.songs.find((s) => s.id === songId) ?? idx.songs[0];
        base = `/demo/${pick.dir ?? pick.id}`;
        const scoreMids = Array.isArray(pick.mids) ? pick.mids : [];
        const rawMids = Array.isArray(pick.raw_mids) ? pick.raw_mids : [];
        midNames =
          source === "raw" && rawMids.length > 0 ? rawMids : scoreMids;
        if (midNames.length === 0) midNames = ["guitar.mid"];
        if (typeof pick.audio === "string" && pick.audio) audioName = pick.audio;
        if (typeof pick.name === "string" && pick.name) songName = pick.name;
        usePlayerStore.getState().setDemoSongs(
          idx.songs.map((s) => ({
            id: s.id,
            name: s.name ?? s.id,
            dir: s.dir ?? s.id,
            mids: Array.isArray(s.mids) ? s.mids : [],
            rawMids: Array.isArray(s.raw_mids) ? s.raw_mids : [],
          })),
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
  usePlayerStore.getState().setMidiSource(source);
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
  if (resume) {
    await resumePlaybackAt(resume.pos, resume.wasPlaying);
  }
}

/** 曲内「记谱后 ↔ 处理前」切换：保留播放位置与播放态 */
export async function switchDemoMidiSource(
  source: "score" | "raw",
): Promise<void> {
  const s = usePlayerStore.getState();
  if (s.midiSource === source || !s.activeDemoId) return;
  const entry = s.demoSongs.find((d) => d.id === s.activeDemoId);
  if (!entry) return;
  if (source === "raw" && entry.rawMids.length === 0) return;
  const wasPlaying = s.isPlaying;
  const pos = activeEngine().currentTime;
  await loadDemoProject(entry.id, source, { pos, wasPlaying });
}

/** 文件选择统一分流：混选 .mid → 直接装载已有产物；纯音频 → 本地一键管线 */
export function handlePickedFiles(files: File[]): void {
  const hasMid = files.some((f) => f.name.toLowerCase().endsWith(".mid"));
  if (hasMid) {
    void loadWebFiles(files);
    return;
  }
  const audio = files.find((f) => /\.(flac|wav|mp3|ogg|m4a|aac)$/i.test(f.name));
  if (audio) void startPipelineJob(audio);
}

/** 本地一键管线桥（score_extraction/pipeline_server.py，端口 8420） */
const PIPE_BASE = "http://127.0.0.1:8420";

/** 选择音频 → 上传给本地管线 → 轮询进度 → 自动装载产物 */
export async function startPipelineJob(audio: File): Promise<void> {
  const setP = usePlayerStore.getState().setProcessing;
  try {
    setP({ stage: "upload", pct: 1, label: "上传音频到本地管线…", elapsed: 0 });
    const r = await fetch(`${PIPE_BASE}/transcribe`, {
      method: "POST",
      headers: { "x-filename": encodeURIComponent(audio.name) },
      body: await audio.arrayBuffer(),
    });
    if (!r.ok) {
      const detail = await r.text().catch(() => "");
      throw new Error(detail || `管线服务返回 ${r.status}`);
    }
    const { job } = (await r.json()) as { job: string };
    for (;;) {
      await new Promise((res) => setTimeout(res, 600));
      const pr = await fetch(`${PIPE_BASE}/progress/${job}`);
      if (!pr.ok) throw new Error("进度查询失败");
      const st = (await pr.json()) as {
        stage: string; pct: number; label: string; elapsed: number;
      };
      setP(st);
      if (st.stage === "done") break;
      if (st.stage === "error") throw new Error(st.label || "管线处理失败");
    }
    // 产物装载：index.json 的 mids 清单 + notation + info（音频用本地 File）
    const idxR = await fetch(`${PIPE_BASE}/files/${job}/index.json`);
    if (!idxR.ok) throw new Error("管线产物缺失（index.json）");
    const idx = (await idxR.json()) as {
      mids?: string[]; name?: string; audio?: string;
    };
    const midNames = Array.isArray(idx.mids) ? idx.mids : [];
    if (midNames.length === 0) throw new Error("管线未产出可播放 MIDI");
    const [infoR, ...midRs] = await Promise.all([
      fetch(`${PIPE_BASE}/files/${job}/info.json`).catch(() => null),
      ...midNames.map((m) =>
        fetch(`${PIPE_BASE}/files/${job}/${m}`).catch(() => null)),
    ]);
    const mids = midNames
      .map((name, i) => ({ name, res: midRs[i] }))
      .filter((m) => m.res && m.res.ok);
    if (mids.length === 0) throw new Error("管线 MIDI 下载失败");
    let notation: NotationMeta | undefined;
    try {
      const nr = await fetch(`${PIPE_BASE}/files/${job}/notation/notation.json`);
      if (nr.ok) {
        notation = parseNotationMeta(await nr.text(), {
          kind: "url",
          baseUrl: `${PIPE_BASE}/files/${job}`,
        });
      }
    } catch {
      /* 无记谱输出 */
    }
    setP({ stage: "done", pct: 100, label: "装载卷帘与乐谱…", elapsed: 0 });
    const name = idx.name ?? audio.name.replace(/\.[^.]+$/, "");
    const loadWith = async (audioBytes?: ArrayBuffer) =>
      buildAndLoad({
        name,
        audioBytes,
        mids: await Promise.all(
          mids.map(async (m) => ({ name: m.name, bytes: await m.res!.arrayBuffer() })),
        ),
        infoText: infoR?.ok ? await infoR.text() : undefined,
        notation,
      });
    let project = await loadWith(await audio.arrayBuffer());
    if (!project.hasAudio) {
      // 本地 File 解码失败 → 从桥取服务端留存的同一份音频再试（本地二次
      // 读取/偶发解码异常时这路能救回来；文件名见 index.json 的 audio 字段）
      try {
        const af = await fetch(
          `${PIPE_BASE}/files/${job}/${encodeURIComponent(idx.audio ?? "")}`);
        if (af.ok) {
          setP({ stage: "done", pct: 100, label: "重试装载原曲音频…", elapsed: 0 });
          project = await loadWith(await af.arrayBuffer());
        }
      } catch {
        /* 服务端副本也取不到 → 走无音频降级 */
      }
    }
    if (!project.hasAudio) {
      window.alert(
        "原曲音频无法在浏览器解码，已用 MIDI 播放（转写与谱面不受影响）");
    }
    usePlayerStore.getState().setDemoSongs([], null); // 非演示曲
    usePlayerStore.getState().setMidiSource("score");
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    const hint = /fetch|network|Failed/i.test(msg)
      ? "本地管线服务未启动：在 score_extraction 目录运行 env/python.exe pipeline_server.py 后重试"
      : msg;
    window.alert(`一键转写失败：${hint}`);
  } finally {
    usePlayerStore.getState().setProcessing(null);
  }
}

export { isTauri };
