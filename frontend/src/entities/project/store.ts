import { create } from "zustand";
import type { DemoSong, PipelineProgress, Project, ViewMode } from "./types";

export type PlaybackSource = "midi" | "audio" | "mix";
export type ThemeName = "dark" | "light";
export type FollowMode = "playhead" | "roll";

function initialTheme(): ThemeName {
  const saved = localStorage.getItem("muse-theme");
  return saved === "dark" ? "dark" : "light"; // 用户偏好浅色，浅色为默认
}

function initialFollowMode(): FollowMode {
  return localStorage.getItem("muse-follow") === "roll" ? "roll" : "playhead";
}

interface PlayerState {
  project: Project | null;
  loading: string | null; // 加载中提示文案
  isPlaying: boolean;
  rate: number;
  playbackSource: PlaybackSource; // 播放音源：MIDI 合成 / 原曲音频 / 叠加
  theme: ThemeName;
  followMode: FollowMode; // 播放跟随：竖线平移 / 卷帘平移（八音盒）
  /** 粗粒度当前时间（250ms 节流，仅供文字显示；卷帘播放头直读引擎） */
  currentTime: number;
  viewMode: ViewMode;
  scoreTrack: string | null; // 乐谱页展示轨（instrument_class；null = 首轨）
  midiSource: "score" | "raw"; // 卷帘 MIDI：记谱后（谱面派生）| 处理前（转写原始）
  processing: PipelineProgress | null; // 一键管线处理中（全局浮层）
  demoSongs: DemoSong[]; // 演示曲库（public/demo v2 清单；空 = 单曲/无）
  activeDemoId: string | null; // 当前加载的演示曲目 id

  setProject: (p: Project | null) => void;
  setLoading: (msg: string | null) => void;
  setPlaying: (b: boolean) => void;
  setRate: (r: number) => void;
  setPlaybackSource: (s: PlaybackSource) => void;
  setTheme: (t: ThemeName) => void;
  setFollowMode: (m: FollowMode) => void;
  setCurrentTime: (t: number) => void;
  setViewMode: (v: ViewMode) => void;
  setScoreTrack: (cls: string | null) => void;
  setMidiSource: (s: "score" | "raw") => void;
  setProcessing: (p: PipelineProgress | null) => void;
  setDemoSongs: (songs: DemoSong[], activeId: string | null) => void;
  toggleMute: (trackId: string) => void;
  toggleSolo: (trackId: string) => void;
}

export const usePlayerStore = create<PlayerState>((set) => ({
  project: null,
  loading: null,
  isPlaying: false,
  rate: 1,
  playbackSource: "midi",
  theme: initialTheme(),
  followMode: initialFollowMode(),
  currentTime: 0,
  viewMode: "roll",
  scoreTrack: null,
  midiSource: "score",
  processing: null,
  demoSongs: [],
  activeDemoId: null,

  setProject: (p) =>
    set({
      project: p,
      currentTime: 0,
      isPlaying: false,
      scoreTrack: null,
      playbackSource: p && p.tracks.length > 0 ? "midi" : "audio",
    }),
  setLoading: (msg) => set({ loading: msg }),
  setPlaying: (b) => set({ isPlaying: b }),
  setRate: (r) => set({ rate: r }),
  setPlaybackSource: (s) => set({ playbackSource: s }),
  setTheme: (t) => {
    // 先落 DOM 再 set()：保证本轮重渲染（含 Canvas 读 CSS 变量）已是新主题
    document.documentElement.dataset.theme = t;
    localStorage.setItem("muse-theme", t);
    set({ theme: t });
  },
  setFollowMode: (m) => {
    localStorage.setItem("muse-follow", m);
    set({ followMode: m });
  },
  setCurrentTime: (t) => set({ currentTime: t }),
  setViewMode: (v) => set({ viewMode: v }),
  setScoreTrack: (cls) => set({ scoreTrack: cls }),
  setMidiSource: (s) => set({ midiSource: s }),
  setProcessing: (p) => set({ processing: p }),
  setDemoSongs: (songs, activeId) => set({ demoSongs: songs, activeDemoId: activeId }),
  toggleMute: (trackId) =>
    set((s) => {
      if (!s.project) return s;
      return {
        project: {
          ...s.project,
          tracks: s.project.tracks.map((t) =>
            t.id === trackId ? { ...t, muted: !t.muted } : t,
          ),
        },
      };
    }),
  toggleSolo: (trackId) =>
    set((s) => {
      if (!s.project) return s;
      return {
        project: {
          ...s.project,
          tracks: s.project.tracks.map((t) =>
            t.id === trackId ? { ...t, solo: !t.solo } : t,
          ),
        },
      };
    }),
}));

/** 静音/独奏过滤后的可见轨道 */
export function visibleTracks(project: Project | null) {
  if (!project) return [];
  const anySolo = project.tracks.some((t) => t.solo);
  return project.tracks.filter((t) => (anySolo ? t.solo : !t.muted));
}
