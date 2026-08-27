/** 统一音符事件（与管线 notes.json schema 对齐） */
export interface Note {
  onset: number; // 秒
  offset: number; // 秒
  pitch: number; // MIDI 音高
  velocity: number; // 0-1（归一化）
}

/** 乐器族（配色与分组的最小单位） */
export type Family =
  | "guitar"
  | "bass"
  | "keys"
  | "strings"
  | "winds"
  | "brass"
  | "synth"
  | "vocal"
  | "perc"
  | "fx";

/** 一条乐器轨（MIDI 解析后的展示单元；颜色按 family×theme 动态解析） */
export interface Track {
  id: string;
  name: string; // 显示名（中文优先）
  family: Family;
  program: number; // GM 音色号（MIDI 合成播放用）
  isDrum: boolean; // channel 10 鼓轨
  notes: Note[];
  muted: boolean;
  solo: boolean;
}

export interface Project {
  name: string;
  bpm?: number;
  duration: number; // 秒（音频解码后回填，无音频时用 MIDI 尾部）
  tracks: Track[];
  hasAudio: boolean;
  notation?: NotationMeta; // 记谱层产物（管线 notation/ 目录存在时）
}

/** 记谱轨道元信息（notation.json 摘要，谱面文件按约定路径拼） */
export interface NotationTrackMeta {
  instrumentClass: string;
  minBar: number; // 单乐器谱裁剪起始小节（量化域绝对小节号）
  firstOnsetSec: number; // 首音原始时间（忠实域起始小节推算用）
  noteCount: number;
}

export interface NotationMeta {
  kind: "url" | "fs"; // 浏览器演示（fetch 相对路径）| Tauri 目录（read_bytes）
  baseUrl: string; // kind=url → "/demo"；kind=fs → 曲目目录绝对路径
  key?: string; // 调号估计（"E major"）
  analysis?: { summary?: string } & Record<string, unknown>; // T4 乐曲分析
  tracks: NotationTrackMeta[];
  /** 真实时间→谱面 QL 分段线性表（rubato 曲目光标同步；缺省回退名义 bpm） */
  timeMap?: [number, number][];
}

export type ViewMode = "roll" | "score";
export type ScoreMode = "quantized" | "faithful";

/** 演示曲库条目（public/demo v2 清单） */
export interface DemoSong {
  id: string; // 子目录名（稳定 key）
  name: string; // 显示名（音频文件名 stem）
}
