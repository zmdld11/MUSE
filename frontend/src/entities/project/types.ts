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

/** 简谱/歌词谱用：量化事件的谱面片段（notation.json tracks[].events[].frags） */
export interface NotationFrag {
  bar: number;
  offset: number; // 小节内位置（四分音符单位）
  dur: string; // 时值（Fraction 字符串，四分音符单位）
  tie: string | null; // "start" | "continue" | "stop" | null
}

/** 量化事件（人声轨保留全量，供歌词谱/简谱渲染） */
export interface NotationEvent {
  pitch: number;
  onset_sec: number;
  offset_sec: number;
  bar: number;
  frags: NotationFrag[];
  tie?: string | null;
  lyric?: string; // 歌词字（多字一音时拼接；无 LRC 时缺省）
  ornament?: string | null; // "vibrato" | "glissando_up" | "glissando_down" | null
  voice?: number;
}

/** 记谱轨道元信息（notation.json 摘要，谱面文件按约定路径拼） */
export interface NotationTrackMeta {
  instrumentClass: string;
  minBar: number; // 单乐器谱裁剪起始小节（量化域绝对小节号）
  firstOnsetSec: number; // 首音原始时间（忠实域起始小节推算用）
  noteCount: number;
  events?: NotationEvent[]; // 人声轨保留全量（歌词谱/简谱用）
}

/** 歌词增强层（人声专项 v2）：lines 供简谱按歌词行断行 */
export interface NotationLyrics {
  lines: { t0: number; t1: number; text: string }[];
  source_file?: string;
}

export interface NotationMeta {
  kind: "url" | "fs"; // 浏览器演示（fetch 相对路径）| Tauri 目录（read_bytes）
  baseUrl: string; // kind=url → "/demo"；kind=fs → 曲目目录绝对路径
  key?: string; // 调号估计（"E major"）
  analysis?: {
    summary?: string;
    chords?: number;
    /** 进行模板命中（第二阶段 #7）：hits 键 → 次数；labels 键 → 中文名 */
    progressions?: {
      hits: Record<string, number>;
      labels: Record<string, string>;
    };
    /** 和弦性质大类：名称 → 数量/占比 */
    chord_quality_stats?: Record<string, { count: number; fraction: number }>;
    /** 大类 → 具体和弦 label → 次数（hover 弹窗细分） */
    chord_labels_by_category?: Record<string, Record<string, number>>;
  } & Record<string, unknown>; // T4 乐曲分析
  tracks: NotationTrackMeta[];
  lyrics?: NotationLyrics; // 歌词增强层（无 LRC 时缺省 = 纯旋律谱）
  timeSignature?: string; // "4/4"（简谱拍号头）
  /** 真实时间→谱面 QL 分段线性表（rubato 曲目光标同步；缺省回退名义 bpm） */
  timeMap?: [number, number][];
}

export type ViewMode = "roll" | "score";

/** 卷帘音源：谱面派生 MIDI（记谱后）| 转写原始 MIDI（处理前对照） */
export type MidiSource = "score" | "raw";

/** 一键管线进度（本地桥轮询；stage: upload/bpm/separate/transcribe/notation/done/error） */
export interface PipelineProgress {
  stage: string;
  pct: number;
  label: string;
  elapsed: number; // 秒
}

/** 演示曲库条目（public/demo v2 清单） */
export interface DemoSong {
  id: string; // 子目录名（稳定 key）
  name: string; // 显示名（音频文件名 stem）
  dir: string; // demo 根下的子目录
  mids: string[]; // 记谱后（score_mid）清单
  rawMids: string[]; // 处理前（转写原始）清单，空 = 无对照
}
