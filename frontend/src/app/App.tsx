import { useEffect } from "react";
import { Check, Loader2 } from "lucide-react";
import { usePlayerStore, visibleTracks } from "@/entities/project/store";
import { audioEngine } from "@/features/playback/audioEngine";
import { midiEngine } from "@/features/playback/midiEngine";
import {
  activeEngine,
  handleEngineEnded,
  seekBy,
  togglePlay,
} from "@/features/playback/control";
import { TransportBar } from "@/widgets/transport-bar/TransportBar";
import { InstrumentPanel } from "@/widgets/instrument-panel/InstrumentPanel";
import { PlayerPage } from "@/pages/player/PlayerPage";
import { ScorePage } from "@/pages/score/ScorePage";
import { formatTime } from "@/shared/utils/cn";

export default function App() {
  const viewMode = usePlayerStore((s) => s.viewMode);
  const project = usePlayerStore((s) => s.project);
  const theme = usePlayerStore((s) => s.theme);

  // 主题已由 store.setTheme 落到 documentElement（先 DOM 后渲染，Canvas 读变量才是新值）
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // 播完落地：全部引擎停下，暂停态 + 时间停在末尾（再按播放 = 从头播）
  useEffect(() => {
    audioEngine.onEnded = handleEngineEnded;
    midiEngine.onEnded = handleEngineEnded;
    return () => {
      audioEngine.onEnded = null;
      midiEngine.onEnded = null;
    };
  }, []);

  // 静音/独奏 → MIDI 引擎各轨增益（即时；原曲模式仅影响卷帘显示）
  useEffect(() => {
    const ids = new Set(visibleTracks(project).map((t) => t.id));
    midiEngine.setAudibility(ids);
  }, [project?.tracks]);

  // 粗粒度时间节流（只供文字时间码；卷帘/进度条走各自 rAF）
  useEffect(() => {
    const id = setInterval(() => {
      const eng = activeEngine();
      if (eng.isPlaying) {
        usePlayerStore.getState().setCurrentTime(eng.currentTime);
      }
    }, 250);
    return () => clearInterval(id);
  }, []);

  // 键盘快捷键：空格 播放/暂停，←/→ ±5s
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName ?? "";
      if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;
      if (e.code === "Space") {
        e.preventDefault();
        void togglePlay();
      } else if (e.code === "ArrowLeft") {
        seekBy(-5);
      } else if (e.code === "ArrowRight") {
        seekBy(5);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="app-bg flex h-screen w-screen flex-col overflow-hidden">
      <TransportBar />
      <div className="flex min-h-0 flex-1">
        {project && <InstrumentPanel />}
        <main className="relative min-w-0 flex-1">
          {viewMode === "roll" ? <PlayerPage /> : <ScorePage />}
        </main>
      </div>
      <GlobalOverlays />
    </div>
  );
}

/** 全局浮层（盖住卷帘/乐谱两页）：加载提示 + 一键管线进度。
 * 原先 LoadingOverlay 只挂在卷帘页，乐谱页播放时音色加载无反馈，用户
 * 以为卡死（2026-08-28 用户报告）——提升到 App 层。 */
function GlobalOverlays() {
  const loading = usePlayerStore((s) => s.loading);
  const processing = usePlayerStore((s) => s.processing);
  return (
    <>
      {loading && (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-3 bg-base/60 backdrop-blur-sm">
          <Loader2 className="h-8 w-8 animate-spin text-accent" />
          <p className="text-sm text-content-2">{loading}</p>
        </div>
      )}
      {processing && <ProcessingOverlay stage={processing.stage} pct={processing.pct} label={processing.label} elapsed={processing.elapsed} />}
    </>
  );
}

const PIPE_STEPS: { id: string; label: string }[] = [
  { id: "upload", label: "上传音频" },
  { id: "bpm", label: "BPM 检测" },
  { id: "separate", label: "乐器分离" },
  { id: "transcribe", label: "多乐器转写" },
  { id: "notation", label: "记谱 + 打包" },
];

function ProcessingOverlay(props: {
  stage: string;
  pct: number;
  label: string;
  elapsed: number;
}) {
  const idx = PIPE_STEPS.findIndex((s) => s.id === props.stage);
  const activeIdx = props.stage === "done" ? PIPE_STEPS.length : idx;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-base/70 backdrop-blur-md">
      <div className="w-[440px] max-w-[88vw] rounded-2xl border border-stroke bg-surface-1/95 px-8 py-7 shadow-2xl">
        <div className="mb-1 text-sm font-medium text-content-1">
          一键转写中…
        </div>
        <div className="mb-5 truncate text-xs text-content-3">{props.label}</div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
          <div
            className="h-full rounded-full bg-accent transition-[width] duration-500"
            style={{ width: `${Math.max(1, Math.min(100, props.pct))}%` }}
          />
        </div>
        <div className="tnum mt-2 flex justify-between text-[11px] text-content-3">
          <span>{Math.floor(props.pct)}%</span>
          <span>已用 {formatTime(props.elapsed)}</span>
        </div>
        <div className="mt-5 space-y-2.5">
          {PIPE_STEPS.map((s, i) => (
            <div key={s.id} className="flex items-center gap-2.5 text-xs">
              {i < activeIdx ? (
                <Check className="h-3.5 w-3.5 shrink-0 text-accent" />
              ) : i === activeIdx ? (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-accent" />
              ) : (
                <span className="inline-block h-3.5 w-3.5 shrink-0 rounded-full border border-stroke" />
              )}
              <span className={i <= activeIdx ? "text-content-2" : "text-content-3"}>
                {s.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
