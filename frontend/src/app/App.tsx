import { useEffect } from "react";
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
    </div>
  );
}
