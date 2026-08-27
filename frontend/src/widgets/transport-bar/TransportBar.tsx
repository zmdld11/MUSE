import { useEffect, useRef } from "react";
import {
  ArrowLeftRight,
  FastForward,
  FolderOpen,
  Moon,
  Pause,
  Play,
  Rewind,
  SkipBack,
  SkipForward,
  Sun,
} from "lucide-react";
import { usePlayerStore } from "@/entities/project/store";
import {
  activeEngine,
  setPlaybackSource,
  SPEED_STEPS,
  seekBy,
  seekTo,
  setPlaybackRate,
  togglePlay,
} from "@/features/playback/control";
import { openDirectoryProject, loadDemoProject } from "@/features/library/loadProject";
import { IconButton, Segmented } from "@/shared/ui/controls";
import { formatTime } from "@/shared/utils/cn";

export function TransportBar() {
  const isPlaying = usePlayerStore((s) => s.isPlaying);
  const rate = usePlayerStore((s) => s.rate);
  const duration = usePlayerStore((s) => s.project?.duration ?? 0);
  const currentTime = usePlayerStore((s) => s.currentTime);
  const hasAudio = usePlayerStore((s) => s.project?.hasAudio ?? false);
  const hasTracks = usePlayerStore((s) => (s.project?.tracks.length ?? 0) > 0);
  const playbackSource = usePlayerStore((s) => s.playbackSource);
  const theme = usePlayerStore((s) => s.theme);
  const setTheme = usePlayerStore((s) => s.setTheme);
  const viewMode = usePlayerStore((s) => s.viewMode);
  const setViewMode = usePlayerStore((s) => s.setViewMode);
  const followMode = usePlayerStore((s) => s.followMode);
  const setFollowMode = usePlayerStore((s) => s.setFollowMode);
  const demoSongs = usePlayerStore((s) => s.demoSongs);
  const activeDemoId = usePlayerStore((s) => s.activeDemoId);
  const fillRef = useRef<HTMLDivElement>(null);
  const knobRef = useRef<HTMLDivElement>(null);

  // 进度条 rAF 直更（不经过 React 状态，避免每帧重渲染）
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const eng = activeEngine();
      if (eng.duration > 0) {
        const pct = `${(eng.currentTime / eng.duration) * 100}%`;
        if (fillRef.current) fillRef.current.style.width = pct;
        if (knobRef.current) knobRef.current.style.left = pct;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const canPlay = hasTracks || hasAudio;

  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b border-stroke px-3">
      {/* 品牌区 */}
      <div className="flex items-center gap-2 pl-1 pr-2">
        <span className="text-base font-semibold tracking-[0.18em] text-accent">
          MUSE
        </span>
      </div>
      <IconButton
        title="打开目录（管线输出目录或音频文件夹）"
        onClick={() => void openDirectoryProject()}
      >
        <FolderOpen className="h-[18px] w-[18px]" />
      </IconButton>

      {/* 中央控制区 */}
      <div className="flex flex-1 items-center justify-center gap-1">
        <IconButton title="上一首（播放列表将于 M3 开放）" disabled>
          <SkipBack className="h-[18px] w-[18px]" />
        </IconButton>
        <IconButton title="回退 5 秒（←）" onClick={() => seekBy(-5)}>
          <Rewind className="h-[18px] w-[18px]" />
        </IconButton>
        <IconButton
          title={isPlaying ? "暂停（空格）" : "播放（空格）"}
          variant="accent"
          size="lg"
          disabled={!canPlay}
          onClick={() => void togglePlay()}
        >
          {isPlaying ? (
            <Pause className="h-5 w-5" />
          ) : (
            <Play className="h-5 w-5 translate-x-[1px]" />
          )}
        </IconButton>
        <IconButton title="快进 5 秒（→）" onClick={() => seekBy(5)}>
          <FastForward className="h-[18px] w-[18px]" />
        </IconButton>
        <IconButton title="下一首（播放列表将于 M3 开放）" disabled>
          <SkipForward className="h-[18px] w-[18px]" />
        </IconButton>
      </div>

      {/* 进度条 */}
      <div
        className="group relative h-6 w-44 shrink-0 cursor-pointer"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          handleScrub(e);
        }}
        onPointerMove={(e) => {
          if (e.buttons === 1) handleScrub(e);
        }}
      >
        <div className="absolute top-1/2 h-1 w-full -translate-y-1/2 rounded-full bg-surface-2" />
        <div
          ref={fillRef}
          className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-accent"
          style={{ width: 0 }}
        />
        <div
          ref={knobRef}
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent opacity-0 shadow transition-opacity duration-150 group-hover:opacity-100"
          style={{ left: 0 }}
        />
      </div>

      {/* 时间码 */}
      <span className="tnum w-[92px] shrink-0 text-center text-xs text-content-2">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>

      {/* 演示曲库切歌（多曲清单时出现） */}
      {demoSongs.length > 1 && (
        <select
          title="演示曲目"
          value={activeDemoId ?? ""}
          onChange={(e) => {
            void loadDemoProject(e.target.value).catch(() => {
              usePlayerStore.getState().setLoading(null);
            });
          }}
          className="h-8 max-w-[180px] shrink-0 cursor-pointer rounded-md border-none bg-surface-1 px-2 text-xs text-content-1 outline-none hover:bg-surface-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {demoSongs.map((s) => (
            <option key={s.id} value={s.id} style={{ background: "var(--color-surface-3)" }}>
              {s.name}
            </option>
          ))}
        </select>
      )}

      {/* 倍速 */}
      <select
        title="播放速度"
        value={rate}
        onChange={(e) => setPlaybackRate(Number(e.target.value))}
        className="h-8 shrink-0 cursor-pointer rounded-md border-none bg-surface-1 px-2 text-xs text-content-1 outline-none hover:bg-surface-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        {SPEED_STEPS.map((s) => (
          <option key={s} value={s} style={{ background: "var(--color-surface-3)" }}>
            {s.toFixed(2).replace(/\.?0+$/, "")}x
          </option>
        ))}
      </select>

      {/* 音源切换 */}
      <Segmented
        options={[
          { value: "midi", label: "MIDI" },
          { value: "audio", label: "原曲" },
          { value: "mix", label: "叠加" },
        ]}
        value={playbackSource}
        onChange={(v) => void setPlaybackSource(v)}
      />

      {/* 视图切换 */}
      <Segmented
        options={[
          { value: "roll", label: "卷帘" },
          { value: "score", label: "乐谱" },
        ]}
        value={viewMode}
        onChange={setViewMode}
      />

      {/* 跟随模式：竖线平移 ↔ 卷帘平移（八音盒） */}
      <IconButton
        title={
          followMode === "roll"
            ? "跟随模式：卷帘平移（八音盒）— 点击切回竖线平移"
            : "跟随模式：竖线平移 — 点击切换为卷帘平移（八音盒）"
        }
        className={followMode === "roll" ? "text-accent" : undefined}
        onClick={() => setFollowMode(followMode === "roll" ? "playhead" : "roll")}
      >
        <ArrowLeftRight className="h-[18px] w-[18px]" />
      </IconButton>

      {/* 主题切换 */}
      <IconButton
        title={theme === "dark" ? "切换浅色主题" : "切换深色主题"}
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      >
        {theme === "dark" ? (
          <Sun className="h-[18px] w-[18px]" />
        ) : (
          <Moon className="h-[18px] w-[18px]" />
        )}
      </IconButton>
    </header>
  );
}

function handleScrub(e: React.PointerEvent<HTMLDivElement>) {
  const rect = e.currentTarget.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const eng = activeEngine();
  if (eng.duration > 0) {
    seekTo(ratio * eng.duration);
  }
}
