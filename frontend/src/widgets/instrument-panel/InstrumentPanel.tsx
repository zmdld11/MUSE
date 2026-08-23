import { Headphones, Volume2, VolumeX } from "lucide-react";
import { usePlayerStore, visibleTracks } from "@/entities/project/store";
import type { Track } from "@/entities/project/types";
import { FAMILY_NAMES, familyColor } from "@/shared/theme/instrumentColors";
import { cn } from "@/shared/utils/cn";

export function InstrumentPanel() {
  const project = usePlayerStore((s) => s.project);
  const theme = usePlayerStore((s) => s.theme);
  if (!project) return null;
  const visible = new Set(visibleTracks(project).map((t) => t.id));

  return (
    <aside className="flex w-52 shrink-0 flex-col border-r border-stroke bg-surface-1/40 backdrop-blur-xl">
      <div className="px-4 pb-1 pt-3.5 text-[11px] font-medium tracking-wider text-content-3">
        乐器 · {project.tracks.length}
      </div>
      <div className="flex-1 overflow-y-auto px-2 pb-3">
        {project.tracks.map((t) => (
          <TrackRow key={t.id} track={t} visible={visible.has(t.id)} theme={theme} />
        ))}
      </div>
      {project.bpm ? (
        <div className="tnum border-t border-stroke px-4 py-2 text-[11px] text-content-3">
          BPM {Math.round(project.bpm)}
        </div>
      ) : null}
    </aside>
  );
}

function TrackRow({
  track,
  visible,
  theme,
}: {
  track: Track;
  visible: boolean;
  theme: "dark" | "light";
}) {
  const toggleMute = usePlayerStore((s) => s.toggleMute);
  const toggleSolo = usePlayerStore((s) => s.toggleSolo);
  const setScoreTrack = usePlayerStore((s) => s.setScoreTrack);
  const scoreTrack = usePlayerStore((s) => s.scoreTrack);
  // 轨 id 形如 "{instrument_class}.mid:0"，与记谱轨道对齐
  const cls = track.id.split(".mid:")[0];
  const selected = scoreTrack === cls;

  return (
    <div
      className={cn(
        "group flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 ring-1 ring-inset transition-all duration-150 hover:bg-surface-2",
        selected ? "ring-accent/50" : "ring-transparent",
        !visible && "opacity-35",
      )}
      onClick={() => setScoreTrack(cls)}
      title="点击在乐谱页查看该轨单乐器谱"
    >
      <span
        className="h-2.5 w-2.5 shrink-0 rounded-[3px]"
        style={{ backgroundColor: familyColor(theme, track.family) }}
      />
      <div className="min-w-0 flex-1 leading-tight">
        <div className="truncate text-[13px] text-content-1">{track.name}</div>
        <div className="text-[10px] text-content-3">
          {FAMILY_NAMES[track.family]} · {track.notes.length} 音符
        </div>
      </div>
      <button
        type="button"
        title={track.muted ? "取消静音" : "静音（显示层）"}
        aria-label={track.muted ? "取消静音" : "静音"}
        onClick={() => toggleMute(track.id)}
        className={cn(
          "flex h-6 w-6 items-center justify-center rounded text-content-3 transition-colors hover:bg-surface-2 hover:text-content-1",
          track.muted && "text-accent",
        )}
      >
        {track.muted ? (
          <VolumeX className="h-3.5 w-3.5" />
        ) : (
          <Volume2 className="h-3.5 w-3.5 opacity-0 transition-opacity group-hover:opacity-100" />
        )}
      </button>
      <button
        type="button"
        title={track.solo ? "取消独奏" : "独奏（显示层）"}
        aria-label={track.solo ? "取消独奏" : "独奏"}
        onClick={() => toggleSolo(track.id)}
        className={cn(
          "flex h-6 w-6 items-center justify-center rounded text-content-3 transition-colors hover:bg-surface-2 hover:text-content-1",
          track.solo && "text-accent",
        )}
      >
        <Headphones
          className={cn(
            "h-3.5 w-3.5 transition-opacity",
            !track.solo && "opacity-0 group-hover:opacity-100",
          )}
        />
      </button>
    </div>
  );
}
