import { useEffect, useMemo, useRef, useState } from "react";
import { FileMusic, Loader2 } from "lucide-react";
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay";
import { usePlayerStore } from "@/entities/project/store";
import type { NotationMeta } from "@/entities/project/types";
import { activeEngine } from "@/features/playback/control";
import { tauriReadBytes } from "@/features/library/fileAccess";
import { familyColor } from "@/shared/theme/instrumentColors";
import { cn } from "@/shared/utils/cn";

/**
 * 乐谱视图（M4）：OSMD 渲染管线 notation/ 产物。
 * - 轨道页签切换单乐器谱（乐器面板行点击也会切）
 * - 播放光标联动：OSMD cursor 逐步推进，与卷帘共享同一条 playback 时间轴；
 *   时间→步数映射在渲染后用 iterator 时间戳走一遍建立
 */
export function ScoreView() {
  const project = usePlayerStore((s) => s.project);
  const notation = project?.notation;
  if (!notation || notation.tracks.length === 0) return <EmptyScore />;

  return (
    <div className="flex h-full flex-col">
      <ScoreToolbar notation={notation} />
      <div className="min-h-0 flex-1 overflow-auto px-6 pb-8 pt-4">
        <ScoreCanvas notation={notation} />
      </div>
    </div>
  );
}

function EmptyScore() {
  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="flex max-w-md flex-col items-center gap-4 rounded-2xl border border-stroke bg-surface-1/60 px-10 py-10 text-center backdrop-blur-xl">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-accent/15">
          <FileMusic className="h-8 w-8 text-accent" />
        </div>
        <h2 className="text-lg font-medium text-content-1">本曲目没有记谱输出</h2>
        <p className="text-sm leading-relaxed text-content-2">
          乐谱来自管线记谱层（MUSE_MULTI_INSTRUMENT=1 产出的 notation/ 目录）。
          <br />
          演示曲目自带 7 轨单乐器谱（吉他含 TAB）。
        </p>
      </div>
    </div>
  );
}

function ScoreToolbar({ notation }: { notation: NotationMeta }) {
  const project = usePlayerStore((s) => s.project);
  const theme = usePlayerStore((s) => s.theme);
  const scoreTrack = usePlayerStore((s) => s.scoreTrack);
  const setScoreTrack = usePlayerStore((s) => s.setScoreTrack);

  const active = scoreTrack ?? notation.tracks[0].instrumentClass;

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-stroke px-6 py-2.5">
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
        {notation.tracks.map((t) => {
          const mt = project?.tracks.find((tr) =>
            tr.id.startsWith(`${t.instrumentClass}.mid:`),
          );
          const selected = t.instrumentClass === active;
          return (
            <button
              key={t.instrumentClass}
              type="button"
              onClick={() => setScoreTrack(t.instrumentClass)}
              className={cn(
                "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-all duration-150",
                selected
                  ? "border-accent/60 bg-accent/15 text-content-1"
                  : "border-stroke text-content-2 hover:bg-surface-2 hover:text-content-1",
              )}
            >
              <span
                className="h-2 w-2 rounded-full"
                style={{
                  backgroundColor: familyColor(theme, mt?.family ?? "fx"),
                }}
              />
              {mt?.name ?? t.instrumentClass}
            </button>
          );
        })}
      </div>
      {notation.analysis?.summary ? (
        <span
          className="tnum hidden shrink-0 rounded-full border border-stroke px-2.5 py-1 text-[11px] text-content-2 lg:inline"
          title="AI 乐曲分析（T4 v0：基于和弦轨与 Billboard 语料统计）"
        >
          {notation.analysis.summary}
        </span>
      ) : null}
      {notation.key ? (
        <span className="tnum shrink-0 text-[11px] text-content-3">
          {notation.key}
        </span>
      ) : null}
    </div>
  );
}

function ScoreCanvas({ notation }: { notation: NotationMeta }) {
  const project = usePlayerStore((s) => s.project);
  const scoreTrack = usePlayerStore((s) => s.scoreTrack);
  const active = scoreTrack ?? notation.tracks[0].instrumentClass;
  const trackMeta = useMemo(
    () => notation.tracks.find((t) => t.instrumentClass === active),
    [notation, active],
  );

  const containerRef = useRef<HTMLDivElement>(null);
  const osmdRef = useRef<OpenSheetMusicDisplay | null>(null);
  const stepTimesRef = useRef<number[]>([]);
  const lastStepRef = useRef(-1);
  const lastScrolledRef = useRef(-1);
  const [renderToken, setRenderToken] = useState(0); // 渲染完成信号
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 谱面加载/渲染（轨道或模式切换时重建 OSMD 实例）
  useEffect(() => {
    let cancelled = false;
    const el = containerRef.current;
    if (!el || !trackMeta) return;
    setBusy(true);
    setError(null);
    stepTimesRef.current = [];
    lastStepRef.current = -1;
    el.innerHTML = "";

    (async () => {
      try {
        const file = `${active}.musicxml`;
        let xml: string;
        if (notation.kind === "fs") {
          xml = new TextDecoder().decode(
            await tauriReadBytes(`${notation.baseUrl}/notation/solo/${file}`),
          );
        } else {
          const r = await fetch(`${notation.baseUrl}/notation/solo/${file}`);
          if (!r.ok) throw new Error(`谱面文件缺失（${file}）`);
          xml = await r.text();
        }

        const opts = {
          autoResize: true,
          backend: "svg",
          followCursor: false,
          drawTitle: false,
          drawSubtitle: false,
          drawCredits: false,
          drawLyricist: false,
          drawingParameters: "default",
        } as ConstructorParameters<typeof OpenSheetMusicDisplay>[1];
        let instance = new OpenSheetMusicDisplay(el, opts);
        try {
          await instance.load(xml);
          instance.render();
        } catch {
          // TAB part 兼容兜底：剥离 TAB 谱表后重载
          el.innerHTML = "";
          instance = new OpenSheetMusicDisplay(el, opts);
          await instance.load(stripTabParts(xml));
          instance.render();
        }
        if (cancelled) return;
        osmdRef.current = instance;

        // 时间→步数表：iterator 逐步走一遍（时间戳为谱面相对四分音符位置）
        const bpm = project?.bpm ?? 120;
        const offsetSec = trackMeta.minBar * 4 * (60 / bpm);
        const cursor = osmdRef.current.cursor;
        cursor.show();
        cursor.reset();
        const times: number[] = [];
        for (let i = 0; i < 30000; i++) {
          const it = (cursor as unknown as {
            iterator?: {
              endReached: boolean;
              currentTimeStamp?: { realValue: number };
            };
          }).iterator;
          if (!it) break;
          const ql = it.currentTimeStamp?.realValue ?? 0;
          times.push(offsetSec + (ql * 60) / bpm);
          if (it.endReached) break;
          try {
            cursor.next();
          } catch {
            break;
          }
        }
        stepTimesRef.current = times;
        cursor.reset();
        lastStepRef.current = -1;
        setRenderToken((n) => n + 1);
        setBusy(false);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setBusy(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [notation, trackMeta, active, project?.bpm]);

  // 播放光标联动（rAF；暂停时也跟随进度条拖拽）
  useEffect(() => {
    if (renderToken === 0) return;
    let raf = 0;
    const tick = () => {
      const osmd = osmdRef.current;
      const times = stepTimesRef.current;
      if (osmd?.cursor && times.length > 0) {
        const eng = activeEngine();
        const t =
          eng.duration > 0
            ? eng.currentTime
            : usePlayerStore.getState().currentTime;
        let target = -1;
        let lo = 0;
        let hi = times.length - 1;
        while (lo <= hi) {
          const mid = (lo + hi) >> 1;
          if (times[mid] <= t) {
            target = mid;
            lo = mid + 1;
          } else {
            hi = mid - 1;
          }
        }
        const cursor = osmd.cursor;
        if (target < lastStepRef.current) {
          cursor.reset();
          lastStepRef.current = -1;
        }
        while (lastStepRef.current < target) {
          try {
            cursor.next();
          } catch {
            break;
          }
          lastStepRef.current++;
        }
        // 播放中自动滚到光标（步数变化时节流）
        if (
          usePlayerStore.getState().isPlaying &&
          lastScrolledRef.current !== lastStepRef.current
        ) {
          lastScrolledRef.current = lastStepRef.current;
          const cursorEl = (cursor as unknown as { cursorElement?: Element })
            .cursorElement;
          cursorEl?.scrollIntoView({ block: "center", behavior: "smooth" });
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [renderToken]);

  return (
    <div className="relative mx-auto max-w-[1140px]">
      {(busy || error) && (
        <div className="absolute inset-0 z-10 flex items-center justify-center">
          <div className="flex items-center gap-2 rounded-full border border-stroke bg-surface-1/90 px-4 py-2 text-sm text-content-2 shadow-lg backdrop-blur">
            {error ? (
              <>谱面加载失败：{error}</>
            ) : (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> 渲染乐谱…
              </>
            )}
          </div>
        </div>
      )}
      {/* 谱面用纸张底色（乐谱惯例），不随主题反色 */}
      <div className="overflow-hidden rounded-xl bg-[#fffdf8] shadow-lg ring-1 ring-black/5">
        <div ref={containerRef} className="min-h-[420px] px-5 py-3" />
      </div>
    </div>
  );
}

/** 剥离 MusicXML 里的 TAB part（OSMD 渲染兼容兜底） */
function stripTabParts(xml: string): string {
  try {
    const doc = new DOMParser().parseFromString(xml, "application/xml");
    const parts = [...doc.querySelectorAll("part")];
    for (const p of parts) {
      const isTab = !!p.querySelector("clef > sign")?.textContent?.match(/tab/i);
      if (isTab) {
        const id = p.getAttribute("id");
        p.remove();
        doc
          .querySelectorAll(`part-list > score-part[id="${id}"]`)
          .forEach((sp) => sp.remove());
      }
    }
    return new XMLSerializer().serializeToString(doc);
  } catch {
    return xml;
  }
}
