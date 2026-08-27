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
 * - 播放光标 = 自绘覆盖层，逐音锚定：t→QL（timeMap 反查）→当前发声
 *   音符的刻版 x（谱表条目锚点），与卷帘正在播放的音一一对应
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
  const measureRowsRef = useRef<
    { ql: number; x: number; w: number; yTop: number; yBot: number }[]
  >([]);
  const anchorsRef = useRef<{ ql: number; x: number; rowIdx: number }[]>([]);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const lastScrolledRef = useRef(-1);
  const [renderToken, setRenderToken] = useState(0); // 渲染完成信号
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 谱面加载/渲染：OSMD 实例每个容器只建一个，轨道切换时复用（load+render）。
  // 并发防护：StrictMode/HMR 下 effect 会重入，每个 await 之后必须先查
  // cancelled 再 render——否则后一次重入的 innerHTML 清空会被先一次迟到的
  // SVG 覆盖，谱面就画两遍（2026-08-27 用户报告）。
  useEffect(() => {
    let cancelled = false;
    const el = containerRef.current;
    if (!el || !trackMeta) return;
    setBusy(true);
    setError(null);
    measureRowsRef.current = [];
    anchorsRef.current = [];
    lastScrolledRef.current = -1;
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

        // autoResize=false：版心宽度固定，resize 触发的后台重渲染会与本
        // effect 的清空/重画赛跑
        const opts = {
          autoResize: false,
          backend: "svg",
          followCursor: false,
          drawTitle: false,
          drawSubtitle: false,
          drawCredits: false,
          drawLyricist: false,
          drawingParameters: "default",
        } as ConstructorParameters<typeof OpenSheetMusicDisplay>[1];
        if (!osmdRef.current) {
          osmdRef.current = new OpenSheetMusicDisplay(el, opts);
        }
        const instance = osmdRef.current;
        try {
          await instance.load(xml);
          if (cancelled) return;
          instance.render();
        } catch {
          // TAB part 兼容兜底：剥离 TAB 谱表后重载
          await instance.load(stripTabParts(xml));
          if (cancelled) return;
          instance.render();
        }
        if (cancelled) return;

        // 光标几何源：小节行（y/滚动）+ 谱表条目锚点（x）。锚点 = 每个
        // 音符/休止符条目的刻版 x 与绝对 QL——刻版横向间距不均匀（密集
        // 音挤、长音占位大），按时间线性扫谱面必然跑偏；改为逐音跳位，
        // 光标永远停在当前发声的音符上（OSMD issue #480 / alphaTab 同款
        // note-anchored 思路；2026-08-27 用户要求与卷帘音符锁死）。
        type BB = {
          absolutePosition: { x: number; y: number };
          size: { width: number; height: number };
        };
        const inst = instance as unknown as {
          graphic?: {
            MeasureList?: {
              PositionAndShape: BB;
              staffEntries?: {
                getAbsoluteTimestamp?: () => { RealValue?: number };
                PositionAndShape: BB;
              }[];
            }[][];
          };
          Sheet?: {
            SourceMeasures?: {
              AbsoluteTimestamp?: { RealValue?: number; n?: number; d?: number };
            }[];
          };
          Zoom?: number;
        };
        const rows: {
          ql: number;
          x: number;
          w: number;
          yTop: number;
          yBot: number;
        }[] = [];
        const anchors: { ql: number; x: number; rowIdx: number }[] = [];
        const ml = inst.graphic?.MeasureList ?? [];
        for (let i = 0; i < ml.length; i++) {
          let x = Infinity;
          let w = 0;
          let yTop = Infinity;
          let yBot = -Infinity;
          const entries: { ql: number; x: number }[] = [];
          for (const gm of ml[i] ?? []) {
            const bb = gm.PositionAndShape;
            const left = bb.absolutePosition.x - bb.size.width / 2;
            x = Math.min(x, left);
            w = Math.max(w, bb.size.width);
            yTop = Math.min(yTop, bb.absolutePosition.y - bb.size.height / 2);
            yBot = Math.max(yBot, bb.absolutePosition.y + bb.size.height / 2);
            for (const se of gm.staffEntries ?? []) {
              const ts = se.getAbsoluteTimestamp?.();
              if (!ts) continue;
              entries.push({
                ql: Number(ts.RealValue ?? 0),
                x: se.PositionAndShape.absolutePosition.x,
              });
            }
          }
          if (!isFinite(x)) continue;
          const ts = inst.Sheet?.SourceMeasures?.[i]?.AbsoluteTimestamp;
          const qlScore = ts
            ? Number(ts.RealValue ?? (ts.n ?? 0) / (ts.d ?? 1))
            : i * 4;
          // 小节时间戳是渲染谱相对值；timeMap 的 QL 是全曲绝对值
          rows.push({ ql: qlScore + trackMeta.minBar * 4, x, w, yTop, yBot });
          for (const e of entries) {
            anchors.push({
              ql: e.ql + trackMeta.minBar * 4,
              x: e.x,
              rowIdx: rows.length - 1,
            });
          }
        }
        measureRowsRef.current = rows;
        // 同刻去重（钢琴大谱表上下行/多声部同拍）：排序后取最靠左者
        anchors.sort((a, b) => a.ql - b.ql || a.x - b.x);
        const dedup: typeof anchors = [];
        for (const a of anchors) {
          const last = dedup[dedup.length - 1];
          if (last && Math.abs(a.ql - last.ql) < 1e-6) continue;
          dedup.push(a);
        }
        anchorsRef.current = dedup;
        const ov = document.createElement("div");
        ov.style.cssText =
          "position:absolute;z-index:5;pointer-events:none;display:none;" +
          "width:3px;border-radius:2px;background:rgba(225,29,72,0.9);" +
          "box-shadow:0 0 0 1px rgba(225,29,72,0.25);" +
          "transition:left 60ms linear;";
        el.appendChild(ov);
        overlayRef.current = ov;
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
      overlayRef.current?.remove();
      overlayRef.current = null;
    };
  }, [notation, trackMeta, active, project?.bpm]);

  // 播放光标联动（rAF；暂停/拖拽进度条也跟随）：t → QL（timeMap 反查）
  // → 二分"最后一个 onset ≤ QL"的锚点，光标停在当前发声的音符上
  useEffect(() => {
    if (renderToken === 0 || !trackMeta) return;
    let raf = 0;
    const tick = () => {
      const ov = overlayRef.current;
      const rows = measureRowsRef.current;
      const anchors = anchorsRef.current;
      if (ov && rows.length > 0) {
        const eng = activeEngine();
        const t =
          eng.duration > 0
            ? eng.currentTime
            : usePlayerStore.getState().currentTime;
        // t → QL（timeMap 反查；缺省回退名义 bpm）
        const tm = notation.timeMap;
        const bpm = project?.bpm ?? 120;
        const offsetSec = trackMeta.minBar * 4 * (60 / bpm);
        let ql: number;
        if (tm && tm.length >= 2) {
          let lo = 0;
          while (lo < tm.length - 2 && tm[lo + 1][0] < t) lo++;
          const [t0, q0] = tm[lo];
          const [t1, q1] = tm[lo + 1];
          ql = t1 > t0 ? q0 + ((t - t0) / (t1 - t0)) * (q1 - q0) : q0;
        } else {
          ql = ((t - offsetSec) * bpm) / 60 + trackMeta.minBar * 4;
        }
        const inst = osmdRef.current as unknown as { Zoom?: number };
        const zoom = inst?.Zoom ?? 1;
        let x: number;
        let rowIdx: number;
        if (anchors.length > 0) {
          // 当前发声音符 = 最后一个 onset ≤ QL 的谱表条目
          let lo = 0;
          let hi = anchors.length - 1;
          while (lo < hi) {
            const mid = (lo + hi + 1) >> 1;
            if (anchors[mid].ql <= ql) lo = mid;
            else hi = mid - 1;
          }
          x = anchors[lo].x;
          rowIdx = anchors[lo].rowIdx;
        } else {
          // 兜底（无锚点）：小节内 QL 线性插值
          let idx = 0;
          let hi = rows.length - 1;
          while (idx < hi) {
            const mid = (idx + hi + 1) >> 1;
            if (rows[mid].ql <= ql) idx = mid;
            else hi = mid - 1;
          }
          const r = rows[idx];
          const frac = Math.max(0, Math.min(1, (ql - r.ql) / 4));
          x = r.x + frac * r.w;
          rowIdx = idx;
        }
        const r = rows[Math.min(rowIdx, rows.length - 1)];
        ov.style.display = "block";
        ov.style.left = `${10 * x * zoom}px`;
        // 上下各延长 1.2 个谱线间距，保证贯穿多谱表行（标准谱+TAB）
        ov.style.top = `${10 * (r.yTop - 1.2) * zoom}px`;
        ov.style.height = `${10 * (r.yBot - r.yTop + 2.4) * zoom}px`;
        if (lastScrolledRef.current !== rowIdx) {
          lastScrolledRef.current = rowIdx;
          if (usePlayerStore.getState().isPlaying) {
            ov.scrollIntoView({ block: "center", behavior: "smooth" });
          }
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [renderToken, notation, trackMeta, project?.bpm]);

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
