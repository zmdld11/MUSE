import { useEffect, useMemo, useRef, useState } from "react";
import { FileMusic, Loader2, Mic } from "lucide-react";
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay";
import { usePlayerStore } from "@/entities/project/store";
import type { NotationMeta } from "@/entities/project/types";
import { activeEngine } from "@/features/playback/control";
import { tauriReadBytes } from "@/features/library/fileAccess";
import { familyFromInstrumentClass } from "@/features/library/loadProject";
import { familyColor } from "@/shared/theme/instrumentColors";
import { cn } from "@/shared/utils/cn";

/** 人声类（melody/vocal_harmony/choir）暂不出五线谱：人声专用谱
 *  （歌词对照）规划在下一阶段，本阶段先隐藏（2026-08-29 #4） */
function isVocalClass(cls: string): boolean {
  return familyFromInstrumentClass(cls) === "vocal";
}

/**
 * 乐谱视图（M4）：OSMD 渲染管线 notation/ 产物。
 * - 轨道页签切换单乐器谱（乐器面板行点击也会切）
 * - 播放光标 = 自绘覆盖层，逐音锚定：t→QL（timeMap 反查）→当前发声
 *   音符的刻版 x（谱表条目锚点），与卷帘正在播放的音一一对应
 */
export function ScoreView() {
  const project = usePlayerStore((s) => s.project);
  const midiSource = usePlayerStore((s) => s.midiSource);
  const notation = project?.notation;
  if (!notation || notation.tracks.length === 0) return <EmptyScore />;
  if (midiSource === "raw") {
    // 处理前（转写原始）= 听感对照专用，原始 1/48 网格时值出谱不可读
    return (
      <div className="flex h-full items-center justify-center p-8">
        <div className="flex max-w-md flex-col items-center gap-4 rounded-2xl border border-stroke bg-surface-1/60 px-10 py-10 text-center backdrop-blur-xl">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-accent/15">
            <FileMusic className="h-8 w-8 text-accent" />
          </div>
          <h2 className="text-lg font-medium text-content-1">
            处理前模式不提供乐谱
          </h2>
          <p className="text-sm leading-relaxed text-content-2">
            当前卷帘播放的是转写原始 MIDI（处理前对照）。<br />
            切回「记谱后」即可查看量化乐谱与移动光标。
          </p>
        </div>
      </div>
    );
  }

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

  // 人声类不出页签；默认选中首个非人声轨（scoreTrack 由乐器面板点人声轨
  // 置位时，谱面区显示占位卡片）
  const scoreTracks = notation.tracks.filter(
    (t) => !isVocalClass(t.instrumentClass),
  );
  const active =
    scoreTrack ??
    scoreTracks[0]?.instrumentClass ??
    notation.tracks[0].instrumentClass;

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-stroke px-6 py-2.5">
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
        {scoreTracks.map((t) => {
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
      <ChordStatsBadge analysis={notation.analysis} />
      {notation.key ? (
        <span className="tnum shrink-0 text-[11px] text-content-3">
          {notation.key}
        </span>
      ) : null}
    </div>
  );
}

/** 和弦分析徽章（第二阶段 #7）：性质大类 chips（数量+占比），hover 弹出
 *  该类具体和弦细分；进行模板命中（王道/卡农/五度圈…）单独一排小徽章。
 *  旧数据无 chord_quality_stats 时回退 summary 文本。 */
function ChordStatsBadge({
  analysis,
}: {
  analysis: NotationMeta["analysis"];
}) {
  if (!analysis) return null;
  const cats = Object.entries(analysis.chord_quality_stats ?? {}).filter(
    ([, v]) => v.count > 0,
  );
  const labels = analysis.chord_labels_by_category ?? {};
  const total = analysis.chords ?? cats.reduce((s, [, v]) => s + v.count, 0);
  const progHits = Object.entries(analysis.progressions?.hits ?? {}).filter(
    ([, hits]) => hits > 0,
  );
  const progLabels = analysis.progressions?.labels ?? {};

  if (!cats.length && !progHits.length) {
    return analysis.summary ? (
      <span
        className="tnum hidden shrink-0 rounded-full border border-stroke px-2.5 py-1 text-[11px] text-content-2 lg:inline"
        title="AI 乐曲分析（基于和弦轨与 Billboard 语料统计）"
      >
        {analysis.summary}
      </span>
    ) : null;
  }
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-1">
      {progHits.map(([key, hits]) => (
        <span
          key={key}
          className="tnum rounded-full border border-accent/50 bg-accent/10 px-2 py-0.5 text-[11px] text-content-2"
          title="和弦进行模板命中次数（含旋转与变奏匹配）"
        >
          {progLabels[key] ?? key}×{hits}
        </span>
      ))}
      {cats.map(([cat, v]) => (
        <div key={cat} className="group relative">
          <span className="tnum flex cursor-default items-center gap-1 rounded-full border border-stroke px-2 py-0.5 text-[11px] text-content-2 group-hover:border-accent/60 group-hover:bg-surface-2">
            {cat}
            <span className="text-content-3">{v.count}</span>
            <span className="text-content-3">{Math.round(v.fraction * 100)}%</span>
          </span>
          <div className="pointer-events-none absolute top-full right-0 z-20 mt-1.5 w-max min-w-32 translate-y-1 rounded-md border border-stroke bg-surface-1 p-2 opacity-0 shadow-lg transition-all duration-150 group-hover:translate-y-0 group-hover:opacity-100">
            <div className="mb-1 text-[10px] text-content-3">
              {cat}和弦细分（共 {v.count} / {total}）
            </div>
            {Object.entries(labels[cat] ?? {}).map(([label, c]) => (
              <div
                key={label}
                className="tnum flex items-center justify-between gap-4 text-[11px] text-content-2"
              >
                <span>{label}</span>
                <span className="text-content-3">
                  ×{c} · {Math.round((c / Math.max(total, 1)) * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function ScoreCanvas({ notation }: { notation: NotationMeta }) {
  const project = usePlayerStore((s) => s.project);
  const scoreTrack = usePlayerStore((s) => s.scoreTrack);
  const nonVocal = notation.tracks.find(
    (t) => !isVocalClass(t.instrumentClass),
  );
  const active =
    scoreTrack ??
    nonVocal?.instrumentClass ??
    notation.tracks[0].instrumentClass;
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
  const qlPerMeasureRef = useRef(4); // 每小节四分数（拍号 3/4=3、6/8=3）
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
        // 每小节四分数（拍号推导，替代硬编码 4 拍）：SourceMeasures 时间戳
        // 单位 = whole note，相邻小节差的中位数 ×4 即每小节 quarter 数——
        // 3/4、6/8 曲沿用 *4 会整体越播越歪
        const srcMeasures = inst.Sheet?.SourceMeasures ?? [];
        const wholeTs: number[] = [];
        for (let i = 0; i < ml.length && i < srcMeasures.length; i++) {
          const ts = srcMeasures[i]?.AbsoluteTimestamp;
          wholeTs.push(
            ts ? Number(ts.RealValue ?? (ts.n ?? 0) / (ts.d ?? 1)) : i,
          );
        }
        const tsDiffs: number[] = [];
        for (let i = 1; i < wholeTs.length; i++) {
          const d = wholeTs[i] - wholeTs[i - 1];
          if (d > 0.01) tsDiffs.push(d);
        }
        tsDiffs.sort((a, b) => a - b);
        const qlPerMeasure = tsDiffs.length
          ? Math.round(tsDiffs[Math.floor(tsDiffs.length / 2)] * 4 * 24) / 24
          : 4;
        qlPerMeasureRef.current = qlPerMeasure;
        const barShift = trackMeta.minBar * qlPerMeasure;
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
              // OSMD 时间戳单位=全音符（MusicXML n/(4·divisions) 约定），
              // timeMap/QL 是四分音符单位 → ×4 换算（08-28 用户报"歌放到
              // 1/4 光标就到谱尾"= 4× 速率错位的根因）
              entries.push({
                ql: Number(ts.RealValue ?? 0) * 4,
                x: se.PositionAndShape.absolutePosition.x,
              });
            }
          }
          if (!isFinite(x)) continue;
          // 谱内时间统一 whole 单位；timeMap 的 QL 是全曲绝对值（quarter）
          // 小节时间戳是渲染谱相对值 + barShift 平移到全曲绝对域
          const qlWholes = wholeTs[i] ?? i;
          rows.push({ ql: qlWholes * 4 + barShift, x, w, yTop, yBot });
          for (const e of entries) {
            anchors.push({
              ql: e.ql + barShift,
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
        // 无 left transition：rAF 逐帧定位本身平滑；换行时 left 跨千像素
        // 过渡反而产生"向左扫回行首"残影（2026-08-29 光标漂移报告#3）
        ov.style.cssText =
          "position:absolute;z-index:5;pointer-events:none;display:none;" +
          "width:3px;border-radius:2px;background:rgba(225,29,72,0.9);" +
          "box-shadow:0 0 0 1px rgba(225,29,72,0.25);";
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
        const st = usePlayerStore.getState();
        // 装载/换曲窗口保护：引擎未就绪（duration=0）或项目换装中
        // （loading 非空，audioEngine 仍挂旧曲）时时间域是旧的——隐藏
        // 光标，而非拿陈旧 store.currentTime 画新谱（曾表现为"突然跳回
        // 第一行开头、播放恢复后跳回来"，2026-08-29 漂移报告#1）
        if (eng.duration <= 0 || st.loading !== null) {
          ov.style.display = "none";
          lastScrolledRef.current = -1;
          raf = requestAnimationFrame(tick);
          return;
        }
        const t = eng.currentTime;
        // t → QL（timeMap 反查；缺省回退名义 bpm）
        const tm = notation.timeMap;
        const bpm = project?.bpm ?? 120;
        const barQl = trackMeta.minBar * qlPerMeasureRef.current;
        let ql: number;
        if (tm && tm.length >= 2) {
          // 二分找最后一个采样点 ≤ t（此前每帧线性扫，timeMap ~500 点）
          let lo = 0;
          let hi = tm.length - 1;
          while (lo < hi) {
            const mid = (lo + hi + 1) >> 1;
            if (tm[mid][0] <= t) lo = mid;
            else hi = mid - 1;
          }
          const [t0, q0] = tm[lo];
          const [t1, q1] = tm[Math.min(lo + 1, tm.length - 1)];
          ql = t1 > t0 ? q0 + ((t - t0) / (t1 - t0)) * (q1 - q0) : q0;
        } else {
          ql = ((t - barQl * (60 / bpm)) * bpm) / 60 + barQl;
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
          const frac = Math.max(
            0,
            Math.min(1, (ql - r.ql) / qlPerMeasureRef.current),
          );
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

  // 人声轨：不出谱（占位卡片；containerRef 不渲染 → 渲染 effect 自然跳过）
  if (isVocalClass(active)) {
    return (
      <div className="relative mx-auto max-w-[1140px]">
        <VocalScorePlaceholder />
      </div>
    );
  }

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

/** 人声谱占位：人声将有带歌词对照的专用谱面（下一阶段），旋律先看卷帘 */
function VocalScorePlaceholder() {
  return (
    <div className="flex min-h-[420px] items-center justify-center p-8">
      <div className="flex max-w-md flex-col items-center gap-4 rounded-2xl border border-stroke bg-surface-1/60 px-10 py-10 text-center backdrop-blur-xl">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-accent/15">
          <Mic className="h-8 w-8 text-accent" />
        </div>
        <h2 className="text-lg font-medium text-content-1">人声专用谱开发中</h2>
        <p className="text-sm leading-relaxed text-content-2">
          人声将有带歌词对照的专用谱面（下一阶段推出）。
          <br />
          当前可在下方卷帘查看人声旋律的音高与节奏。
        </p>
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
