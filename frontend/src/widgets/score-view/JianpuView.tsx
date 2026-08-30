import { useMemo } from "react";
import type { NotationEvent, NotationMeta, NotationTrackMeta } from "@/entities/project/types";

/**
 * 歌词简谱（人声专项 v2 · E）：notation.json 人声轨量化事件 → 首调数字谱。
 *
 * - 首调映射：大调 1=主音；小调 la-based（记谱按关系大调，主音=6）
 * - 变化音：半音阶表（#4/#5/#1…）
 * - 八度点：高音点上加点、低音点下加点（参考八度 =  tonic 所在 [C4,B4]）
 * - 时值：八分 1 线 / 十六分 2 线；附点 ·；≥2 拍整数用增时线 —（每线 1 拍）
 * - 三连音（1/3、2/3 家族）按听觉时长取线数（1/3→1 线、2/3→0 线）
 * - 休止：0（从 frags 小节内空档合成；空小节 = 整小节休止）
 * - 连音线：tie 片段头上画短弧（v1 逐片段近似）
 * - 歌词：量化事件 lyric 字段直挂（无 LRC = 无字）
 */

const PC: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const LETTERS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const DEG = ["1", "#1", "2", "#2", "3", "4", "#4", "5", "#5", "6", "#6", "7"];

function parseFraction(s: string): number {
  const [a, b] = s.split("/");
  return b ? Number(a) / Number(b) : Number(a);
}

function tonicFromKey(key?: string): { pc: number; mode: "major" | "minor" } {
  const m = /^([A-G])([#b]?)(?:\s*(major|minor))?/i.exec(key ?? "");
  if (!m) return { pc: 0, mode: "major" };
  let pc = PC[m[1].toUpperCase()] + (m[2] === "#" ? 1 : m[2] === "b" ? -1 : 0);
  const mode = (m[3] ?? "major").toLowerCase() === "minor" ? "minor" : "major";
  if (mode === "minor") pc = (((pc + 3) % 12) + 12) % 12; // 关系大调记谱
  return { pc, mode };
}

interface JpItem {
  kind: "note" | "rest";
  digit: string;
  acc?: string;
  oct: number; // >0 高音点数，<0 低音点数
  underlines: number;
  dotted: boolean;
  dashes: number; // 增时线数（每线 1 拍）
  lyric?: string;
  tie?: string | null;
  trill?: boolean; // 颤音记号（tr）
}

function durVisual(ql: number): Pick<JpItem, "underlines" | "dotted" | "dashes"> {
  const eps = 0.02;
  if (Math.abs(ql - 1 / 3) < eps || Math.abs(ql - 2 / 3) < eps) {
    return { underlines: ql < 0.5 ? 1 : 0, dotted: false, dashes: 0 };
  }
  if (ql >= 1.75) {
    return { underlines: 0, dotted: false, dashes: Math.max(1, Math.round(ql) - 1) };
  }
  const cands = [1, 0.5, 0.25, 0.125, 0.0625];
  const u = cands.find((c) => ql >= c - eps) ?? 0.0625;
  return {
    underlines: Math.round(Math.log2(1 / u)),
    dotted: ql > u + eps,
    dashes: 0,
  };
}

function buildBars(events: NotationEvent[], qlPerMeasure: number, tonicPc: number) {
  type Fr = {
    bar: number;
    offset: number;
    dur: number;
    tie: string | null;
    lyric?: string;
    ornament?: string | null;
    pitch: number;
  };
  const frs: Fr[] = [];
  for (const ev of events) {
    ev.frags.forEach((f, i) => {
      frs.push({
        bar: f.bar,
        offset: f.offset,
        dur: parseFraction(f.dur),
        tie: f.tie,
        lyric: i === 0 ? ev.lyric : undefined,
        ornament: i === 0 ? ev.ornament : undefined,
        pitch: ev.pitch,
      });
    });
  }
  frs.sort((a, b) => a.bar - b.bar || a.offset - b.offset || a.pitch - b.pitch);

  const bars = new Map<number, { items: JpItem[]; pos: number }>();
  const ensure = (bar: number) => {
    if (!bars.has(bar)) bars.set(bar, { items: [], pos: 0 });
    return bars.get(bar)!;
  };
  const pushRest = (state: { items: JpItem[]; pos: number }, dur: number) => {
    state.items.push({ kind: "rest", digit: "0", oct: 0, ...durVisual(dur) });
    state.pos += dur;
  };
  for (const f of frs) {
    const st = ensure(f.bar);
    if (f.offset - st.pos > 0.12) pushRest(st, f.offset - st.pos);
    st.items.push({ kind: "note", ...pitchJp(f.pitch, tonicPc),
                    ...durVisual(f.dur), lyric: f.lyric, tie: f.tie,
                    trill: f.ornament === "vibrato" });
    st.pos = f.offset + f.dur;
  }
  const barNos = [...bars.keys()].sort((a, b) => a - b);
  for (const b of barNos) {
    const st = bars.get(b)!;
    if (qlPerMeasure - st.pos > 0.12) pushRest(st, qlPerMeasure - st.pos);
  }
  return barNos.map((b) => ({ bar: b, items: bars.get(b)!.items }));
}

function pitchJp(p: number, tonicPc: number): { digit: string; acc?: string; oct: number } {
  const rel = (((p - tonicPc) % 12) + 12) % 12;
  const tok = DEG[rel];
  const ref = 60 + (((tonicPc - 60) % 12) + 12) % 12;
  return {
    digit: tok[tok.length - 1],
    acc: tok.startsWith("#") ? "#" : undefined,
    oct: Math.floor((p - ref) / 12),
  };
}

export function JianpuView({
  notation,
  track,
  bpm,
}: {
  notation: NotationMeta;
  track: NotationTrackMeta;
  bpm?: number;
}) {
  const { pc, mode } = useMemo(() => tonicFromKey(notation.key), [notation.key]);
  const [tsNum, tsDen] = useMemo(() => {
    const m = /^(\d+)\/(\d+)$/.exec(notation.timeSignature ?? "4/4");
    return m ? [Number(m[1]), Number(m[2])] : [4, 4];
  }, [notation.timeSignature]);
  const qlPerMeasure = (tsNum * 4) / tsDen;
  const barList = useMemo(
    () => buildBars(track.events ?? [], qlPerMeasure, pc),
    [track.events, qlPerMeasure, pc],
  );

  return (
    <div className="mx-auto max-w-[1140px]">
      <div className="overflow-hidden rounded-xl bg-[#fffdf8] shadow-lg ring-1 ring-black/5">
        <div className="px-6 py-5 text-[#1f1f1f]">
          <div className="mb-4 flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <span className="text-xl font-semibold tracking-wide">1={LETTERS[pc]}</span>
            <span className="tnum text-base">
              {notation.timeSignature ?? "4/4"}
            </span>
            {bpm ? <span className="tnum text-sm opacity-70">♩= {Math.round(bpm)}</span> : null}
            {mode === "minor" ? (
              <span className="text-xs opacity-60">小调（6 = la 主音）</span>
            ) : null}
            <span className="ml-auto text-xs opacity-50">
              简谱（首调）· 共 {barList.length} 小节
            </span>
          </div>
          <div className="flex flex-wrap items-start gap-y-5 pb-1 leading-none">
            {barList.map(({ bar, items }, bi) => (
              <div
                key={bar}
                className="flex items-start border-l-[1.5px] border-[#1f1f1f]/70 pl-1 pr-2"
              >
                {items.map((it, i) => (
                  <JpItemView key={i} it={it} />
                ))}
                {bi === barList.length - 1 && (
                  <div className="ml-1 self-stretch border-l-[3px] border-r-[1.5px] border-[#1f1f1f]" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function JpItemView({ it }: { it: JpItem }) {
  return (
    <div className="flex w-8 shrink-0 flex-col items-center">
      {/* 顶部：颤音 tr + 连音弧 + 高音点 */}
      <div className="flex h-3 items-end justify-center">
        {it.trill ? (
          <span className="mr-px text-[10px] font-semibold italic leading-none">
            tr
          </span>
        ) : null}
        {it.tie ? (
          <span className="block h-[6px] w-[22px] rounded-t-full border-t-[1.5px] border-[#1f1f1f]/80" />
        ) : null}
        {it.oct > 0 ? (
          <span className="ml-px text-[10px] leading-none">{"‧".repeat(it.oct)}</span>
        ) : null}
      </div>
      {/* 音符主体：变化音 + 数字 + 附点 + 增时线 */}
      <div className="flex items-baseline whitespace-nowrap text-[17px] font-medium">
        {it.acc ? <span className="text-[12px]">{it.acc}</span> : null}
        <span>{it.digit}</span>
        {it.dotted ? <span className="pl-[1px] text-[15px]">·</span> : null}
        {it.dashes > 0 ? (
          <span className="pl-[2px] text-[15px]">{"—".repeat(it.dashes)}</span>
        ) : null}
      </div>
      {/* 时值下划线 */}
      <div className="mt-[2px] flex w-5 flex-col gap-[2px]">
        {Array.from({ length: it.underlines }).map((_, i) => (
          <span key={i} className="block h-[1.5px] w-full rounded bg-[#1f1f1f]" />
        ))}
      </div>
      {/* 低音点 */}
      <div className="h-3 text-[10px] leading-none">
        {it.oct < 0 ? "‧".repeat(-it.oct) : ""}
      </div>
      {/* 歌词字 */}
      <div className="mt-[2px] min-h-[14px] max-w-[36px] text-[11.5px] leading-tight text-[#1f1f1f]/90">
        {it.lyric ?? ""}
      </div>
    </div>
  );
}
