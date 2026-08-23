import { useEffect, useMemo, useRef, useState } from "react";
import { usePlayerStore, visibleTracks } from "@/entities/project/store";
import { activeEngine, seekTo } from "@/features/playback/control";
import { familyColor } from "@/shared/theme/instrumentColors";
import { formatTime, isBlackKey } from "@/shared/utils/cn";

const KEY_W = 52; // 左侧键盘条宽
const RULER_H = 26; // 顶部时间轴高
const NOTE_H = 14; // 每半音行高
const MAX_CANVAS_W = 16000; // Chromium 单边画布上限保护
const PIN_POS = 0; // 八音盒钉位：0 = 键盘与卷帘交界处（全部视野展示未来内容）

/** 每曲缩放缓存（模块级）：卷帘↔乐谱切换重挂载后不丢用户的手动缩放 */
const zoomCache = new Map<string, number>();

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/**
 * 钢琴卷帘：Canvas 自绘 + sticky 定位条（键盘/时间轴）。
 * 静态音符层只在数据/缩放/主题变化时重绘；播放头是 DOM 元素，rAF 直改 transform。
 *
 * 缩放初始化时序（2026-08-22 压缩 bug 修复）：viewW 必须等 ResizeObserver
 * 首次回报后才可用 —— 重挂载（视图切换回来）时若用初始值算缩放，整幅卷帘
 * 会按错误的视口宽度压缩。per-project init guard + zoomCache 双保险。
 */
export function PianoRoll() {
  const project = usePlayerStore((s) => s.project);
  const theme = usePlayerStore((s) => s.theme);
  const followMode = usePlayerStore((s) => s.followMode);
  const tracks = visibleTracks(project);
  const containerRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLCanvasElement>(null);
  const keysRef = useRef<HTMLCanvasElement>(null);
  const rulerRef = useRef<HTMLCanvasElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);
  const [viewW, setViewW] = useState<number | null>(null); // null = 尚未测量
  const [pps, setPps] = useState(40); // 像素/秒

  const duration = project?.duration ?? 0;
  const notes = useMemo(() => tracks.flatMap((t) => t.notes), [tracks]);

  const [pMin, pMax] = useMemo(() => {
    if (notes.length === 0) return [45, 72];
    let lo = 127;
    let hi = 0;
    for (const n of notes) {
      if (n.pitch < lo) lo = n.pitch;
      if (n.pitch > hi) hi = n.pitch;
    }
    return [Math.max(0, lo - 2), Math.min(127, hi + 2)];
  }, [notes]);

  const rows = pMax - pMin + 1;
  const contentH = Math.max(rows * NOTE_H, 200);
  const contentW = Math.max(240, Math.min(duration * pps, MAX_CANVAS_W));
  const effPps = contentW / Math.max(duration, 0.001);

  // 换曲时重置缩放：整曲约占 1.6 屏（等视口测量完成；仅每曲一次，含重挂载恢复）
  const initedFor = useRef<string | null>(null);
  useEffect(() => {
    if (!project || viewW === null) return;
    if (initedFor.current === project.name) return;
    initedFor.current = project.name;
    const cached = zoomCache.get(project.name);
    setPps(cached ?? clamp(((viewW * 0.9) / Math.max(project.duration, 1)) * 1.6, 6, 120));
  }, [project?.name, project, viewW]);

  // 容器尺寸跟踪
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewW(el.clientWidth));
    ro.observe(el);
    setViewW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  // 静态层重绘（音符 + 键盘 + 时间轴）
  useEffect(() => {
    const main = mainRef.current;
    const keys = keysRef.current;
    const ruler = rulerRef.current;
    if (!main || !keys || !ruler) return;

    const laneC = cssVar("--roll-lane-c", "rgba(255,255,255,0.05)");
    const laneB = cssVar("--roll-lane-b", "rgba(255,255,255,0.015)");
    const gridC = cssVar("--roll-grid", "rgba(255,255,255,0.06)");
    const rulerBg = cssVar("--roll-ruler-bg", "#14151a");
    const rulerLine = cssVar("--roll-ruler-line", "rgba(255,255,255,0.18)");
    const rulerText = cssVar("--roll-ruler-text", "rgba(255,255,255,0.55)");

    const dpr = Math.min(window.devicePixelRatio || 1, contentW > 8000 ? 1 : 2);

    // ---- 主画布：行底色 + 网格 + 音符 ----
    main.width = Math.round(contentW * dpr);
    main.height = Math.round(contentH * dpr);
    main.style.width = `${contentW}px`;
    main.style.height = `${contentH}px`;
    const ctx = main.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, contentW, contentH);

    for (let p = pMin; p <= pMax; p++) {
      const y = (pMax - p) * NOTE_H;
      const pc = p % 12;
      if (pc === 0) {
        ctx.fillStyle = laneC;
        ctx.fillRect(0, y, contentW, NOTE_H);
      } else if (isBlackKey(p)) {
        ctx.fillStyle = laneB;
        ctx.fillRect(0, y, contentW, NOTE_H);
      }
    }

    const tickStep = pickTickStep(effPps);
    ctx.strokeStyle = gridC;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let t = 0; t * effPps < contentW; t += tickStep) {
      const x = Math.round(t * effPps) + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, contentH);
    }
    ctx.stroke();

    // 音符块（按族×主题着色；浅色主题去掉描边更干净——用户偏好）
    ctx.lineWidth = 1;
    const noteStroke = theme === "dark" ? "rgba(0,0,0,0.35)" : null;
    for (const track of tracks) {
      ctx.fillStyle = familyColor(theme, track.family);
      if (noteStroke) ctx.strokeStyle = noteStroke;
      for (const n of track.notes) {
        const x = n.onset * effPps;
        const w = Math.max(2.5, (n.offset - n.onset) * effPps - 1);
        if (x > contentW || x + w < 0) continue;
        const y = (pMax - n.pitch) * NOTE_H + 1.5;
        ctx.globalAlpha = 0.5 + 0.5 * clamp(n.velocity, 0.15, 1);
        ctx.beginPath();
        ctx.roundRect(x, y, w, NOTE_H - 3, 2);
        ctx.fill();
        if (noteStroke && w > 4) {
          ctx.globalAlpha = 0.5;
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;

    // ---- 键盘条 ----
    const kDpr = Math.min(window.devicePixelRatio || 1, 2);
    keys.width = Math.round(KEY_W * kDpr);
    keys.height = Math.round(contentH * kDpr);
    keys.style.width = `${KEY_W}px`;
    keys.style.height = `${contentH}px`;
    const kctx = keys.getContext("2d");
    if (kctx) {
      kctx.setTransform(kDpr, 0, 0, kDpr, 0, 0);
      kctx.clearRect(0, 0, KEY_W, contentH);
      for (let p = pMin; p <= pMax; p++) {
        const y = (pMax - p) * NOTE_H;
        kctx.fillStyle = isBlackKey(p) ? "#23262b" : "#c9ced5";
        kctx.fillRect(0, y, KEY_W, NOTE_H - 0.5);
        kctx.strokeStyle = "rgba(0,0,0,0.25)";
        kctx.strokeRect(0.5, y + 0.5, KEY_W - 1, NOTE_H - 1);
        if (p % 12 === 0) {
          kctx.fillStyle = "#3c4048";
          kctx.font = "9px 'Segoe UI', sans-serif";
          kctx.textAlign = "right";
          kctx.fillText(`C${Math.floor(p / 12) - 1}`, KEY_W - 4, y + NOTE_H - 3.5);
        }
      }
    }

    // ---- 时间轴 ----
    ruler.width = Math.round(contentW * dpr);
    ruler.height = Math.round(RULER_H * dpr);
    ruler.style.width = `${contentW}px`;
    ruler.style.height = `${RULER_H}px`;
    const rctx = ruler.getContext("2d");
    if (rctx) {
      rctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      rctx.fillStyle = rulerBg;
      rctx.fillRect(0, 0, contentW, RULER_H);
      rctx.strokeStyle = rulerLine;
      rctx.fillStyle = rulerText;
      rctx.font = "10px 'Cascadia Mono', 'Consolas', monospace";
      rctx.textAlign = "left";
      rctx.beginPath();
      for (let t = 0; t * effPps < contentW; t += tickStep) {
        const x = Math.round(t * effPps) + 0.5;
        rctx.moveTo(x, RULER_H * 0.45);
        rctx.lineTo(x, RULER_H);
        rctx.fillText(formatTime(t), x + 4, RULER_H * 0.42);
      }
      rctx.stroke();
    }
  }, [tracks, contentW, contentH, effPps, pMin, pMax, theme]);

  // 播放头 rAF + 跟随：竖线平移（走到 78% 视口才翻页）或卷帘平移（八音盒：
  // 竖线钉在视口 42% 处，内容每帧锁定滚动——暂停时仍可自由滚动）
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const eng = activeEngine();
      const ph = playheadRef.current;
      const sc = containerRef.current;
      if (ph && sc && duration > 0 && eng.duration > 0) {
        const x = eng.currentTime * effPps;
        ph.style.transform = `translateX(${x}px)`;
        if (eng.isPlaying) {
          const vw = sc.clientWidth - KEY_W;
          if (followMode === "roll") {
            sc.scrollLeft = Math.max(0, x - vw * PIN_POS);
          } else {
            const sl = sc.scrollLeft;
            if (x < sl || x > sl + vw * 0.78) {
              sc.scrollLeft = Math.max(0, x - vw * 0.15);
            }
          }
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [effPps, duration, followMode]);

  // 八音盒模式下（暂停/缩放/切换时）平滑吸附到钉位，让模式切换有即时视觉反馈
  useEffect(() => {
    if (followMode !== "roll") return;
    const eng = activeEngine();
    if (eng.isPlaying) return; // 播放中由 rAF 逐帧锁定
    const sc = containerRef.current;
    if (!sc || duration <= 0) return;
    const vw = sc.clientWidth - KEY_W;
    sc.scrollTo({
      left: Math.max(0, eng.currentTime * effPps - vw * PIN_POS),
      behavior: "smooth",
    });
  }, [followMode, effPps, duration]);

  // Ctrl+滚轮缩放（原生监听，非 passive 才能 preventDefault）；写入每曲缓存
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        setPps((p) => {
          const next = clamp(p * Math.exp(-e.deltaY * 0.0022), 4, 300);
          const name = usePlayerStore.getState().project?.name;
          if (name) zoomCache.set(name, next);
          return next;
        });
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const scrubFromEvent = (e: React.PointerEvent<HTMLElement>) => {
    if (duration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const t = (e.clientX - rect.left) / effPps;
    seekTo(clamp(t, 0, duration));
    // 八音盒模式下暂停跳转也保持钉位
    if (followMode === "roll") {
      const sc = containerRef.current;
      if (sc) {
        const vw = sc.clientWidth - KEY_W;
        sc.scrollLeft = Math.max(0, t * effPps - vw * PIN_POS);
      }
    }
  };

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full overflow-auto">
      <div
        className="relative"
        style={{ width: KEY_W + contentW, height: RULER_H + contentH }}
      >
        {/* 时间轴（行 sticky 纵向，角落块 sticky 横向） */}
        <div className="sticky top-0 z-30 flex h-[26px] w-full" style={{ background: cssVar("--roll-ruler-bg", "#14151a") }}>
          <div className="sticky left-0 z-40 h-[26px] w-[52px] shrink-0 border-r border-stroke" style={{ background: "inherit" }} />
          <canvas
            ref={rulerRef}
            className="block cursor-pointer"
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              scrubFromEvent(e);
            }}
            onPointerMove={(e) => {
              if (e.buttons === 1) scrubFromEvent(e);
            }}
          />
        </div>
        {/* 主体（键盘 sticky 横向） */}
        <div className="relative flex" style={{ height: contentH }}>
          <canvas ref={keysRef} className="sticky left-0 z-20 block shrink-0" />
          <canvas
            ref={mainRef}
            className="block cursor-crosshair"
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              scrubFromEvent(e);
            }}
          />
          {/* 播放头（x 基准 = 键盘条右侧，z 低于键盘/时间轴） */}
          <div
            ref={playheadRef}
            className="pointer-events-none absolute bottom-0 top-0 z-10 w-[2px] bg-playhead will-change-transform"
            style={{ left: KEY_W, transform: "translateX(0px)" }}
          />
        </div>
      </div>
      </div>
      {/* 八音盒读针轨道：钉在键盘与卷帘交界处（视口固定，不随内容滚动） */}
      {followMode === "roll" && (
        <div
          aria-hidden
          className="pointer-events-none absolute bottom-0 top-0 z-[15] w-0"
          style={{ left: KEY_W }}
        >
          <div className="absolute bottom-0 top-0 -left-px w-[2px] rounded bg-accent/30" />
          <div className="absolute top-[26px] h-0 w-0 -translate-x-1/2 border-x-[5px] border-t-[7px] border-x-transparent border-t-accent" />
        </div>
      )}
    </div>
  );
}

function pickTickStep(pps: number): number {
  for (const s of [1, 2, 5, 10, 15, 30, 60, 120, 300, 600]) {
    if (s * pps >= 70) return s;
  }
  return 600;
}
