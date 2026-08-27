/**
 * MIDI 合成播放引擎（smplr + FluidR3_GM soundfont，采样自 gleitz/midi-js-soundfonts）。
 *
 * 前瞻调度器：每 40ms 把未来 ~120ms（上下文时间）内的音符排进各轨乐器，
 * 每条轨一个 GainNode —— 左栏静音/独奏 = 增益即时开关 + 跳过调度。
 * 倍速通过调度时间映射生效（ctxTime = startCtx + (pos - startPos) / rate），
 * 音符时长按 1/rate 缩放。
 */
import { Soundfont } from "smplr";
import type { Note, Track } from "@/entities/project/types";
import { gleitzInstrumentName } from "@/shared/theme/gmInstruments";

const TICK_MS = 40;
const LOOKAHEAD_CTX = 0.12; // 调度前瞻（AudioContext 秒）
const START_PAD = 0.05; // 起播缓冲

interface VoiceTrack {
  id: string;
  notes: Note[];
  inst: Soundfont;
  gain: GainNode;
  idx: number; // 调度指针：下一个待调度音符
}

export class MidiEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private tracks: VoiceTrack[] = [];
  private durationVal = 0;
  private startCtx = 0;
  private startPos = 0;
  private rateVal = 1;
  private playing = false;
  private timer: number | null = null;
  private stops: Array<(time?: number) => void> = [];
  onEnded: (() => void) | null = null;

  private ensureCtx(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.master = this.ctx.createGain();
      this.master.connect(this.ctx.destination);
    }
    return this.ctx;
  }

  /** 按轨加载音色（每轨一个 GM 音色实例，路由到自己的增益节点） */
  async load(
    tracks: Track[],
    duration: number,
    onProgress?: (msg: string) => void,
  ): Promise<void> {
    const ctx = this.ensureCtx();
    this.stopAll();
    this.tracks = [];
    this.durationVal = duration;
    for (const t of tracks) {
      onProgress?.(`加载音色：${t.name}…`);
      const gain = ctx.createGain();
      gain.connect(this.master!);
      const inst = new Soundfont(ctx, {
        kit: "FluidR3_GM",
        instrument: gleitzInstrumentName(t.program, t.isDrum),
        destination: gain,
        volume: 100,
      });
      await inst.ready;
      const notes = [...t.notes].sort((a, b) => a.onset - b.onset);
      this.tracks.push({ id: t.id, notes, inst, gain, idx: 0 });
    }
  }

  /** 换曲卸载：停声并清空旧曲音色/音符（loaded=false，下次播放走
   * ensureMidiLoaded 重新加载——否则旧曲音色驻留，新曲轨道 id 对不上
   * 被可见性开关全静音 = "切歌后没加载乐器音色"的哑巴状态） */
  unload(): void {
    this.stopAll();
    this.tracks = [];
    this.durationVal = 0;
    this.startPos = 0;
  }

  /** 主音量（叠加模式下降增益防削波） */
  setMasterGain(v: number): void {
    if (this.master) this.master.gain.value = v;
  }

  get loaded(): boolean {
    return this.tracks.length > 0;
  }

  /** 左栏静音/独奏 → 各轨增益（即时，已在响的长音也停） */
  setAudibility(ids: Set<string>): void {
    for (const t of this.tracks) {
      t.gain.gain.value = ids.has(t.id) ? 1 : 0;
    }
  }

  get duration(): number {
    return this.durationVal;
  }

  get currentTime(): number {
    if (!this.playing || !this.ctx) return this.startPos;
    return Math.min(
      this.startPos + (this.ctx.currentTime - this.startCtx) * this.rateVal,
      this.durationVal,
    );
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  play(): void {
    if (this.tracks.length === 0 || this.playing) return;
    const ctx = this.ensureCtx();
    if (ctx.state === "suspended") void ctx.resume();
    if (this.startPos >= this.durationVal - 0.05) this.startPos = 0;
    this.startCtx = ctx.currentTime + START_PAD;
    this.playing = true;
    this.resetPointers(this.startPos);
    this.tick();
    this.timer = window.setInterval(() => this.tick(), TICK_MS);
  }

  pause(): void {
    if (!this.playing) return;
    this.startPos = this.currentTime;
    this.stopAll();
  }

  seek(t: number): void {
    const c = Math.max(0, Math.min(t, this.durationVal));
    if (this.playing) {
      this.stopVoices();
      this.startPos = c;
      this.startCtx = this.ctx!.currentTime + 0.05;
      this.resetPointers(c);
    } else {
      this.startPos = c;
    }
  }

  setRate(r: number): void {
    this.rateVal = r;
    if (this.playing) {
      const p = this.currentTime;
      this.stopVoices();
      this.startPos = p;
      this.startCtx = this.ctx!.currentTime + 0.05;
      this.resetPointers(p);
    }
  }

  private resetPointers(pos: number): void {
    for (const t of this.tracks) {
      let i = 0;
      while (i < t.notes.length && t.notes[i].onset < pos - 1e-6) i++;
      t.idx = i;
    }
  }

  private tick(): void {
    if (!this.playing || !this.ctx) return;
    const now = this.ctx.currentTime;
    const songPos = this.currentTime;
    const horizon = songPos + LOOKAHEAD_CTX * this.rateVal;

    for (const t of this.tracks) {
      if (t.gain.gain.value === 0) continue; // 静音轨：不调度也不推进指针
      while (t.idx < t.notes.length) {
        const n = t.notes[t.idx];
        if (n.onset > horizon) break;
        const time = this.startCtx + (n.onset - this.startPos) / this.rateVal;
        if (time > now - 0.02) {
          const dur = Math.max(0.05, (n.offset - n.onset) / this.rateVal);
          const stop = t.inst.start({
            note: n.pitch,
            velocity: Math.max(1, Math.round(n.velocity * 127)),
            time,
            duration: dur,
          });
          this.stops.push(stop);
        }
        t.idx++;
      }
    }

    if (this.stops.length > 512) this.stops = this.stops.slice(-256);

    if (this.playing && songPos >= this.durationVal - 0.02) {
      this.startPos = this.durationVal;
      this.stopAll();
      this.onEnded?.();
    }
  }

  private stopVoices(): void {
    for (const s of this.stops) {
      try {
        s();
      } catch {
        /* 已自然结束 */
      }
    }
    this.stops = [];
  }

  private stopAll(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.playing = false;
    this.stopVoices();
  }
}

export const midiEngine = new MidiEngine();
