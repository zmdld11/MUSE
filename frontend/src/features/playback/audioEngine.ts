/**
 * Web Audio 播放引擎。
 * 唯一的时间真源：卷帘播放头、进度条全部直读 currentTime，不经过 React 状态。
 * AudioBufferSourceNode 只能启动一次，seek/变速通过"记录偏移 + 重建 source"实现。
 */
export class AudioEngine {
  private ctx: AudioContext | null = null;
  private buffer: AudioBuffer | null = null;
  private src: AudioBufferSourceNode | null = null;
  private masterGain: GainNode | null = null;
  private startedAt = 0; // ctx.currentTime 基准
  private offsetAt = 0; // buffer 内偏移（秒）
  private rateVal = 1;
  private playing = false;

  onEnded: (() => void) | null = null;

  private ensureCtx(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.masterGain = this.ctx.createGain();
      this.masterGain.connect(this.ctx.destination);
    }
    return this.ctx;
  }

  /** 主音量（叠加模式下降增益防削波） */
  setMasterGain(v: number): void {
    if (this.masterGain) this.masterGain.gain.value = v;
  }

  async load(arrayBuffer: ArrayBuffer): Promise<number> {
    const ctx = this.ensureCtx();
    this.stopSource();
    this.playing = false;
    this.offsetAt = 0;
    this.buffer = await ctx.decodeAudioData(arrayBuffer);
    return this.buffer.duration;
  }

  /** 换曲卸载（解码失败降级用）：停声并清 buffer——否则 duration 还报
   * 旧曲的长度（decodeAudioData 抛异常时 this.buffer 保持旧值） */
  unload(): void {
    this.stopSource();
    this.playing = false;
    this.offsetAt = 0;
    this.buffer = null;
  }

  get duration(): number {
    return this.buffer?.duration ?? 0;
  }

  get currentTime(): number {
    if (!this.playing || !this.ctx || !this.buffer) return this.offsetAt;
    const t = this.offsetAt + (this.ctx.currentTime - this.startedAt) * this.rateVal;
    return Math.min(t, this.buffer.duration);
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  get rate(): number {
    return this.rateVal;
  }

  play(): void {
    if (!this.buffer || this.playing) return;
    const ctx = this.ensureCtx();
    if (ctx.state === "suspended") void ctx.resume();
    const src = ctx.createBufferSource();
    src.buffer = this.buffer;
    src.playbackRate.value = this.rateVal;
    src.connect(this.masterGain!);
    src.onended = () => {
      if (this.src !== src) return; // 手动 stop 触发的 ended，忽略
      this.playing = false;
      this.offsetAt = this.duration;
      this.src = null;
      this.onEnded?.();
    };
    src.start(0, Math.min(this.offsetAt, this.duration));
    this.startedAt = ctx.currentTime;
    this.src = src;
    this.playing = true;
  }

  pause(): void {
    if (!this.playing) return;
    this.offsetAt = this.currentTime;
    this.stopSource();
    this.playing = false;
  }

  seek(t: number): void {
    const clamped = Math.max(0, Math.min(t, this.duration));
    if (this.playing) {
      this.stopSource();
      this.offsetAt = clamped;
      this.playing = false;
      this.play();
    } else {
      this.offsetAt = clamped;
    }
  }

  setRate(r: number): void {
    this.rateVal = r;
    if (this.playing) {
      const t = this.currentTime;
      this.stopSource();
      this.offsetAt = t;
      this.playing = false;
      this.play();
    }
  }

  private stopSource(): void {
    if (this.src) {
      this.src.onended = null;
      try {
        this.src.stop();
      } catch {
        /* 已停止 */
      }
      this.src = null;
    }
  }
}

export const audioEngine = new AudioEngine();
