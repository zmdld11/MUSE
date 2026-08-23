/**
 * 文件访问适配器：Tauri 环境（桌面端）与 Web 环境双实现。
 * 桌面端走 Rust 命令（scan_dir / read_bytes），Web 端走 File API —— 网页版迁移时只换这层。
 */
export const isTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export interface DirScan {
  audio?: string;
  mids: string[];
  info?: string;
}

const AUDIO_EXTS = new Set(["wav", "flac", "mp3", "ogg", "m4a", "aac"]);

export function isAudioFilename(name: string): boolean {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return AUDIO_EXTS.has(ext);
}

/** 系统目录选择对话框（仅 Tauri） */
export async function pickDirectory(): Promise<string | null> {
  if (!isTauri) return null;
  const { open } = await import("@tauri-apps/plugin-dialog");
  const r = await open({
    directory: true,
    multiple: false,
    title: "打开管线输出目录或音频文件夹",
  });
  return typeof r === "string" ? r : null;
}

/** 扫描目录：找音频、.mid、info.json（仅 Tauri，命令在 src-tauri/src/lib.rs） */
export async function tauriScanDir(dir: string): Promise<DirScan> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<DirScan>("scan_dir", { dir });
}

/** 读任意文件字节（仅 Tauri；二进制走 tauri::ipc::Response，不经 JSON 序列化） */
export async function tauriReadBytes(path: string): Promise<ArrayBuffer> {
  const { invoke } = await import("@tauri-apps/api/core");
  const r = await invoke<ArrayBuffer>("read_bytes", { path });
  return r;
}
