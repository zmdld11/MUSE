/**
 * 窗口级拖入装载（issue #1）：
 * - Tauri 桌面端：onDragDropEvent 拿真实路径——单个目录 → 目录装载
 *   （同「打开目录」）；文件集 → 读字节重建 File 走 handlePickedFiles 分流
 *   （音频±lrc → 一键管线；含 .mid → 直接装载卷帘）。
 *   Tauri v2 dragDrop 默认开启，HTML5 drop 在桌面端不触发，必须走这条路。
 * - Web 模式（dev/网页版）：HTML5 dragover/drop（dataTransfer.files，仅文件级；
 *   目录拖入的 webkitGetAsEntry 递归不做——桌面端路径已覆盖）。
 *
 * installDragDrop(onDragging) 挂全局监听、返回清理函数；onDragging 驱动
 * App 层的全窗遮罩反馈。
 */
import { handlePickedFiles, loadDirectoryProject } from "./loadProject";
import { isTauri, tauriReadBytes, tauriScanDir } from "./fileAccess";

/** 可接收的文件后缀（与识别入口 accept 对齐；目录不走这里） */
const PICKUP_RE = /\.(flac|wav|mp3|ogg|m4a|aac|mid|lrc|json)$/i;

export function installDragDrop(onDragging: (v: boolean) => void): () => void {
  if (isTauri) {
    let unlisten: (() => void) | null = null;
    let disposed = false;
    void (async () => {
      const { getCurrentWebview } = await import("@tauri-apps/api/webview");
      unlisten = await getCurrentWebview().onDragDropEvent((ev) => {
        const p = ev.payload;
        if (p.type === "enter" || p.type === "over") {
          onDragging(true);
        } else {
          onDragging(false);
          if (p.type === "drop") void handleDroppedPaths(p.paths);
        }
      });
      if (disposed) unlisten?.();
    })();
    return () => {
      disposed = true;
      unlisten?.();
    };
  }

  // Web：计数器法防子元素 dragleave 抖动
  let depth = 0;
  const onEnter = (e: DragEvent) => {
    e.preventDefault();
    depth++;
    onDragging(true);
  };
  const onOver = (e: DragEvent) => e.preventDefault();
  const onLeave = (e: DragEvent) => {
    e.preventDefault();
    if (--depth <= 0) {
      depth = 0;
      onDragging(false);
    }
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    depth = 0;
    onDragging(false);
    const files = Array.from(e.dataTransfer?.files ?? []);
    if (files.length > 0) handlePickedFiles(files);
  };
  window.addEventListener("dragenter", onEnter);
  window.addEventListener("dragover", onOver);
  window.addEventListener("dragleave", onLeave);
  window.addEventListener("drop", onDrop);
  return () => {
    window.removeEventListener("dragenter", onEnter);
    window.removeEventListener("dragover", onOver);
    window.removeEventListener("dragleave", onLeave);
    window.removeEventListener("drop", onDrop);
  };
}

async function handleDroppedPaths(paths: string[]): Promise<void> {
  if (paths.length === 0) return;
  // 单一路径且是目录 → 目录装载。scan_dir 对文件路径会报错（read_dir 失败），
  // 据此判别；空目录（无音频无 mid）落到文件分支后自然无动作。
  if (paths.length === 1) {
    try {
      const scan = await tauriScanDir(paths[0]);
      if (scan.audio || scan.mids.length > 0) {
        await loadDirectoryProject(paths[0], scan);
        return;
      }
    } catch {
      /* 非目录/不可读 → 按文件处理 */
    }
  }
  const files: File[] = [];
  for (const p of paths) {
    const name = p.split(/[\\/]/).pop() ?? p;
    if (!PICKUP_RE.test(name)) continue;
    try {
      files.push(new File([await tauriReadBytes(p)], name));
    } catch (e) {
      console.warn("[drop] 读取失败，跳过", p, e);
    }
  }
  if (files.length > 0) handlePickedFiles(files);
}
