/**
 * 常驻"识别新歌"入口：选音频 → 本地一键管线（转写完自动装载）。
 * 原先该入口只在空首页有——装了歌之后想换歌识别只能刷新回空首页
 * （2026-08-28 用户"换歌识别怎么办"），顶栏常驻后随时可换。
 */
import { useRef } from "react";
import { FileAudio } from "lucide-react";
import { handlePickedFiles } from "./loadProject";
import { IconButton } from "@/shared/ui/controls";

export function TranscribeButton() {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <>
      <IconButton
        title="识别新歌（选择音频一键转写）"
        onClick={() => inputRef.current?.click()}
      >
        <FileAudio className="h-[18px] w-[18px]" />
      </IconButton>
      <input
        ref={inputRef}
        type="file"
        hidden
        accept=".flac,.wav,.mp3,.ogg,.m4a,.aac,.mid,.json"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length > 0) handlePickedFiles(files);
          e.target.value = "";
        }}
      />
    </>
  );
}
