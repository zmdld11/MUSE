import { useRef, useState } from "react";
import { FolderOpen, Music4 } from "lucide-react";
import { usePlayerStore } from "@/entities/project/store";
import { PianoRoll } from "@/widgets/piano-roll/PianoRoll";
import {
  handlePickedFiles,
  isTauri,
  loadDemoProject,
  loadWebFiles,
  openDirectoryProject,
} from "@/features/library/loadProject";
import { Button } from "@/shared/ui/controls";

export function PlayerPage() {
  const project = usePlayerStore((s) => s.project);
  const loading = usePlayerStore((s) => s.loading);
  const processing = usePlayerStore((s) => s.processing);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dirInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="relative h-full w-full">
      <PianoRoll />
      {!project && !loading && !processing && (
        <EmptyState
          onPickDir={() => {
            if (isTauri) {
              void openDirectoryProject().catch((e: Error) => window.alert(e.message));
            } else {
              dirInputRef.current?.click();
            }
          }}
          onPickFiles={() => fileInputRef.current?.click()}
        />
      )}

      {/* Web 模式文件入口：纯音频 → 本地一键管线；自带 mid/info → 直接装载 */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        hidden
        accept=".flac,.wav,.mp3,.ogg,.m4a,.aac,.mid,.json"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length > 0) handlePickedFiles(files);
          e.target.value = "";
        }}
      />
      <input
        ref={dirInputRef}
        type="file"
        hidden
        multiple
        {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length > 0) void loadWebFiles(files);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function EmptyState({
  onPickDir,
  onPickFiles,
}: {
  onPickDir: () => void;
  onPickFiles: () => void;
}) {
  const [error, setError] = useState<string | null>(null);

  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <div className="flex max-w-md flex-col items-center gap-5 rounded-2xl border border-stroke bg-surface-1/60 px-10 py-9 text-center backdrop-blur-xl">
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-accent/15">
          <Music4 className="h-7 w-7 text-accent" />
        </div>
        <div>
          <h2 className="text-base font-medium text-content-1">打开一首曲子</h2>
          <p className="mt-1 text-xs leading-relaxed text-content-2">
            选择管线输出目录（含 .mid 与音频）或音频文件夹
            <br />
            拖拽打开将在 M3 支持
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="accent" onClick={onPickDir}>
            <FolderOpen className="h-4 w-4" />
            打开目录
          </Button>
          {!isTauri && (
            <Button onClick={onPickFiles}>选择文件</Button>
          )}
          <Button
            onClick={() =>
              void loadDemoProject().catch((e: Error) => setError(e.message))
            }
          >
            加载演示曲目
          </Button>
        </div>
        {error && <p className="text-xs text-red-400">{error}</p>}
      </div>
    </div>
  );
}
