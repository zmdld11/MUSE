import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";
import { spawn, exec } from "node:child_process";
import net from "node:net";
import path from "node:path";
import fs from "node:fs";

// 本地转写桥随 dev server 一起启动（2026-08-31 用户需求：开前端 = 管线
// 一起开，不再分开两条命令）。8420 已被占用则复用不重复起；退出时清理。
// MUSE_NO_PIPELINE=1 可关掉此行为。
function musePipelineBridge(): Plugin {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const script = path.join(root, "score_extraction", "pipeline_server.py");
  const children: ReturnType<typeof spawn>[] = [];

  const portOpen = (port: number) =>
    new Promise<boolean>((resolve) => {
      const s = net.connect({ port, host: "127.0.0.1" }, () => {
        s.destroy();
        resolve(true);
      });
      s.on("error", () => resolve(false));
    });

  const killAll = () => {
    for (const c of children.splice(0)) {
      if (c.pid === undefined) continue;
      // Windows 需整树击杀（python 可能再派生 MSST/demucs 子进程）
      if (process.platform === "win32")
        exec(`taskkill /PID ${c.pid} /T /F`);
      else c.kill("SIGTERM");
    }
  };

  return {
    name: "muse-pipeline-bridge",
    configureServer(server) {
      if (process.env.MUSE_NO_PIPELINE) {
        console.log("[pipeline-bridge] MUSE_NO_PIPELINE=1，跳过");
        return;
      }
      server.httpServer?.once("listening", async () => {
        const start = async (reason: string) => {
          if (await portOpen(8420)) {
            if (reason === "boot")
              console.log("[pipeline-bridge] 127.0.0.1:8420 已有管线服务，复用");
            return;
          }
          const pyPath = process.platform === "win32"
            ? path.join(root, "env", "python.exe")
            : path.join(root, "env", "bin", "python");
          if (!fs.existsSync(pyPath) || !fs.existsSync(script)) {
            console.log(`[pipeline-bridge] 未找到 ${pyPath} 或 pipeline_server.py，跳过`);
            return;
          }
          console.log(`[pipeline-bridge] ${reason === "boot" ? "启动" : "重启"}本地转写管线 http://127.0.0.1:8420`);
          const child = spawn(pyPath, [script], { cwd: root, stdio: "inherit" });
          children.push(child);
          child.on("exit", async (code) => {
            console.log(`[pipeline-bridge] 管线进程退出 code=${code}`);
            // vite 还在而管线挂了（崩溃/被外部结束）→ 3s 后自动拉起
            if (server.httpServer?.listening) {
              await new Promise((r) => setTimeout(r, 3000));
              if (!server.httpServer.listening) return;
              if (await portOpen(8420)) return;
              start("respawn").catch(() => {});
            }
          });
        };
        await start("boot");
      });
      process.once("exit", killAll);
      process.once("SIGINT", () => {
        killAll();
        process.exit(0);
      });
      process.once("SIGTERM", () => {
        killAll();
        process.exit(0);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), musePipelineBridge()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  // Tauri 开发模式约定：固定端口，不清屏
  clearScreen: false,
  server: { port: 5173, strictPort: true },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    target: "chrome105",
    minify: !process.env.TAURI_ENV_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
  },
});
