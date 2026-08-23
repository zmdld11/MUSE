use serde::Serialize;
use std::fs;
use std::path::Path;

/// 目录扫描结果（前端 features/library 消费）
#[derive(Serialize)]
pub struct DirScan {
    pub audio: Option<String>,
    pub mids: Vec<String>,
    pub info: Option<String>,
}

const AUDIO_EXTS: [&str; 6] = ["wav", "flac", "mp3", "ogg", "m4a", "aac"];

/// 扫描目录一层：找音频、.mid、info.json（管线输出目录结构）
#[tauri::command]
fn scan_dir(dir: String) -> Result<DirScan, String> {
    let mut result = DirScan {
        audio: None,
        mids: Vec::new(),
        info: None,
    };
    let entries = fs::read_dir(Path::new(&dir)).map_err(|e| format!("无法读取目录: {e}"))?;
    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .collect();
    files.sort_by_key(|e| e.file_name());

    for entry in files {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        let ext = path
            .extension()
            .map(|x| x.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if AUDIO_EXTS.contains(&ext.as_str()) {
            // 多个音频时优先无损（wav/flac）
            let better = match &result.audio {
                None => true,
                Some(prev) => {
                    let prev_lossless = prev.to_lowercase().ends_with("wav")
                        || prev.to_lowercase().ends_with("flac");
                    !prev_lossless && (ext == "wav" || ext == "flac")
                }
            };
            if better {
                result.audio = Some(path.to_string_lossy().to_string());
            }
        } else if ext == "mid" {
            result.mids.push(path.to_string_lossy().to_string());
        } else if name == "info.json" {
            result.info = Some(path.to_string_lossy().to_string());
        }
    }
    Ok(result)
}

/// 读任意文件字节：二进制走 tauri::ipc::Response（不经 JSON 序列化，音频可达几十 MB）
#[tauri::command]
fn read_bytes(path: String) -> Result<tauri::ipc::Response, String> {
    fs::read(Path::new(&path))
        .map(tauri::ipc::Response::new)
        .map_err(|e| format!("无法读取文件 {path}: {e}"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![scan_dir, read_bytes])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
