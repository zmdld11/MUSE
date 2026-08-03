"""VER2.0 Pipeline: 5-layer AMT architecture.

Layer 1: Acoustic frontend (librosa load)
Layer 2: Transcription model (basic-pitch raw onset/frame probs)
Layer 3: Frame-level post-processing (HMM + threshold + CC)
Layer 4: Note-level post-processing (onset refine + harmonic prune + merge)
Layer 5: Notation assembly (voice + score + export)
"""
import logging
import os
import json
import subprocess

import librosa
import numpy as np
import pretty_midi

from src.config import config
from src.bpm_detect import detect_bpm
from src.source_separate import separate_tracks
from src.transcriber import transcribe
from src.frame_post import process_frames
from src.note_post import refine_notes
from src.quantize_timing import quantize_onsets
from src.voice_assign import assign_voices
from src.chord_detect import detect_chords
from src.key_estimate import estimate_key
from src.score_assemble import assemble_score
from src.export_score import export_score

logger = logging.getLogger(__name__)

# MIDI program map for common instruments
_INST_PROGRAMS = {
    "piano": 0,
    "guitar": 24,
    "bass": 33,
    "violin": 40,
    "cello": 42,
    "strings": 48,
    "vocal": 52,
}


def _write_pretty_midi(notes: list[dict], midi_path: str, bpm: float,
                       inst_name: str = "piano",
                       time_sig: str = "4/4") -> None:
    """Write note list to a MIDI file via pretty_midi.

    Each note dict must have: onset, offset (seconds), pitch (MIDI).
    Optional: amplitude (RMS, mapped to velocity), voice (int).
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Explicit time signature (MuseScore needs this)
    ts_parts = time_sig.split("/")
    pm.time_signature_changes.append(pretty_midi.TimeSignature(
        numerator=int(ts_parts[0]), denominator=int(ts_parts[1]), time=0.0,
    ))

    # Group notes by voice, or put everything in one instrument
    voices = set(n.get("voice", 1) for n in notes)
    program = _INST_PROGRAMS.get(inst_name, 0)

    for voice_id in sorted(voices) if voices else [1]:
        inst = pretty_midi.Instrument(program=program)
        voice_notes = [n for n in notes if n.get("voice", 1) == voice_id]
        for n in voice_notes:
            amp = n.get("amplitude", 0.1)
            velocity = max(20, min(100, int(amp * 80 + 40)))
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=int(n["pitch"]),
                start=float(n["onset"]),
                end=float(n["offset"]),
            )
            inst.notes.append(note)
        pm.instruments.append(inst)

    pm.write(midi_path)


def _process_instrument(audio_path: str, inst_name: str, bpm: float,
                        output_dir: str,
                        chords: list[dict] | None = None) -> bool:
    """Run Layers 2-5 on one instrument track. Returns True on success."""
    # 输出目录用歌曲目录 (VER2.4: 钢琴轨传入降噪临时文件, 不能用文件名派生目录)
    # Layer 2: Transcription
    logger.info(f"  [{inst_name}] Layer 2: Transcribing...")
    result = transcribe(audio_path)
    if result["frame_probs"].size == 0 or result["frame_probs"].max() < 0.01:
        logger.warning(f"  [{inst_name}] No signal detected, skipping")
        return False

    # Layer 3: Frame-level post-processing
    logger.info(f"  [{inst_name}] Layer 3: Frame post-processing...")
    model_sr = result.get("sr", config.SR)
    model_hop = result.get("hop_length", config.HOP_LENGTH)
    candidates = process_frames(
        result["onset_probs"],
        result["frame_probs"],
        hop_length=model_hop,
        sr=model_sr,
    )
    if len(candidates) == 0:
        logger.warning(f"  [{inst_name}] No candidates after frame post-processing")
        return False

    # Load audio for onset refinement (use model's SR for consistency)
    audio, _sr = librosa.load(audio_path, sr=model_sr, mono=True)

    # Layer 4: Note-level post-processing
    logger.info(f"  [{inst_name}] Layer 4: Note post-processing...")
    # VER2.3: BP 后处理候选直接转秒 (跳过谐波/调性过滤 — A/B 证明它们砍 50% 真音符)
    # VER2.4: 膝跳回声过滤 — 删 (onset_prob<0.15 且 confidence<0.5): 无真实击键的弱音
    # (melodia_trick 捡起的衰减尾巴/共鸣回声, 真实录音 F1 0.56→0.65)
    notes = [{
        "onset": round(c.get("onset_time", c["onset_frame"] * model_hop / model_sr), 4),
        "offset": round(c.get("offset_time", c["offset_frame"] * model_hop / model_sr), 4),
        "pitch": c["pitch"],
        "confidence": c["confidence"],
        "amplitude": 0.1,
    } for c in candidates
        if not (c.get("onset_prob", 1.0) < 0.15 and c["confidence"] < 0.5)]
    if len(notes) == 0:
        logger.warning(f"  [{inst_name}] No notes after refinement")
        return False

    logger.info(f"  [{inst_name}] {len(notes)} notes after refinement")

    # Layer 4b: Timing quantization (disabled — basic-pitch onset errors > grid/2)
    # logger.info(f"  [{inst_name}] Layer 4b: Timing quantization...")
    # notes = quantize_onsets(notes, bpm, config.DEFAULT_TIME_SIG)

    # Layer 5a: Voice assignment
    notes = assign_voices(notes, inst_name)

    # Layer 5b: Key estimation (from all notes accumulated)
    key_sig = estimate_key(notes)

    # Layer 5c: Score assembly
    note_count = len(notes)
    score = assemble_score(
        inst_name, notes, bpm, key_sig,
        config.DEFAULT_TIME_SIG, chords if inst_name == "guitar" else None,
    )

    # Layer 5d: Export — MIDI → MuseScore CLI (primary), export_score fallback
    output_stem = os.path.join(output_dir, inst_name)
    os.makedirs(os.path.dirname(output_stem), exist_ok=True)

    midi_path = output_stem + ".mid"
    xml_path = output_stem + ".musicxml"
    success = False

    # Export: pretty_midi → .mid → MuseScore CLI → fix BPM → .musicxml
    try:
        import re as _re
        _write_pretty_midi(notes, midi_path, bpm, inst_name, config.DEFAULT_TIME_SIG)
        logger.info(f"  [{inst_name}] MIDI written: {midi_path}")

        musescore = config.MUSESCORE_PATH
        if os.path.exists(musescore):
            result = subprocess.run(
                [musescore, "-f", "-o", xml_path, midi_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and os.path.exists(xml_path):
                # Fix BPM: MuseScore CLI uses default 117, patch to actual bpm
                with open(xml_path, "r", encoding="utf-8") as _f:
                    _xml = _f.read()
                _xml = _re.sub(r'(<sound[^"]*tempo=")117("[^>]*>)', rf'\g<1>{bpm}\g<2>', _xml)
                _xml = _re.sub(r'<per-minute>117</per-minute>', f'<per-minute>{int(bpm)}</per-minute>', _xml)
                with open(xml_path, "w", encoding="utf-8") as _f:
                    _f.write(_xml)
                logger.info(f"  [{inst_name}] MuseScore XML ({note_count} notes): {xml_path}")
                success = True
            else:
                logger.warning(f"  [{inst_name}] MuseScore CLI failed (rc={result.returncode})")
        else:
            logger.warning(f"  [{inst_name}] MuseScore not found at {musescore}")
    except Exception as exc:
        logger.warning(f"  [{inst_name}] MuseScore export failed: {exc}")

    # Fallback: score_assemble → export_score
    if not success:
        try:
            result_path = export_score(score, output_stem)
            if result_path is not None:
                logger.info(f"  [{inst_name}] Exported via music21 fallback ({note_count} notes): {result_path}")
                success = True
        except Exception as exc:
            logger.error(f"  [{inst_name}] music21 export crashed: {exc}", exc_info=True)

    if not success:
        logger.error(f"  [{inst_name}] All export methods failed")
        return False
    return True


def run_pipeline(audio_path: str, output_dir: str | None = None) -> str:
    """Full 5-layer AMT pipeline."""
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, song_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger.info(f"=== Pipeline VER2.0 start: {song_name} ===")

    # Step 1: BPM
    bpm = detect_bpm(audio_path) or config.DEFAULT_BPM
    if bpm > 120:
        logger.info(f"  Halving BPM: {bpm} → {bpm / 2:.1f}")
        bpm = round(bpm / 2, 1)
    logger.info(f"[1/5] BPM: {bpm}")

    # Step 2: Source separation (Layer 1)
    logger.info("[2/5] Source separation (htdemucs_6s)...")
    tracks = separate_tracks(audio_path, output_dir)

    # Step 3-5: Process each track
    # 2026-08-02: ONLY_PIANO 时只转录 piano (VER2.4 需分离钢琴轨做 wiener 降噪,
    # 但 bass/guitar/vocals 未适配, 不再生成)
    all_notes = {}
    for inst_name, wav_path in tracks.items():
        if inst_name == "drums":
            all_notes[inst_name] = []
            continue
        if config.ONLY_PIANO and inst_name != "piano":
            logger.info(f"  [ONLY_PIANO] 跳过 {inst_name} (未适配)")
            continue

        # Piano: VER2.4 (2026-08-02) — 用钢琴轨而非原音频, 并 wiener 降噪.
        # A/B 证明: 原音频有鼓/贝斯/人声串扰, 分离钢琴轨的 demucs 雪花会抖模型输入;
        # 钢琴轨+wiener 真实录音 F1 0.46→0.63 (弱音过滤组合后 0.65).
        if inst_name == "piano":
            from scipy.signal import wiener
            import scipy.io.wavfile as _wf
            audio, _sr = librosa.load(wav_path, sr=config.SR, mono=True)
            audio_wn = np.nan_to_num(wiener(audio, mysize=9))
            src_audio = os.path.join(output_dir, "piano_denoised.wav")
            _wf.write(src_audio, config.SR,
                      (audio_wn * 32767).clip(-32768, 32767).astype(np.int16))
        else:
            src_audio = wav_path
        try:
            _process_instrument(src_audio, inst_name, bpm, output_dir)
        except Exception as exc:
            logger.error(f"  [{inst_name}] Pipeline crashed: {exc}", exc_info=True)

    # Write info.json
    info = {"song": song_name, "bpm": bpm, "time_signature": config.DEFAULT_TIME_SIG}
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    logger.info(f"=== Pipeline complete: {output_dir} ===")
    return output_dir
