import logging, os, json, tempfile
from src.config import config
from src.bpm_detect import detect_bpm
from src.source_separate import separate_tracks
from src.pitch_detect import detect_pitch_mono, detect_pitch_piano
from src.chord_detect import detect_chords
from src.key_estimate import estimate_key
from src.guitar_tab import assign_guitar_fingering
from src.score_assemble import assemble_score
from src.export_score import export_score

logger = logging.getLogger(__name__)


def run_pipeline(audio_path: str, output_dir: str | None = None) -> str:
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, song_name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                                  logging.StreamHandler()])

    logger.info(f"=== Pipeline start: {song_name} ===")

    # 1. BPM
    bpm = detect_bpm(audio_path) or config.DEFAULT_BPM
    logger.info(f"[1/6] BPM: {bpm}")

    # 2. Source separation
    logger.info("[2/6] Source separation (htdemucs_6s)...")
    tracks = separate_tracks(audio_path, output_dir)

    # 3. Pitch detection per track
    logger.info("[3/6] Pitch detection...")
    all_notes = {}
    for inst_name, wav_path in tracks.items():
        if inst_name == "drums":
            all_notes[inst_name] = []  # skip drums
            continue
        elif inst_name == "piano":
            notes = detect_pitch_piano(wav_path)
        else:
            notes = detect_pitch_mono(wav_path)
        all_notes[inst_name] = notes
        logger.info(f"  {inst_name}: {len(notes)} notes")

    # Bass octave shift: move low pitches up one octave for readability
    if "bass" in all_notes:
        for n in all_notes["bass"]:
            if n["pitch"] < 50:
                n["pitch"] += 12

    # 4. Guitar chords (best-effort)
    chords = []
    if "guitar" in tracks:
        logger.info("[4/6] Chord detection (guitar)...")
        chords = detect_chords(tracks["guitar"])

    # 5. Key estimation
    logger.info("[5/6] Key estimation...")
    combined_notes = []
    for notes in all_notes.values():
        combined_notes.extend(notes)
    key_sig = estimate_key(combined_notes)
    logger.info(f"  Key: {key_sig}")

    # 6. Score assembly + export
    logger.info("[6/6] Score assembly + export...")
    for inst_name, notes in all_notes.items():
        if inst_name == "drums":
            continue  # drum score requires different logic; skip for VER1.0
        if len(notes) == 0:
            logger.warning(f"  {inst_name}: no notes, skipping score")
            continue

        is_guitar = (inst_name == "guitar")
        if is_guitar:
            notes = assign_guitar_fingering(notes)
            logger.info(f"  {inst_name}: guitar fingering assigned")

        score = assemble_score(
            inst_name, notes, bpm, key_sig,
            config.DEFAULT_TIME_SIG, chords if is_guitar else None,
            is_guitar,
        )

        output_stem = os.path.join(output_dir, inst_name)
        export_score(score, output_stem)

    # Write info.json
    info = {
        "song": song_name,
        "bpm": bpm,
        "key": key_sig,
        "time_signature": config.DEFAULT_TIME_SIG,
        "tracks": {
            name: len(notes) for name, notes in all_notes.items()
        },
        "chord_count": len(chords),
    }
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    logger.info(f"=== Pipeline complete: {output_dir} ===")
    return output_dir
