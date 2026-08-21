"""Parser for Guitar Pro 7/8 ``.gp`` files.

Modern ``.gp`` files are ZIP containers. Musical content is stored in
``Content/score.gpif`` as XML. This parser has no third-party Guitar Pro
dependency and expands reusable Beat/Note entities into concrete events.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

NOTE_VALUE_QUARTERS = {
    "Long": 16.0, "DoubleWhole": 8.0, "Whole": 4.0, "Half": 2.0,
    "Quarter": 1.0, "Eighth": 0.5, "16th": 0.25, "32nd": 0.125,
    "64th": 0.0625, "128th": 0.03125,
}


@dataclass
class GuitarpNote:
    onset: float
    offset: float
    duration: float
    pitch: int
    string: int | None
    fret: int | None
    bar: int
    beat: str
    note_id: str
    voice_id: str
    tie_origin: bool = False
    tie_destination: bool = False


def _property_map(note: ET.Element) -> dict[str, ET.Element]:
    return {
        node.attrib.get("name", ""): node
        for node in note.findall("./Properties/Property")
    }


def _rhythm_quarters(rhythm: ET.Element) -> float:
    value = rhythm.findtext("NoteValue", "Quarter")
    if value not in NOTE_VALUE_QUARTERS:
        raise ValueError(f"Unsupported GP rhythm: {value}")
    quarters = NOTE_VALUE_QUARTERS[value]
    dot = rhythm.find("AugmentationDot")
    if dot is not None:
        quarters *= 1.0 + 0.5 * int(dot.attrib.get("count", "1"))
    tuplet = rhythm.find("Tuplet")
    if tuplet is not None:
        numerator = float(tuplet.findtext("Numerator", "3"))
        denominator = float(tuplet.findtext("Denominator", "2"))
        if numerator <= 0 or denominator <= 0:
            raise ValueError(f"Invalid tuplet in rhythm {rhythm.attrib.get('id')}")
        quarters *= numerator / denominator
    return quarters


def _tempo_map(root: ET.Element) -> list[tuple[int, float]]:
    output = []
    for node in root.findall("./MasterTrack/Automations/Automation"):
        if node.findtext("Type") != "Tempo":
            continue
        value = node.findtext("Value", "").split()
        if value:
            output.append((int(node.findtext("Bar", "0")), float(value[0])))
    return sorted(output) or [(0, 120.0)]


def _bar_quarters(time_signature: str) -> float:
    numerator, denominator = map(int, time_signature.split("/"))
    return numerator * 4.0 / denominator


def parse_gpif(source: str | Path) -> dict:
    """Parse a modern ``.gp`` container or an extracted ``score.gpif`` XML."""
    source = Path(source)
    embedded_audio = None
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            root = ET.fromstring(archive.read("Content/score.gpif"))
            for name in archive.namelist():
                if name.startswith("Content/Assets/") and name.lower().endswith(
                        (".flac", ".mp3", ".ogg", ".wav")):
                    embedded_audio = name
                    break
    else:
        root = ET.fromstring(source.read_bytes())

    rhythms = {node.attrib["id"]: _rhythm_quarters(node) for node in root.find("Rhythms")}
    beats = {node.attrib["id"]: node for node in root.find("Beats")}
    voices = {node.attrib["id"]: node for node in root.find("Voices")}
    bars = {node.attrib["id"]: node for node in root.find("Bars")}
    notes = {node.attrib["id"]: node for node in root.iter("Note")}
    master_bars = root.find("MasterBars").findall("MasterBar")
    tempo_automations = _tempo_map(root)

    expanded: list[GuitarpNote] = []
    current_time = 0.0
    tempo_index = 0
    current_bpm = tempo_automations[0][1]
    bar_reports = []

    for bar_index, master_bar in enumerate(master_bars):
        while (tempo_index + 1 < len(tempo_automations) and
               tempo_automations[tempo_index + 1][0] <= bar_index):
            tempo_index += 1
            current_bpm = tempo_automations[tempo_index][1]
        time_signature = master_bar.findtext("Time", "4/4")
        bar_quarters = _bar_quarters(time_signature)
        second_per_quarter = 60.0 / current_bpm
        occupied_quarters = 0.0

        for bar_id in master_bar.findtext("Bars", "").split():
            bar = bars.get(bar_id)
            if bar is None:
                continue
            for voice_id in bar.findtext("Voices", "-1 -1 -1 -1").split():
                if voice_id == "-1":
                    continue
                voice = voices.get(voice_id)
                if voice is None:
                    continue
                beat_time = current_time
                voice_quarters = 0.0
                for beat_id in voice.findtext("Beats", "").split():
                    beat = beats.get(beat_id)
                    if beat is None:
                        continue
                    rhythm = beat.find("Rhythm")
                    quarters = rhythms.get(rhythm.attrib.get("ref", ""), 0.0)
                    duration = quarters * second_per_quarter
                    for note_id in beat.findtext("Notes", "").split():
                        note = notes.get(note_id)
                        if note is None:
                            continue
                        properties = _property_map(note)
                        midi = properties.get("Midi")
                        string = properties.get("String")
                        fret = properties.get("Fret")
                        tie = note.find("Tie")
                        expanded.append(GuitarpNote(
                            onset=float(beat_time),
                            offset=float(beat_time + duration),
                            duration=float(duration),
                            pitch=int(midi.findtext("Number")) if midi is not None else -1,
                            string=int(string.findtext("String")) if string is not None else None,
                            fret=int(fret.findtext("Fret")) if fret is not None else None,
                            bar=bar_index,
                            beat=beat_id,
                            note_id=note_id,
                            voice_id=voice_id,
                            tie_origin=tie is not None and tie.attrib.get("origin") == "true",
                            tie_destination=tie is not None and tie.attrib.get("destination") == "true",
                        ))
                    beat_time += duration
                    voice_quarters += quarters
                occupied_quarters = max(occupied_quarters, voice_quarters)

        bar_reports.append({
            "bar": bar_index,
            "time_signature": time_signature,
            "nominal_quarters": bar_quarters,
            "occupied_quarters": occupied_quarters,
            "start": current_time,
            "end": current_time + bar_quarters * second_per_quarter,
        })
        current_time += bar_quarters * second_per_quarter

    expanded.sort(key=lambda event: (event.onset, event.pitch, event.string or -1))
    tracks = root.findall("./Tracks/Track")
    return {
        "source": str(source),
        "embedded_audio": embedded_audio,
        "gp_version": root.findtext("GPVersion"),
        "tempo_automations": [{"bar": bar, "bpm": bpm} for bar, bpm in tempo_automations],
        "master_bars": len(master_bars),
        "tracks": [{
            "id": track.attrib.get("id", ""),
            "name": track.findtext("Name", ""),
            "instrument_type": track.findtext("./InstrumentSet/Type", ""),
            "program": int(track.findtext("./Sounds/Sound/MIDI/Program", "25") or 25),
            "tuning": [int(value) for value in track.findtext(
                "./Staves/Staff/Properties/Property[@name='Tuning']/Pitches",
                "40 45 50 55 59 64").split()],
        } for track in tracks],
        "notes": [asdict(event) for event in expanded],
        "bars": bar_reports,
        "duration": current_time,
    }


def extract_embedded_audio(gp_path: str | Path, output_path: str | Path) -> Path:
    with zipfile.ZipFile(gp_path) as archive:
        candidates = [name for name in archive.namelist()
                      if name.startswith("Content/Assets/") and name.lower().endswith(
                          (".flac", ".mp3", ".ogg", ".wav"))]
        if not candidates:
            raise FileNotFoundError(f"No embedded audio in {gp_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(archive.read(candidates[0]))
        return output_path


def write_reference_midi(parsed: dict, midi_path: str | Path) -> Path:
    import pretty_midi

    if not parsed["tracks"]:
        raise ValueError("GPIF has no tracks")
    bpm = parsed["tempo_automations"][0]["bpm"]
    program = parsed["tracks"][0].get("program", 25)
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=480)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=program, name="Guitar Reference")
    for event in parsed["notes"]:
        pitch = int(event["pitch"])
        if not (0 <= pitch <= 127):
            continue
        instrument.notes.append(pretty_midi.Note(
            velocity=80, pitch=pitch,
            start=max(0.0, float(event["onset"])),
            end=max(0.05, float(event["offset"]))))
    pm.instruments.append(instrument)
    midi_path = Path(midi_path)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(midi_path))
    return midi_path


def save_parsed(parsed: dict, json_path: str | Path) -> Path:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(parsed, stream, ensure_ascii=False, indent=2)
    return json_path
