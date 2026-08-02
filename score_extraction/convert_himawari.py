"""Convert custom notation (大一C++项目格式) to MIDI + MusicXML piano grand staff."""
import os
import pretty_midi

SRC = r"d:\program_project\MUSE\music\夜の向日葵 - 松本文紀.txt"
OUT = r"d:\program_project\MUSE\score_extraction\output\himawari_reference"

# C_Scale[octave][note_idx], matching C++: default lvl=3 → C4=60
C_SCALE = [
    [24,26,28,29,31,33,35],
    [36,38,40,41,43,45,47],
    [48,50,52,53,55,57,59],
    [60,62,64,65,67,69,71],   # default octave
    [72,74,76,77,79,81,83],
    [84,86,88,89,91,93,95],
    [96,98,100,101,103,105,107],
]
C_SHARP = [
    [25,27,-1,30,32,34,-1],
    [37,39,-1,42,44,46,-1],
    [49,51,-1,54,56,58,-1],
    [61,63,-1,66,68,70,-1],
    [73,75,-1,78,80,82,-1],
    [85,87,-1,90,92,94,-1],
    [97,99,-1,102,104,106,-1],
]


def parse_line(line, delay_ms=731):
    """Character-by-character parser matching C++ play_single().

    Returns (events, total_time_sec).
    events: list of (onset_sec, offset_sec, midi_pitch)
    total_time_sec: includes rests (matching C++ tick accumulation).
    """
    events = []
    current_time = 0.0
    beat_sec = delay_ms / 1000.0
    base_tick = 672
    tick_sec = beat_sec / base_tick

    s = line.strip()
    is_chord = False
    nbuf = []          # buffered MIDI pitches (0 = rest)
    ctn = base_tick    # current tick counter for duration

    i, n = 0, len(s)
    while i < n:
        c = s[i]

        if c in '[{':
            is_chord = True
            i += 1
        elif c in ']}':
            is_chord = False
            i += 1
        elif c == '|':
            i += 1
        elif c == ' ':
            if not is_chord and nbuf:
                dur_sec = ctn * tick_sec
                for midi in nbuf:
                    if midi != 0:
                        events.append((current_time, current_time + dur_sec, midi))
                nbuf.clear()
                current_time += ctn * tick_sec
                ctn = base_tick
            i += 1
        elif c == '_':
            ctn //= 2
            i += 1
        elif c == '.':
            ctn = int(ctn * 1.5)
            i += 1
        elif c == '-':
            ctn += base_tick
            i += 1
        elif c == '0':
            nbuf.append(0)
            i += 1
        elif c.isdigit():
            note_idx = int(c) - 1
            lvl = 3
            is_sharp = False
            i += 1
            while i < n and s[i] in ('^', ',', '#'):
                if s[i] == '^': lvl += 1
                elif s[i] == ',': lvl -= 1
                elif s[i] == '#': is_sharp = True
                i += 1
            lvl = max(0, min(6, lvl))
            if is_sharp:
                midi = C_SHARP[lvl][note_idx]
                if midi < 0:
                    midi = C_SCALE[lvl][note_idx] + 1
            else:
                midi = C_SCALE[lvl][note_idx]
            nbuf.append(midi)
        else:
            i += 1

    # Flush trailing buffer (matching C++ s+' ')
    if nbuf:
        dur_sec = ctn * tick_sec
        for midi in nbuf:
            if midi != 0:
                events.append((current_time, current_time + dur_sec, midi))
        current_time += ctn * tick_sec

    return events, current_time


def main():
    import argparse
    ap = argparse.ArgumentParser(description="txt 简谱 → MIDI + MusicXML")
    ap.add_argument("--transpose", type=int, default=0,
                    help="整体移调半音数 (E 大调 = +4), 默认 0 (C 大调)")
    ap.add_argument("--out", default=OUT, help="输出目录")
    args = ap.parse_args()
    transpose = args.transpose
    out_dir = args.out

    with open(SRC, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    delay_ms = int(lines[0].strip())
    bpm = 60000 / delay_ms
    print(f"Delay: {delay_ms}ms, BPM: {bpm:.1f}")

    lines = lines[1:]

    # Filter: skip empty lines and numeric tempo-change lines
    def _is_numeric(s):
        s = s.strip().replace('.', '').replace('-', '')
        return s.isdigit()

    music_lines = []
    for i, l in enumerate(lines):
        stripped = l.strip()
        if not stripped:
            continue
        if _is_numeric(stripped):
            delay_ms = int(stripped)
            bpm = 60000 / delay_ms
            print(f"  Tempo change at line {i}: {delay_ms}ms → {bpm:.1f} BPM")
            continue
        music_lines.append((i, stripped))

    # Pairwise timing: C++ play(s1,s2) plays right & left in parallel.
    # Next pair starts after max(right_dur, left_dur).
    right_events = []
    left_events = []
    pair_time = 0.0

    for li in range(0, len(music_lines) - 1, 2):
        r_events, r_dur = parse_line(music_lines[li][1], delay_ms)
        l_events, l_dur = parse_line(music_lines[li + 1][1], delay_ms)

        for e in r_events:
            right_events.append((e[0] + pair_time, e[1] + pair_time, e[2]))
        for e in l_events:
            left_events.append((e[0] + pair_time, e[1] + pair_time, e[2]))

        pair_time += max(r_dur, l_dur)

    print(f"Right hand: {len(right_events)} notes, Left hand: {len(left_events)} notes")

    # 整体移调 (E 大调 GT = +4 半音, 主旋律线条不变)
    if transpose:
        right_events = [(o, f, p + transpose) for o, f, p in right_events]
        left_events = [(o, f, p + transpose) for o, f, p in left_events]
        print(f"Transposed +{transpose} semitones")

    # ---- MIDI output ----
    # 2026-08-02 修复: 左右手分到不同 instrument (通道).
    # 之前混在同一通道, pretty_midi 写盘时重叠同音高音符被 note-on 重触发截断
    # (右手长音被左手同音高短音吃掉尾段, 34 处与 txt 不一致, 播放听感也受影响).
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    for name, events in [("Right", right_events), ("Left", left_events)]:
        inst = pretty_midi.Instrument(program=0, name=name)
        for onset, offset, pitch in events:
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=onset, end=offset))
        pm.instruments.append(inst)

    os.makedirs(out_dir, exist_ok=True)
    midi_path = os.path.join(out_dir, "himawari_reference.mid")
    pm.write(midi_path)
    print(f"MIDI: {midi_path}")

    # ---- MusicXML: piano grand staff via music21 ----
    # Voice 1 = right (treble), Voice 2 = left (bass)
    # Notes grouped by onset → chords; split at bar lines → tied fragments.
    from music21 import stream, meter, key, tempo, note, chord, clef, tie
    from collections import defaultdict

    sec_per_beat = delay_ms / 1000.0
    ql_per_bar = 4.0

    STD_DURS = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

    def ql(sec):
        return sec / sec_per_beat

    def quantize(d):
        d = max(0.125, d)
        return min(STD_DURS, key=lambda x: abs(x - d))

    def split_bars(onset_ql, dur_ql):
        """Split a note at bar lines → [(bar_idx, offset_in_bar, frag_dur_ql, tie_type)]."""
        frags = []
        rem = dur_ql
        cur = onset_ql
        while rem > 0.001:
            bi = int(cur / ql_per_bar)
            available = (bi + 1) * ql_per_bar - cur
            frag = min(rem, available)
            frag_q = quantize(frag)
            off = cur - bi * ql_per_bar
            is_first = len(frags) == 0
            after = rem - frag_q
            if is_first and after <= 0.001:
                tt = None
            elif is_first:
                tt = "start"
            elif after <= 0.001:
                tt = "stop"
            else:
                tt = "continue"
            frags.append((int(bi), off, frag_q, tt))
            cur += frag_q
            rem -= frag_q
            if frag_q < 0.001:
                break
        return frags

    # Phase 1: group events by onset → chords, then split at bar lines
    # measure_voices[(bar_idx, voice_id)] = [(offset_in_bar, dur_ql, tie_type, [pitches])]
    measure_voices = defaultdict(list)

    for voice_id, events in [(1, right_events), (2, left_events)]:
        # Group by onset
        onset_groups = {}
        for onset, offset, pitch in events:
            onset_ql = round(ql(onset) * 8) / 8
            if onset_ql not in onset_groups:
                onset_groups[onset_ql] = {"pitches": [], "max_offset_sec": 0}
            onset_groups[onset_ql]["pitches"].append(pitch)
            onset_groups[onset_ql]["max_offset_sec"] = max(
                onset_groups[onset_ql]["max_offset_sec"], offset)

        for onset_ql, grp in onset_groups.items():
            dur_ql = round(ql(grp["max_offset_sec"]) - onset_ql, 3)
            dur_ql = max(dur_ql, 0.125)

            for bi, off, d, tt in split_bars(onset_ql, dur_ql):
                measure_voices[(bi, voice_id)].append((off, d, tt, sorted(grp["pitches"])))

    # Phase 2: build measures per voice, then XML post-process for grand staff
    measures = {}
    for (mi, voice_id), items in measure_voices.items():
        if mi not in measures:
            m = stream.Measure()
            m.number = mi + 1
            measures[mi] = m

        v = stream.Voice()
        v.id = str(voice_id)

        items.sort(key=lambda x: x[0])
        j = 0
        while j < len(items):
            same = [items[j]]
            cur_off = items[j][0]
            j += 1
            while j < len(items) and abs(items[j][0] - cur_off) < 0.001:
                same.append(items[j])
                j += 1

            all_pitches = []
            for _, _d, _tt, pitches in same:
                all_pitches.extend(pitches)
            all_pitches = sorted(set(all_pitches))
            sample_dur = same[0][1]
            sample_tie = same[0][2]

            if len(all_pitches) == 1:
                n_obj = note.Note(all_pitches[0])
            else:
                n_obj = chord.Chord([note.Note(p) for p in all_pitches])

            n_obj.duration.quarterLength = sample_dur
            if sample_tie is not None:
                n_obj.tie = tie.Tie(sample_tie)
            if voice_id == 2:
                n_obj.stemDirection = "down"

            v.insert(cur_off, n_obj)

        measures[mi].insert(0, v)

    # Phase 3: build score
    if 0 in measures:
        m0 = measures[0]
        m0.insert(0, tempo.MetronomeMark(number=int(bpm)))
        m0.insert(0, key.Key("C"))
        m0.insert(0, meter.TimeSignature("4/4"))
        m0.insert(0, clef.TrebleClef())

    score = stream.Score()
    piano_part = stream.Part()
    piano_part.partName = "Piano"
    for mi in sorted(measures.keys()):
        piano_part.append(measures[mi])
    score.insert(0, piano_part)

    # Write initial XML, then post-process for grand staff
    import tempfile
    tmp_xml = os.path.join(tempfile.gettempdir(), "_himawari_tmp.musicxml")
    score.write("musicxml", fp=tmp_xml)

    with open(tmp_xml, "r", encoding="utf-8") as f:
        xml = f.read()

    # --- Post-process: add grand staff (staves=2, bass clef, staff assignments) ---
    import re as _re

    # 1. Replace first </attributes> with staves + bass clef + </attributes>
    # Extract indentation from the original </attributes> line
    m = _re.search(r'^(\s*)</attributes>', xml, _re.MULTILINE)
    if m:
        indent = m.group(1)
        extra = (
            f'{indent}<staves>2</staves>\n'
            f'{indent}  <staff-details number="2">\n'
            f'{indent}    <staff-lines>5</staff-lines>\n'
            f'{indent}  </staff-details>\n'
            f'{indent}  <clef number="2">\n'
            f'{indent}    <sign>F</sign>\n'
            f'{indent}    <line>4</line>\n'
            f'{indent}  </clef>\n'
            f'{indent}</attributes>'
        )
        xml = _re.sub(
            r'^(\s*)</attributes>',
            lambda _m: extra,
            xml,
            count=1,
            flags=_re.MULTILINE
        )

    # 2. Add <staff>1</staff> to voice=1 notes, <staff>2</staff> to voice=2 notes
    xml = _re.sub(
        r'(<voice>1</voice>)',
        r'\1\n            <staff>1</staff>',
        xml
    )
    xml = _re.sub(
        r'(<voice>2</voice>)',
        r'\1\n            <staff>2</staff>',
        xml
    )

    xml_path = os.path.join(out_dir, "himawari_reference.musicxml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)
    os.unlink(tmp_xml)
    print(f"MusicXML: {xml_path}")
    print("Done!")


if __name__ == "__main__":
    main()
