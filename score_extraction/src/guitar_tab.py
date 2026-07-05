import logging
from src.config import config

logger = logging.getLogger(__name__)

OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]  # low E -> high E


def get_positions(midi_pitch: int) -> list[tuple[int, int]]:
    positions = []
    for s, open_midi in enumerate(OPEN_STRING_MIDI):
        fret = midi_pitch - open_midi
        if 0 <= fret <= config.MAX_FRET:
            positions.append((s, fret))
    return positions


def assign_guitar_fingering(notes: list[dict]) -> list[dict]:
    if len(notes) == 0:
        return []

    candidates = []
    for note in notes:
        pos = get_positions(note["pitch"])
        if not pos:
            pos = [(0, -1)]  # sentinel for unplayable
        candidates.append(pos)

    n = len(notes)
    dp = []
    for i in range(n):
        dp_i = []
        for j, (s, f) in enumerate(candidates[i]):
            if i == 0:
                cost = config.OPEN_STRING_BIAS if f == 0 else 0.0
                dp_i.append((cost, -1))
            else:
                best_cost, best_prev = float("inf"), -1
                for k, (ps, pf) in enumerate(candidates[i - 1]):
                    if pf == -1 or f == -1:
                        step_cost = 5.0
                    else:
                        step_cost = (
                            config.FRET_WEIGHT * abs(f - pf)
                            + config.STRING_WEIGHT * abs(s - ps)
                        )
                        if f == 0:
                            step_cost += config.OPEN_STRING_BIAS
                    total = dp[i - 1][k][0] + step_cost
                    if total < best_cost:
                        best_cost, best_prev = total, k
                dp_i.append((best_cost, best_prev))
        dp.append(dp_i)

    # Backtrack
    result = []
    best_last = min(range(len(dp[-1])), key=lambda j: dp[-1][j][0])
    for i in range(n - 1, -1, -1):
        s, f = candidates[i][best_last]
        note_copy = dict(notes[i])
        note_copy["string"] = int(s) + 1
        note_copy["fret"] = int(f)
        result.append(note_copy)
        _, best_last = dp[i][best_last] if i > 0 else (0, -1)

    result.reverse()
    return result
