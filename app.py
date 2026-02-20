from __future__ import annotations

import os
from datetime import datetime
import json
from pathlib import Path

import gradio as gr
import mido
import torch
from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.config import DELTA, MAX_TIME_IN_SECONDS, TIME_RESOLUTION
from anticipation.convert import events_to_midi, midi_to_events
from anticipation.sample import generate
from anticipation.vocab import CONTROL_OFFSET, DUR_OFFSET, TIME_OFFSET


_MODEL = None
_MODEL_ID = None


def _pick_device(device: str) -> str:
    device = (device or "auto").lower().strip()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("ANTICIPATION_DEVICE must be 'auto', 'cpu', or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_model() -> AutoModelForCausalLM:
    global _MODEL, _MODEL_ID
    if _MODEL is not None:
        return _MODEL
    model_id = 'stanford-crfm/music-medium-800k'
    device = _pick_device(os.environ.get("ANTICIPATION_DEVICE", "auto"))
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    model.to(device)

    _MODEL = model
    _MODEL_ID = model_id
    return model


def _top_p() -> float:
    value = os.environ.get("ANTICIPATION_TOP_P", "0.98")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError("ANTICIPATION_TOP_P must be a float") from exc


def _extract_tempo_and_tpb(midi_path: str) -> tuple[int, int]:
    midi = mido.MidiFile(midi_path)
    tempo = 500000
    for track in midi.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = int(msg.tempo)
                return tempo, int(midi.ticks_per_beat)
    return tempo, int(midi.ticks_per_beat)


def _apply_tempo_and_tpb(mid: mido.MidiFile, tempo_us: int, tpb: int) -> mido.MidiFile:
    src_tpb = int(mid.ticks_per_beat)
    src_tempo = 500000
    scale = (tpb * src_tempo) / (src_tpb * tempo_us)

    out = mido.MidiFile(ticks_per_beat=tpb)
    for idx, track in enumerate(mid.tracks):
        out_track = mido.MidiTrack()
        if idx == 0:
            out_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        for msg in track:
            new_time = int(round(msg.time * scale))
            out_track.append(msg.copy(time=new_time))
        out.tracks.append(out_track)
    return out


def _empty_midi_with_tempo(tempo_us: int, tpb: int) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    mid.tracks.append(track)
    return mid


def _events_to_midi_or_empty(events: list[int], tempo_us: int, tpb: int) -> mido.MidiFile:
    if not events:
        return _empty_midi_with_tempo(tempo_us, tpb)
    mid = events_to_midi(events)
    return _apply_tempo_and_tpb(mid, tempo_us, tpb)


def _clear_region_events(events: list[int], start_s: float, end_s: float) -> list[int]:
    # events are [time, duration, note] tokens with TIME_OFFSET/DUR_OFFSET applied.
    # time/duration are in ticks at TIME_RESOLUTION (100 ticks/s = 10ms).
    start_ticks = int(round(TIME_RESOLUTION * start_s))
    end_ticks = int(round(TIME_RESOLUTION * end_s))
    out: list[int] = []
    for time, dur, note in zip(events[0::3], events[1::3], events[2::3]):
        t = time - TIME_OFFSET
        d = dur - DUR_OFFSET
        # Truncate notes that cross into the cleared region.
        if t < start_ticks and t + d > start_ticks:
            d = start_ticks - t
            if d <= 0:
                continue
        # Drop note onsets inside the cleared region.
        if start_ticks <= t < end_ticks:
            continue
        out.extend([TIME_OFFSET + t, DUR_OFFSET + d, note])
    return out


def _rebase_to_zero(events: list[int]) -> list[int]:
    if not events:
        return []
    offset_ticks = ops.min_time(events, seconds=False)
    return ops.translate(events, -offset_ticks, seconds=False)


def _plan_inpaint_window(
    total_len_s: float,
    start_s: float,
    end_s: float,
    max_window_s: float,
) -> tuple[float, float]:
    if total_len_s <= max_window_s:
        return 0.0, total_len_s

    window_len = max_window_s
    center = (start_s + end_s) / 2.0
    window_start = center - window_len / 2.0
    window_end = window_start + window_len

    if window_start < 0:
        window_start = 0.0
        window_end = window_len
    if window_end > total_len_s:
        window_end = total_len_s
        window_start = window_end - window_len

    return window_start, window_end


def _build_prompt_window(events: list[int], max_s: float = 10.0) -> tuple[list[int], float, float]:
    total_len_s = ops.max_time(events, seconds=True)
    window_start_s = 0.0 if total_len_s <= max_s else total_len_s - max_s
    prompt = ops.clip(events, window_start_s, total_len_s, clip_duration=False, seconds=True)
    # Keep tick precision for time shifts (avoid rounding from seconds).
    offset_ticks = ops.min_time(prompt, seconds=False)
    prompt_rebased = ops.translate(prompt, -offset_ticks, seconds=False)
    prompt_len_s = ops.max_time(prompt_rebased, seconds=True)
    return prompt_rebased, prompt_len_s, total_len_s


def _continue_step(
    model: AutoModelForCausalLM,
    history: list[int],
    step_s: float,
    top_p: float,
    prompt_window_s: float,
) -> list[int]:
    prompt, prompt_len_s, history_end_s = _build_prompt_window(history, max_s=prompt_window_s)
    proposal = generate(
        model,
        start_time=prompt_len_s,
        end_time=prompt_len_s + step_s,
        inputs=prompt,
        top_p=top_p,
        progress=False,
    )
    continuation = ops.clip(
        proposal,
        prompt_len_s,
        prompt_len_s + step_s,
        clip_duration=False,
        seconds=True,
    )
    shift_s = history_end_s - prompt_len_s
    continuation_abs = ops.translate(continuation, shift_s, seconds=True)
    return ops.sort(history + continuation_abs)


def continue_midi(
    midi_path: str,
    continue_length_s: float,
    n_samples: int,
    output_dir: str,
    prompt_window_s: float,
) -> str:
    if not midi_path or not isinstance(midi_path, str):
        raise ValueError("midi_path must be a local file path string")
    if continue_length_s is None or continue_length_s <= 0:
        raise ValueError("continue_length_s must be > 0")
    if n_samples is None or int(n_samples) <= 0:
        raise ValueError("n_samples must be >= 1")
    if not output_dir:
        raise ValueError("output_dir is required")
    if prompt_window_s is None or float(prompt_window_s) <= 0:
        raise ValueError("prompt_window_s must be > 0")

    model = _load_model()
    top_p = _top_p()

    print(
        "[continue] params "
        f"midi_path={midi_path} continue_length_s={continue_length_s} "
        f"n_samples={n_samples} output_dir={output_dir} "
        f"prompt_window_s={prompt_window_s}"
    )

    events = midi_to_events(midi_path)
    total_len_s = ops.max_time(events, seconds=True)
    print(f"[continue] events={len(events)//3} total_len={total_len_s:.2f}s")
    tempo_us, tpb = _extract_tempo_and_tpb(midi_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(midi_path).stem

    prompt_window_s = min(float(prompt_window_s), float(MAX_TIME_IN_SECONDS))
    outputs: list[str] = []
    for sample_idx in range(int(n_samples)):
        history = events.copy()
        remaining = float(continue_length_s)
        while remaining > 0:
            step = 5.0 if remaining >= 5.0 else remaining
            history = _continue_step(model, history, step, top_p, prompt_window_s)
            remaining -= step

        out_file = output_path / f"{stem}_continue_{timestamp}_{sample_idx}.mid"
        mid = events_to_midi(history)
        mid = _apply_tempo_and_tpb(mid, tempo_us, tpb)
        mid.save(str(out_file))
        outputs.append(str(out_file))

    return "\n".join(outputs)


def inpaint_midi(
    midi_path: str,
    start_s: float,
    end_s: float,
    n_samples: int,
    output_dir: str,
    prompt_window_s: float,
) -> str:
    if not midi_path or not isinstance(midi_path, str):
        raise ValueError("midi_path must be a local file path string")
    if start_s is None or end_s is None or float(end_s) <= float(start_s):
        raise ValueError("end_s must be > start_s")
    if n_samples is None or int(n_samples) <= 0:
        raise ValueError("n_samples must be >= 1")
    if not output_dir:
        raise ValueError("output_dir is required")
    if prompt_window_s is None or float(prompt_window_s) <= 0:
        raise ValueError("prompt_window_s must be > 0")

    model = _load_model()
    top_p = _top_p()

    print(
        "[inpaint] params "
        f"midi_path={midi_path} start_s={start_s} end_s={end_s} "
        f"n_samples={n_samples} output_dir={output_dir} "
        f"prompt_window_s={prompt_window_s}"
    )

    events = midi_to_events(midi_path)
    tempo_us, tpb = _extract_tempo_and_tpb(midi_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_len_s = ops.max_time(events, seconds=True)
    print(f"[inpaint] events={len(events)//3} total_len={total_len_s:.2f}s")
    if float(end_s) > total_len_s:
        raise ValueError("end_s exceeds MIDI length")

    max_window_s = min(float(prompt_window_s), float(MAX_TIME_IN_SECONDS))
    if (float(end_s) - float(start_s)) > max_window_s:
        raise ValueError("mask length exceeds max window")

    window_start_s, window_end_s = _plan_inpaint_window(
        total_len_s=total_len_s,
        start_s=float(start_s),
        end_s=float(end_s),
        max_window_s=max_window_s,
    )
    local_start_s = float(start_s) - window_start_s
    local_end_s = float(end_s) - window_start_s

    window_events = ops.clip(
        events,
        window_start_s,
        window_end_s,
        clip_duration=False,
        seconds=True,
    )
    window_events = _rebase_to_zero(window_events)

    history = ops.clip(window_events, 0, local_start_s, clip_duration=False, seconds=True)
    future = ops.clip(
        window_events,
        local_end_s,
        window_end_s - window_start_s,
        clip_duration=False,
        seconds=True,
    )
    anticipated = [CONTROL_OFFSET + tok for tok in future]
    print(
        f"[inpaint] window_start={window_start_s:.2f}s window_end={window_end_s:.2f}s "
        f"history_events={len(history)//3} future_events={len(future)//3}"
    )

    cleared = _clear_region_events(events, float(start_s), float(end_s))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(midi_path).stem

    outputs: list[dict[str, str]] = []
    for sample_idx in range(int(n_samples)):
        inpainted = generate(
            model,
            local_start_s,
            local_end_s,
            inputs=history,
            controls=anticipated,
            top_p=top_p,
            progress=False,
        )
        completed_local = ops.sort(inpainted + future)
        inpaint_region = ops.clip(
            completed_local,
            local_start_s,
            local_end_s,
            clip_duration=False,
            seconds=True,
        )
        inpaint_region = ops.translate(inpaint_region, window_start_s, seconds=True)
        full_events = ops.sort(cleared + inpaint_region)

        full_path = output_path / f"{stem}_inpaint_full_{timestamp}_{sample_idx}.mid"
        part_path = output_path / f"{stem}_inpaint_part_{timestamp}_{sample_idx}.mid"

        full_mid = events_to_midi(full_events)
        full_mid = _apply_tempo_and_tpb(full_mid, tempo_us, tpb)
        full_mid.save(str(full_path))

        part_events = _rebase_to_zero(inpaint_region)
        part_mid = events_to_midi(part_events)
        part_mid = _apply_tempo_and_tpb(part_mid, tempo_us, tpb)
        part_mid.save(str(part_path))

        outputs.append({"full": str(full_path), "part": str(part_path)})

    return json.dumps(outputs, ensure_ascii=False)


def accompany_midi(
    melody_path: str,
    accomp_history_path: str,
    start_s: float,
    continue_length_s: float,
    n_samples: int,
    output_dir: str,
    prompt_window_s: float,
) -> str:
    if not melody_path or not isinstance(melody_path, str):
        raise ValueError("melody_path must be a local file path string")
    if start_s is None or float(start_s) < 0:
        raise ValueError("start_s must be >= 0")
    if continue_length_s is None or float(continue_length_s) <= 0:
        raise ValueError("continue_length_s must be > 0")
    if n_samples is None or int(n_samples) <= 0:
        raise ValueError("n_samples must be >= 1")
    if not output_dir:
        raise ValueError("output_dir is required")
    if prompt_window_s is None or float(prompt_window_s) <= 0:
        raise ValueError("prompt_window_s must be > 0")

    model = _load_model()
    top_p = _top_p()

    print(
        "[accompany] params "
        f"melody_path={melody_path} "
        f"accomp_history_path={accomp_history_path} "
        f"start_s={start_s} continue_length_s={continue_length_s} "
        f"n_samples={n_samples} prompt_window_s={prompt_window_s}"
    )

    melody = midi_to_events(melody_path)
    total_len_s = ops.max_time(melody, seconds=True)
    if float(start_s) >= total_len_s:
        raise ValueError("start_s exceeds melody length")
    end_s = float(start_s) + float(continue_length_s)
    if end_s > total_len_s:
        raise ValueError("continue_length_s exceeds melody length")
    print(
        f"[accompany] melody_len={total_len_s:.2f}s events={len(melody)//3} "
        f"start={float(start_s):.2f}s continue={float(continue_length_s):.2f}s end={end_s:.2f}s"
    )

    history: list[int] = []
    if accomp_history_path and isinstance(accomp_history_path, str):
        history = midi_to_events(accomp_history_path)
        history = ops.clip(history, 0, float(start_s), clip_duration=False, seconds=True)
    print(f"[accompany] history_events={len(history)//3}")

    tempo_us, tpb = _extract_tempo_and_tpb(melody_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(melody_path).stem

    max_window_s = min(float(prompt_window_s), float(MAX_TIME_IN_SECONDS))
    outputs: list[dict[str, str]] = []
    max_window_s = min(float(prompt_window_s), float(MAX_TIME_IN_SECONDS))
    if (end_s - float(start_s)) > max_window_s:
        raise ValueError("continue_length_s exceeds max window")

    window_end = end_s
    window_start = max(0.0, window_end - max_window_s)
    offset_ticks = int(round(TIME_RESOLUTION * window_start))
    local_start = float(start_s) - window_start
    local_end = end_s - window_start
    print(
        f"[accompany] window_start={window_start:.2f}s window_end={window_end:.2f}s "
        f"local_start={local_start:.2f}s local_end={local_end:.2f}s"
    )

    history_window = ops.clip(
        history,
        window_start,
        float(start_s),
        clip_duration=False,
        seconds=True,
    )
    history_local = ops.translate(history_window, -offset_ticks, seconds=False)

    melody_seg = ops.clip(melody, float(start_s), end_s, clip_duration=False, seconds=True)
    melody_local = ops.translate(melody_seg, -offset_ticks, seconds=False)
    internal_shift_s = 0.0
    if local_start < float(DELTA):
        internal_shift_s = float(DELTA) - local_start
        shift_ticks = int(round(TIME_RESOLUTION * internal_shift_s))
        history_local = ops.translate(history_local, shift_ticks, seconds=False)
        melody_local = ops.translate(melody_local, shift_ticks, seconds=False)
        local_start += internal_shift_s
        local_end += internal_shift_s
        print(
            f"[accompany] internal_shift={internal_shift_s:.2f}s "
            f"gen_local_start={local_start:.2f}s gen_local_end={local_end:.2f}s"
        )

    anticipated = [CONTROL_OFFSET + tok for tok in melody_local]
    print(
        f"[accompany] melody_seg_events={len(melody_seg)//3} "
        f"history_window_events={len(history_window)//3} "
        f"controls_tokens={len(anticipated)//3}"
    )
    if not melody_seg:
        print("[accompany] warning: melody segment is empty; controls are empty")
    if anticipated:
        ctrl_min = ops.min_time(anticipated, seconds=True)
        ctrl_max = ops.max_time(anticipated, seconds=True)
        print(f"[accompany] controls_time_range={ctrl_min:.2f}s..{ctrl_max:.2f}s")
    if history_local:
        hist_min = ops.min_time(history_local, seconds=True)
        hist_max = ops.max_time(history_local, seconds=True)
        print(f"[accompany] history_time_range={hist_min:.2f}s..{hist_max:.2f}s")

    for sample_idx in range(int(n_samples)):
        proposal = generate(
            model,
            local_start,
            local_end,
            inputs=history_local,
            controls=anticipated,
            top_p=top_p,
            progress=False,
        )
        if not proposal and not history_local:
            # Cold-start with no history often collapses to REST-only output, which is
            # removed by unpad() and appears as an empty proposal. Retry from t=0.
            print("[accompany] empty proposal; retrying with gen_start=0")
            proposal = generate(
                model,
                0.0,
                local_end,
                inputs=history_local,
                controls=anticipated,
                top_p=top_p,
                progress=False,
            )

        segment_local = ops.clip(
            proposal,
            local_start,
            local_end,
            clip_duration=False,
            seconds=True,
        )
        shift_back_s = window_start - internal_shift_s
        segment_abs = ops.translate(segment_local, shift_back_s, seconds=True)
        print(
            f"[accompany] sample={sample_idx} proposal_events={len(proposal)//3} "
            f"segment_events={len(segment_abs)//3}"
        )

        accompaniment = ops.sort(history + segment_abs)
        part_events = segment_abs

        full_path = output_path / f"{stem}_accomp_full_{timestamp}_{sample_idx}.mid"
        part_path = output_path / f"{stem}_accomp_part_{timestamp}_{sample_idx}.mid"
        merged_path = output_path / f"{stem}_accomp_merged_{timestamp}_{sample_idx}.mid"

        if not accompaniment:
            print("[accompany] empty accompaniment; writing empty MIDI")
        if not part_events:
            print("[accompany] empty part segment; writing empty MIDI")
        full_mid = _events_to_midi_or_empty(accompaniment, tempo_us, tpb)
        full_mid.save(str(full_path))

        part_mid = _events_to_midi_or_empty(_rebase_to_zero(part_events), tempo_us, tpb)
        part_mid.save(str(part_path))

        melody_for_merge = ops.clip(melody, 0, end_s, clip_duration=False, seconds=True)
        merged_events = ops.sort(accompaniment + melody_for_merge)
        merged_mid = _events_to_midi_or_empty(merged_events, tempo_us, tpb)
        merged_mid.save(str(merged_path))

        outputs.append(
            {
                "full_accompaniment": str(full_path),
                "part_accompaniment": str(part_path),
                "merged": str(merged_path),
            }
        )

    return json.dumps(outputs, ensure_ascii=False)


def _build_app() -> gr.TabbedInterface:
    continue_iface = gr.Interface(
        fn=continue_midi,
        inputs=[
            gr.Textbox(label="MIDI Path", placeholder="C:/path/to/input.mid"),
            gr.Number(label="Continue Length (s)", value=5.0),
            gr.Number(label="Samples", value=1, precision=0),
            gr.Textbox(label="Output Dir", value="outputs"),
            gr.Number(label="Prompt Window (s, max 100)", value=100.0),
        ],
        outputs=gr.Textbox(label="Output Paths (one per line)"),
        title="Anticipatory Continue",
        description="Provide a MIDI path and generate continuations using fixed 5s steps.",
    )

    inpaint_iface = gr.Interface(
        fn=inpaint_midi,
        inputs=[
            gr.Textbox(label="MIDI Path", placeholder="C:/path/to/input.mid"),
            gr.Number(label="Inpaint Start (s)", value=0.0),
            gr.Number(label="Inpaint End (s)", value=5.0),
            gr.Number(label="Samples", value=1, precision=0),
            gr.Textbox(label="Output Dir", value="outputs"),
            gr.Number(label="Prompt Window (s, max 100)", value=100.0),
        ],
        outputs=gr.Textbox(label="Output JSON"),
        title="Anticipatory Inpaint",
        description="Inpaint a region using up to 100s of surrounding context.",
    )

    accompany_iface = gr.Interface(
        fn=accompany_midi,
        inputs=[
            gr.Textbox(label="Melody Path", placeholder="C:/path/to/melody.mid"),
            gr.Textbox(label="Accomp History Path (optional)", value=""),
            gr.Number(label="Start (s)", value=0.0),
            gr.Number(label="Continue Length (s)", value=5.0),
            gr.Number(label="Samples", value=1, precision=0),
            gr.Textbox(label="Output Dir", value="outputs"),
            gr.Number(label="Prompt Window (s, max 100)", value=100.0),
        ],
        outputs=gr.Textbox(label="Output JSON"),
        title="Anticipatory Accompaniment",
        description=(
            "Generate accompaniment aligned to the melody timeline. "
            "start_s is a cursor on the melody time axis."
        ),
    )

    return gr.TabbedInterface(
        [continue_iface, inpaint_iface, accompany_iface],
        ["Continue", "Inpaint", "Accompaniment"],
        title="Anticipatory Music Transformer",
    )


if __name__ == "__main__":
    _build_app().launch(mcp_server=True, server_port=7871)
