from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import gradio as gr
import mido
import torch
from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.config import MAX_TIME_IN_SECONDS
from anticipation.convert import events_to_midi, midi_to_events
from anticipation.sample import generate


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

    events = midi_to_events(midi_path)
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


def _build_app() -> gr.Interface:
    return gr.Interface(
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


if __name__ == "__main__":
    _build_app().launch(mcp_server=True, server_port=7871)
