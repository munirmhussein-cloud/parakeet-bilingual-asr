"""Parakeet inference entry points."""

from pathlib import Path


def transcribe_file(audio_path: str | Path, *, model_name: str) -> str:
    """Transcribe a single audio file with a Parakeet-compatible model.

    This placeholder keeps the package structure stable until model loading and
    decoding are implemented in a later sprint.
    """

    raise NotImplementedError(
        f"Parakeet inference is not implemented yet for {Path(audio_path)} using {model_name}."
    )
