"""Minimal custom-provider contract. Credentials must come from the environment.

Replace the function body with your model's submit/poll/download implementation.
No vendor API is implied by this example; it is deliberately not a fake endpoint.
"""
from pathlib import Path


def generate(*, task: dict, model: str, output_dir: Path,
             candidate_index: int, parameters: dict) -> Path:
    # task['prompt']: public generation prompt. Never send task['rubric'] to the generator.
    # task['references']: {} for t2v, first for kft, first+last for flt.
    # Save the downloaded video under output_dir and return its existing path.
    raise NotImplementedError('Connect your generation provider here; see docs/GENERATION.md')
