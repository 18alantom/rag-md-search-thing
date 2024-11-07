from collections.abc import Callable
from typing import Literal

import click
import numpy as np
import numpy.typing as npt
import ollama

ModelType = Literal["encoder", "searcher"]

Embedding16 = npt.NDArray[np.float16]
Encoder = Callable[[str], Embedding16]


def check_model(model: str, model_type: ModelType = "encoder", silent: bool = False, throw: bool = False):
    non_silent = not silent
    non_silent and click.echo(f"Checking if {click.style(model_type, fg='cyan')} is up... ", nl=False)
    try:
        ollama.embeddings(model=model, prompt="testing if model is up")
    except Exception as e:
        if throw:
            raise e

        non_silent and click.echo(f"{e}", fg="red")
        return False
    non_silent and non_silent and click.echo(click.style(" \u2714", fg="green"))
    return True


def get_encoder(model: str = "nomic-embed-text") -> Encoder:
    def encoder(text: str) -> Embedding16:
        embedding = ollama.embeddings(model=model, prompt=text)["embedding"]
        return np.array(embedding, dtype=np.float16)

    return encoder
