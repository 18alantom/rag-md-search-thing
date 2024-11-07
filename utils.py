from typing import Literal

import click
import ollama

ModelType = Literal["encoder", "searcher"]


def check_model(model: str, model_type: ModelType = "encoder"):
    click.echo(
        f"Checking if {click.style(model_type, fg='cyan')} is up... ", nl=False
    )
    try:
        ollama.embeddings(model=model, prompt="testing if model is up")
    except Exception as e:
        click.echo(f"{e}", fg="red")
        return False
    click.echo(click.style(" \u2714", fg="green"))
    return True
