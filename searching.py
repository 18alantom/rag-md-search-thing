from pathlib import Path
from textwrap import indent

import click
import numpy as np

from db import DB
from utils import Embedding16, Encoder, check_model, get_encoder


class Search:
    db: DB
    model_encoder: str
    encoder: Encoder
    index: list[dict]

    def __init__(self, db_path: str, model_encoder: str):
        self.db = DB(db_path)
        self.model_encoder = model_encoder
        check_model(self.model_encoder, "encoder", silent=True, throw=True)
        self.encoder = get_encoder(self.model_encoder)
        self.index = self.db.all(self.model_encoder)

    def run(self):
        click.echo("Starting search, enter " + click.style("q", fg="yellow") + " to quit.")
        while True:
            r = click.prompt(
                text="",
                prompt_suffix=click.style(">", fg="magenta"),
            )

            if r == "q":
                break

            embedding = self.encoder(r)
            sims = [(ind, cosine_similarity(embedding, ind["embedding"])) for ind in self.index]
            sims = [(ind, sim) for ind, sim in sims if sim > 0.6]
            if not sims:
                click.secho("  No results found.", fg="yellow", dim=True)
                print()
                continue

            sims.sort(key=lambda x: x[1], reverse=True)
            for i, (ind, sim) in enumerate(sims[:5]):
                click.secho(f"  {i+1}. ", nl=False)
                file = Path(ind["file"])
                click.secho(f"{file}#{ind['anchor']} ", fg="cyan", underline=True, nl=False)
                click.echo(click.style(" · ", dim=True) + click.style(f"{sim:.4f}", fg="blue", dim=True))
                click.secho(format_chunk(ind["chunk"] + "..."), dim=True)
                print()


def run(db_path: str, model_encoder: str):
    Search(db_path, model_encoder).run()


def cosine_similarity(a: Embedding16, b: Embedding16) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def format_chunk(chunk: str) -> str:
    c = "\n".join(l.strip() for l in chunk.splitlines() if l.strip() and not l.startswith("#"))
    return indent(c[:96] + "...", prefix="     ")
