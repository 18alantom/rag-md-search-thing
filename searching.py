from collections.abc import Callable, Iterator
from pathlib import Path
from textwrap import indent

import click
import numpy as np
import ollama

from db import DB
from utils import Embedding16, Encoder, check_model, get_encoder

Searcher = Callable[[str, list[str]], Iterator[dict[str, dict]]]


class Search:
    db: DB
    model_encoder: str
    encoder: Encoder
    model_searcher: str
    searcher: Searcher
    index: list[dict]

    def __init__(self, db_path: str, model_encoder: str, model_searcher: str):
        self.db = DB(db_path)

        self.model_encoder = model_encoder
        check_model(self.model_encoder, "encoder", silent=True, throw=True)
        self.encoder = get_encoder(self.model_encoder)

        self.model_searcher = model_searcher
        check_model(self.model_searcher, "searcher", silent=True, throw=True)
        self.searcher = get_searcher(self.model_searcher)

        self.index = self.db.all(self.model_encoder)

    def run(self):
        click.echo("Starting search, enter " + click.style("q", fg="yellow") + " to quit.")
        while True:
            r = click.prompt(
                text="",
                prompt_suffix=click.style(">", fg="magenta", bold=True),
            )

            if r == "q":
                break

            embedding = self.encoder(r)
            sims = [(ind, cosine_similarity(embedding, ind["embedding"])) for ind in self.index]
            sims = [(ind, sim) for ind, sim in sims if sim > 0.55][:5]

            if not sims:
                click.secho("  No results found.", fg="yellow", dim=True)
                print()
                continue

            sims.sort(key=lambda x: x[1], reverse=True)
            click.secho("\u25cb ", fg="green", bold=True, nl=False)
            for res in self.searcher(r, [ind["chunk"] for ind, _ in sims]):
                click.secho(res["message"]["content"], fg="bright_white", nl=False)

            click.secho("\n\nReferences:", bold=True)
            for i, (ind, sim) in enumerate(sims):
                click.secho(f"{i+1}. ", fg="yellow", nl=False)
                file = Path(ind["file"]).relative_to(
                    Path(".").absolute()
                )  # will throw if not called from repo root
                click.secho(f"{file}#{ind['anchor']} ", fg="cyan", underline=True, nl=False)
                click.echo(click.style(" · ", dim=True) + click.style(f"{sim:.4f}", fg="blue", dim=True))
                # click.secho(format_chunk(ind["chunk"] + "..."), dim=True)
            print()

    def _answer(self) -> str: ...


def run(db_path: str, model_encoder: str, model_searcher: str):
    Search(db_path, model_encoder, model_searcher).run()


def cosine_similarity(a: Embedding16, b: Embedding16) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def format_chunk(chunk: str) -> str:
    c = "\n".join(line.strip() for line in chunk.splitlines() if line.strip() and not line.startswith("#"))
    return indent(c[:96] + "...", prefix="     ")


def get_searcher(model_searcher: str) -> Searcher:
    # Prompt
    template = """Use the following context to answer the question at the end.
    If you don't know the answer, say that you don't know, don't make up an answer.
    Try to be concise and break the answer into multiple steps.
    {context}
    Question: {question}
    Helpful Answer:"""

    def searcher(question: str, context: list[str]):
        return ollama.chat(
            model=model_searcher,
            messages=[
                {"role": "user", "content": template.format(context="\n".join(context), question=question)}
            ],
            stream=True,
        )

    return searcher
