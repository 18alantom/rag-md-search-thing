import time
from collections.abc import Callable
from pathlib import Path

import click
import ollama

import utils
from db import DB

Encoder = Callable[[str], list[float]]


def run(folder: str, extension: str, model: str, database: str):
    start = time.time()
    if not utils.check_model(model, "encoder"):
        return click.echo(click.style("Model is not up, aborting", fg="red"))

    db = DB(database)
    index_folder(folder, extension, model, db=db)
    db.commit()

    duration = time.time() - start
    return click.echo(
        "Indexing " + click.style("complete", fg="green") + click.style(f" {duration:.2f}s", dim=True)
    )


def index_folder(folder: str, extension: str, model: str, db: DB):
    click.echo(
        "Indexing " + click.style(extension, fg="cyan") + f" files in {click.style(folder, fg='cyan')}"
    )

    files = [f for f in Path(folder).glob(f"**/*.{extension}")]
    click.echo(f"Found {click.style(len(files), fg='yellow')} files:")

    encoder = _get_encoder(model)
    for i, f in enumerate(files):
        click.echo(
            click.style(f" {i + 1:3d}. ", dim=True) + click.style(f, fg="cyan"),
            nl=False,
        )
        with open(f, "r") as file:
            content = file.read()
        chunks = _get_chunks(content)
        _index_chunks(chunks, f, db, model, encoder)


def _index_chunks(chunks: list[str], file: Path, db: DB, model: str, encoder: Encoder):
    click.echo(f", encoding {click.style(len(chunks), fg='yellow')} chunks", nl=False)
    start = time.time()

    count = 0
    for chunk in chunks:
        anchor = _get_anchor(chunk)
        if db.exists(anchor, file.absolute().as_posix(), model, chunk):
            count += 1
            continue

        try:
            embedding = encoder(chunk)
            db.store(
                anchor,
                chunk,
                embedding,
                file.absolute().as_posix(),
                model,
            )
            count += 1
        except Exception:
            pass

    duration = time.time() - start
    if count == len(chunks):
        done = click.style(" \u2714 ", fg="green")
    else:
        done = click.style(" * ", fg="yellow") + click.style(f" {count}/{len(chunks)} ", dim=True)

    click.echo(done + click.style(f"{duration:.2f}s", dim=True))


def _get_anchor(chunk: str) -> str:
    header = chunk.split("\n")[0].replace("## ", "")
    if header.startswith("# "):
        header = header[2:]

    return header.lower().replace(" ", "-")


def _get_chunks(content: str) -> list[str]:
    chunks = content.split("\n## ")
    for c in range(len(chunks)):
        if chunks[c].startswith("# "):
            continue
        chunks[c] = "## " + chunks[c]
    return chunks


def _get_encoder(model: str = "nomic-embed-text") -> Encoder:
    def encoder(text: str) -> list[float]:
        return ollama.embeddings(model=model, prompt=text)["embedding"]

    return encoder
