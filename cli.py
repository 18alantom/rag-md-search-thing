#! /usr/bin/env python3


import click

import indexing
import searching


@click.group()
def cli(): ...


@cli.command("search")
@click.option(
    "--db-path",
    "-d",
    help="Path to the index database",
)
@click.option(
    "--folder",
    "-f",
    help="Folder to search in",
)
@click.option(
    "--model-encoder",
    help="Model to use for the encoder",
    default="nomic-embed-text",
)
@click.option(
    "--model-searcher",
    help="Model to use for the searcher",
    default="llama3.2",
)
def search(db_path: str, folder: str, model_encoder: str, model_searcher: str):
    if folder:
        indexing.run(folder, model_encoder, db_path)
        print()
    searching.run(db_path, model_encoder, model_searcher, folder)


@cli.command("index")
@click.argument(
    "folder",
    type=click.Path(exists=True),
)
@click.option(
    "--model",
    "-m",
    help="Model to use for the encoder",
    default="nomic-embed-text",
)
@click.option(
    "--db-path",
    "-d",
    help="Path to the index database",
)
def index(folder: str, model: str, db_path: str):
    indexing.run(folder, model, db_path)


if __name__ == "__main__":
    cli()
