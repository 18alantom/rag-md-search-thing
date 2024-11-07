#! /usr/bin/env python3


import click

import indexing
import searching


@click.group()
def cli(): ...


@cli.command("search")
@click.option(
    "--model-encoder",
    help="Model to use for the encoder",
    default="nomic-embed-text",
)
@click.option(
    "--db-path",
    "-d",
    help="Path to the index database",
)
def search(model_encoder: str, db_path: str):
    searching.run(db_path, model_encoder)


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
@click.option(
    "--extension",
    "-e",
    help='Extensions to index (eg "md,txt")',
    default="md",
)
def index(folder: str, extension: str, model: str, db_path: str):
    indexing.run(folder, extension, model, db_path)


@cli.command("delete")
def delete():
    click.echo("Deleting...")


if __name__ == "__main__":
    cli()
