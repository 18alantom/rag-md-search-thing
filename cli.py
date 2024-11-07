#! /usr/bin/env python3


import click

import indexing


@click.group()
def cli(): ...


@cli.command("search")
def search():
    click.echo("Searching...")


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
    "--database",
    "-d",
    help="Path to the index database",
)
@click.option(
    "--extension",
    "-e",
    help='Extensions to index (eg "md,txt")',
    default="md",
)
def index(folder: str, extension: str, model: str, database: str):
    indexing.run(folder, extension, model, database)


@cli.command("delete")
def delete():
    click.echo("Deleting...")


if __name__ == "__main__":
    cli()
