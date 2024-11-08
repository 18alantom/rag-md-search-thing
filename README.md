## What is this?

An experiment in seeing how quickly an RAG based document search thing can be
made. It obviously doesn't work very well, everything runs zero-shot (no fine
tuning) and I haven't really read up on RAG. But it allows me to see how a
system like this can be really powerful when it comes to answering search
queries.


## How to run?

1. Setup [ollama](https://github.com/ollama/ollama/tree/main)
2. Pull the following models: `llama3.2`, `nomic-embed-text` (you can use other models too, these are defaults)
3. Install the dependencies in `requirements.txt`
4. Run `./cli.py search --folder ./path/to/folder`

To see the cli docs:

```bash
./cli.py --help
./cli.py search --help
./cli.py index --help # if you pass --folder, index is auto called on search
```

Both models run locally and with decent speed on my laptop (M1 Pro, 32GB RAM).