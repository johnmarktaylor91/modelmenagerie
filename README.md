# Model Menagerie

The code behind [Model Menagerie](https://modelmenagerie.ai), a browsable catalog of deep
neural network architectures traced with [TorchLens](https://github.com/johnmarktaylor91/torchlens).

Menagerie keeps a roster of published models, works out how to run each one (its environment,
weights and inputs), runs it through TorchLens, checks that the trace is valid, and turns the
result into the graphs, data and metadata the website shows.

This repository is a snapshot of the project's code, refreshed from the working repository.
It does not include the project's data: the model roster, per-model records, research notes,
run ledgers and the site's catalog files stay with the working repository. Tests that need
that data are skipped here (`public-skips.txt`).

## What is here

| Path | What it is |
| --- | --- |
| `menagerie/` | The catalog pipeline: the model catalog, environments per model, graph rendering, validation and its ledger. |
| `menagerie/pipeline/` | The current pipeline: research how to run a model, write its recipe, run a forward pass and a TorchLens trace, validate and publish. |
| `menagerie/crawler/` | The earlier crawler (July to September 2026) that built the roster of found models. |
| `menagerie/classics/` | Hand-built reconstructions of historical and hard-to-install models, some ported from other open-source projects (see `NOTICE`). |
| `envs/` | Environment definitions for the frameworks Menagerie runs models under (PyTorch, TensorFlow, JAX, Paddle, MLX and others). |
| `site/` | The Astro website. It builds only with the catalog files in `site/site-data/`, which are not included. |
| `tests/` | The test suite. |
| `docs/` | The validation method, metadata schemas and the TorchLens interfaces Menagerie relies on. |

## Getting started

```bash
pip install -e ".[dev,test]"
pytest
```

The crawler's own suite under `menagerie/crawler/tests` runs only when named, and parts of it read
recorded runs that are not included here. Bulk outputs go
to `~/scratch/menagerie/` by default; set `MENAGERIE_WORK_ROOT` to move them.

## Licence

Apache License 2.0 (`LICENSE`). Third-party code in `menagerie/classics/` keeps its own licence;
see `NOTICE` and `menagerie/classics/LICENSES/`. The website's text and dataset are published
under CC BY 4.0 on the site itself.

## Contributing

Issues and suggestions are welcome. This repository is generated from a private working
repository, so pull requests are applied there by hand rather than merged here.
