# Enterprise Support Agent MVP

Production-shaped MVP for an enterprise customer-support AI agent.

The project will grow checkpoint by checkpoint from a single Python entry point into a small agent system with an API, tools, retrieval, safety checks, evals, and temporary public deployment.

## Run locally

```bash
python app.py
```

## Run tests and evals

```bash
python -m pytest -q
python eval_runner.py
```

## Production limitations

This is not a production system. It will intentionally start with fake data, small evals, local development defaults, limited observability, no enterprise auth/RBAC, no SLA, and no load testing.
