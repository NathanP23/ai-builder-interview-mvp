# Enterprise Support Agent MVP

Production-shaped MVP for an enterprise customer-support AI agent.

The project will grow checkpoint by checkpoint from a single Python entry point into a small agent system with an API, tools, retrieval, safety checks, evals, and temporary public deployment.

## Run locally

```bash
PYTHONDONTWRITEBYTECODE=1 python app.py
```

## Run API locally

```bash
PYTHONDONTWRITEBYTECODE=1 python -m uvicorn api.server:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Customer 1842 says order O-991 was charged twice. Check the refund policy and prepare a refund if allowed."}'
```

## Run tests and evals

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python -m evals.eval_runner
```

## Production limitations

This is not a production system. It will intentionally start with fake data, small evals, local development defaults, limited observability, no enterprise auth/RBAC, no SLA, and no load testing.
