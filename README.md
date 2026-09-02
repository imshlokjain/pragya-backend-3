# PRAGYA — Backend (Phase 1 scaffold)

Foundation for the PRAGYA backend: DB schema, FastAPI skeleton, and mock
`/risk`, `/rainfall`, `/river` endpoints so the frontend and other teammates
have something real to build against while the ML and RAG pieces are built.

## What's here

- `backend/database/models.py` — SQLAlchemy models for all 9 core tables
  (zones, rainfall/river/satellite observations, flood_detections,
  risk_predictions, sop_documents, sop_chunks, audit_events).
- `backend/services/mock_data.py` — deterministic mock risk/rainfall/river
  data, clearly tagged `is_prototype` / `SIMULATED` so nothing looks like a
  live government observation. **This is the file to swap out** once the ML
  teammate's model and the real ingestion pipelines exist — everything else
  (routers, schemas) stays the same.
- `backend/api/` — REST routers: `/api/v1/risk/{zone_id}`,
  `/api/v1/rainfall/{zone_id}`, `/api/v1/river/{zone_id}`.
- `docker-compose.yml` — Postgres/PostGIS + Redis + backend, wired together.

## Running it

### Option A — Full stack with Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

This starts Postgres/PostGIS + Redis, runs `alembic upgrade head` automatically,
then starts the API. Visit `http://localhost:8000/docs`.

### Option B — Local Postgres, no Docker

```bash
# 1. Install Postgres 16 + PostGIS locally, then:
psql -c "CREATE USER pragya WITH PASSWORD 'pragya' SUPERUSER;"
psql -c "CREATE DATABASE pragya OWNER pragya;"
psql -d pragya -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# 2. Install deps and apply migrations
pip install -r requirements.txt
cp .env.example .env   # already points at localhost:5432
alembic upgrade head

# 3. Seed the pilot district's 5 zones (real PostGIS geometry, not mock data)
python -m backend.scripts.seed_zones

# 4. Run the API
uvicorn backend.main:app --reload
```

Then visit `http://localhost:8000/docs`, or:

```bash
curl http://localhost:8000/api/v1/risk/ZONE_03
```

### Database migrations

Schema lives in `backend/database/models.py`; migrations live in
`backend/alembic/versions/`. The initial migration
(`5106a3cfd9db_initial_schema...py`) creates all 9 core tables plus PostGIS
GIST spatial indexes on every geometry column, and has been tested against a
real Postgres/PostGIS instance.

To make a schema change: edit `models.py`, then:

```bash
alembic revision --autogenerate -m "describe your change"
alembic upgrade head
```

One gotcha already fixed for you: GeoAlchemy2's `Geometry` type creates its
own GIST index automatically when a table is created, so `env.py` is
configured (via `render_item`) to avoid autogenerate emitting a duplicate
`create_index` for geometry columns. If you ever see a `DuplicateTable` error
on a geometry index after autogenerating, delete the redundant
`op.create_index(...)` line for that column from the generated migration.

## What's real vs. still mocked

- **Real (from Postgres):** zone identity, names, geometry, population,
  vulnerability_index — via `backend/services/zone_service.py`, seeded by
  `backend/scripts/seed_zones.py`. `/api/v1/zones` reads live DB rows.
- **Still mocked:** risk scores, rainfall, river levels — `mock_data.py`
  generates these deterministically per zone, clearly tagged
  `is_prototype: true` / `SIMULATED`. Swapped for real values once
  ingestion (Phase 2) and the trained model (Phase 3) exist.

## Endpoints so far

- `GET /api/v1/zones` — real zones from the database
- `GET /api/v1/risk/{zone_id}` — mock risk prediction with SHAP-style drivers
- `GET /api/v1/rainfall/{zone_id}` / `GET /api/v1/river/{zone_id}` — mock observations
- `POST /api/v1/scenario` — what-if engine (TRD §22): re-runs the risk
  function against a modified rainfall multiplier (1.0–1.5x per MVP.md) and
  optional river-level increase, returns baseline vs. scenario comparison,
  always tagged `is_hypothetical: true`

## Contracts for teammates

- **ML teammate**: your model's output needs to match `RiskPredictionOut` in
  `backend/schemas/risk.py` — `risk_score`, `risk_category`, `confidence`,
  and a `drivers` list of `{feature, value, contribution, direction}` (SHAP
  output). Once you export a model artifact, `predict_risk()` in
  `mock_data.py` gets replaced with a real call to it.
- **RAG teammate**: I'll need `search_sop(query, zone_context) -> [{clause,
  document, section, page, score}]` — not built yet, comes in Phase 4.
- **Frontend teammate**: everything under `/api/v1/*` is yours to consume.
  Swagger docs at `/docs` once the server is running.

## Not built yet (next phases)

- Real ingestion pipelines (currently mock data only)
- Alembic migrations (models exist, migrations don't yet)
- Agent orchestrator / tool calling
- What-if scenario endpoint
- SOP RAG + response plan generation
- Auth/RBAC, audit logging middleware
