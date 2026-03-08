# MarketFoundry API

> **Stack:** Modal (serverless GPU)  
> **Cost:** $0 to start  
> **Live endpoint:** `https://marija-vukic--market-foundry-api-fastapi-app.modal.run`

## How it works

MarketFoundry provides the **compute and intelligence layer** — document classification, sectioning, and causal triple extraction via OneKE + Qwen.

**You bring your own Neo4j database.** Pass your credentials with each request and extracted triples are written directly to your own graph instance. No credentials? No problem — results are still returned as JSON.

```
Your Document
     ↓
MarketFoundry API (Modal GPU)
  → classify → section → extract triples
     ↓                        ↓
  JSON response         Your Neo4j instance
                        (optional, yours only)
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/process` | Upload document + optional Neo4j credentials → extract knowledge graph |
| `POST` | `/query` | Run Cypher against your own Neo4j instance |
| `GET`  | `/health` | Liveness check |
| `GET`  | `/docs` | Swagger UI |

---

## Usage

### Without Neo4j (JSON output only)

```bash
curl -X POST "https://marija-vukic--market-foundry-api-fastapi-app.modal.run/process" \
  -F "file=@earnings_call.pdf"
```

### With your own Neo4j (triples persisted to your graph)

```bash
curl -X POST "https://marija-vukic--market-foundry-api-fastapi-app.modal.run/process" \
  -F "file=@earnings_call.pdf" \
  -F "neo4j_uri=neo4j+s://xxxx.databases.neo4j.io" \
  -F "neo4j_username=neo4j" \
  -F "neo4j_password=yourpassword"
```

Response:
```json
{
  "filename": "earnings_call.pdf",
  "document_type": "Earnings Call Transcript",
  "sections_identified": ["financials", "mdna", "outlook"],
  "neo4j_relationships_written": 24,
  "neo4j_used": true,
  "status": "success"
}
```

### Query your graph

```bash
curl -X POST "https://marija-vukic--market-foundry-api-fastapi-app.modal.run/query" \
  -F "cypher=MATCH (s)-[r]->(o) RETURN s.name, type(r), o.name LIMIT 25" \
  -F "neo4j_uri=neo4j+s://xxxx.databases.neo4j.io" \
  -F "neo4j_username=neo4j" \
  -F "neo4j_password=yourpassword"
```

### Interactive docs
`https://marija-vukic--market-foundry-api-fastapi-app.modal.run/docs`

---

## Get a free Neo4j instance

1. Go to [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/)
2. Sign up and click **Create Free Instance**
3. Save the URI, username, and password shown — password is only displayed once

Free tier: 200k nodes, 400k relationships, $0/month.

---

## Deploy your own instance

```bash
pip install modal
modal setup
modal deploy src/api/modal_app.py
```

No secrets needed — just Modal login. Users supply their own Neo4j credentials per request.
