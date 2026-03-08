# MarketFoundry API

> **Stack:** Modal (serverless GPU)  
> **Cost:** $0 to start  
> **Live endpoint:** `https://marija-vukic--market-foundry-api-fastapi-app.modal.run`  
> **Docs:** `https://marija-vukic--market-foundry-api-fastapi-app.modal.run/docs`  
> **GitHub:** https://github.com/jessicabat/market-foundry  
> **Website:** https://jessicabat.github.io/market-foundry/

---

## How it works

MarketFoundry is an **open-source, API-first knowledge extraction engine** that converts any financial document format into queryable knowledge graphs.

It provides the **compute and intelligence layer** — document classification, sectioning, and causal triple extraction via OneKE + Qwen.

**Build Your Own Database.** Pass your Neo4j AuraDB credentials with each request and extracted triples are written directly to your own graph instance. No credentials? No problem — results are still returned as JSON.

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
| `GET`  | `/result/{job_id}` | Poll for results from a submitted job |
| `POST` | `/query` | Run a read-only Cypher query against your own Neo4j instance |
| `GET`  | `/health` | Liveness check |
| `GET`  | `/docs` | Swagger UI |

---

## Usage

### 1. Submit a job

**Without Neo4j (JSON output only):**

```bash
curl -X POST "https://marija-vukic--market-foundry-api-fastapi-app.modal.run/process" \
  -F "file=@/path/to/your/document.pdf"
```

**With your own Neo4j (triples also written to your graph):**

```bash
curl -X POST "https://marija-vukic--market-foundry-api-fastapi-app.modal.run/process" \
  -F "file=@/path/to/your/document.pdf" \
  -F "neo4j_uri=neo4j+s://xxxx.databases.neo4j.io" \
  -F "neo4j_username=neo4j" \
  -F "neo4j_password=yourpassword"
```

Response:
```json
{
  "job_id": "fc-JobID",
  "status": "processing",
  "poll_url": "/result/fc-JobID",
  "message": "Job started for 'document.pdf'. Poll /result/fc-JobID to get results."
}
```

---

### 2. Poll for results

```bash
curl "https://marija-vukic--market-foundry-api-fastapi-app.modal.run/result/fc-JobID"
```

While running:
```json
{"status": "processing", "message": "Still running, check back soon."}
```

When complete:
```json
{
  "status": "complete",
  "result": {
    "filename": "document.pdf",
    "document_type": "Earnings Call Transcript",
    "triples": [
      {
        "head": "Apple Inc.",
        "head_type": "company",
        "relation": "reported",
        "relation_type": "financial_result",
        "tail": "record quarterly revenue of $124.3 billion",
        "tail_type": "financial_metric"
      }
    ],
    "triple_count": 1,
    "neo4j_relationships_written": 1,
    "neo4j_used": true,
    "status": "success"
  }
}
```

---

### 3. Query your graph (optional)

Once triples are written to Neo4j, you can query your graph directly via the API:

```bash
curl -X POST "https://marija-vukic--market-foundry-api-fastapi-app.modal.run/query" \
  -F "cypher=MATCH (s)-[r]->(o) RETURN s.name, type(r), o.name LIMIT 25" \
  -F "neo4j_uri=neo4j+s://xxxx.databases.neo4j.io" \
  -F "neo4j_username=neo4j" \
  -F "neo4j_password=yourpassword"
```

Response:
```json
{
  "results": [
    {"s.name": "Apple Inc.", "type(r)": "REPORTED", "o.name": "record quarterly revenue"}
  ],
  "count": 1
}
```

> Only `MATCH` and `CALL` queries are permitted — the endpoint is read-only.

---

## Supported file formats

`.pdf`, `.docx`, `.txt`, `.html`, `.json`

---

## Get a free Neo4j instance

1. Go to [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/)
2. Sign up and click **Create Free Instance**
3. Save the URI, username, and password shown — **password is only displayed once**

Free tier: 200k nodes, 400k relationships, $0/month.

---

## Deploy your own instance

```bash
pip install modal
modal setup
modal deploy src/api/modal_app.py
```

No secrets needed — just Modal login. Users supply their own Neo4j credentials per request.
