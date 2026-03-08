"""
Market Foundry — Modal API
==========================
Deploys the MarketFoundry pipeline as a public serverless API on Modal.

BYOD (Bring Your Own Database) pattern:
  Users pass their own Neo4j credentials per request.
  MarketFoundry provides the compute + intelligence layer only.

Deploy:
    cd market-foundry
    modal deploy src/api/modal_app.py

Test (without Neo4j):
    curl -X POST https://<your-modal-url>/process \
      -F "file=@earnings_call.pdf"

Test (with your own Neo4j):
    curl -X POST https://<your-modal-url>/process \
      -F "file=@earnings_call.pdf" \
      -F "neo4j_uri=neo4j+s://xxxx.databases.neo4j.io" \
      -F "neo4j_username=neo4j" \
      -F "neo4j_password=yourpassword"
"""

import modal
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent.parent  # .../market-foundry/

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "poppler-utils", "tesseract-ocr", "libmagic1")
    .pip_install(
        "accelerate==1.1.1",
        "beautifulsoup4==4.14.3",
        "datasets==4.5.0",
        "fastapi==0.129.2",
        "huggingface-hub==0.36.2",
        "jinja2==3.1.6",
        "joblib==1.5.2",
        "langchain==0.3.3",
        "langchain-community==0.3.2",
        "langchain-core==0.3.10",
        "langchain-openai==0.2.0",
        "langchain-text-splitters==0.3.0",
        "lxml==6.0.2",
        "neo4j==5.28.1",
        "nltk==3.9.1",
        "numpy==1.26.4",
        "openai==1.55.3",
        "pandas==2.2.3",
        "pdfminer-six==20251230",
        "pdfplumber==0.11.9",
        "protobuf==6.33.5",
        "pypdf==4.3.1",
        "python-docx",
        "docx2txt",
        "python-dotenv==1.2.1",
        "python-multipart==0.0.22",
        "pyyaml==6.0.2",
        "safetensors==0.7.0",
        "scikit-learn==1.6.1",
        "sentence-transformers==3.3.0",
        "sentencepiece==0.2.0",
        "torch==2.4.0",
        "transformers==4.44.0",
        "tokenizers==0.19.1",
        "uvicorn==0.41.0",
        "requests==2.32.5",
        "tenacity==8.5.0",
        "tiktoken==0.12.0",
        "tqdm==4.67.3",
        "typing-extensions==4.15.0",
        "sqlalchemy==2.0.46",
        "rapidfuzz==3.10.1",
        "pydantic==2.12.5",
        "gradio==4.44.0",
        "annotated-types==0.7.0",
    )
    .run_commands(
        "git clone https://github.com/OpenSPG/OneKE /opt/OneKE || true",
        "pip install -e /opt/OneKE 2>/dev/null || true",
        "python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')\"",
    )
    .add_local_dir(_repo_root, remote_path="/root/market-foundry", copy=True)
)

# ---------------------------------------------------------------------------
# App — no secrets needed since users provide their own Neo4j credentials
# ---------------------------------------------------------------------------
app = modal.App("market-foundry-api", image=image)

model_cache = modal.Volume.from_name("market-foundry-model-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A10G",
    volumes={"/model-cache": model_cache},
    timeout=1800,
    scaledown_window=300,
)
class MarketFoundryPipeline:

    @modal.enter()
    def setup(self):
        """
        Runs once per container start.
        Loads classifier + vectorizer only.
        Neo4j is connected per-request using user-supplied credentials.
        """
        repo_root = "/root/market-foundry"
        src_path = f"{repo_root}/src"
        oneke_path = f"{repo_root}/OneKE"

        for p in [src_path, f"{src_path}/utils", oneke_path, f"{oneke_path}/src", repo_root]:
            if p not in sys.path:
                sys.path.insert(0, p)

        os.environ["REPO_ROOT"] = repo_root

        import importlib.util

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        load_module("utils", f"{src_path}/utils/__init__.py")
        load_module("utils.document_classification", f"{src_path}/utils/document_classification.py")
        load_module("utils.document_sectioning", f"{src_path}/utils/document_sectioning.py")
        load_module("utils.document_extraction", f"{src_path}/utils/document_extraction.py")

        from utils.document_classification import load_tfidf_model, load_tfidf_vectorizer
        self.model = load_tfidf_model(f"{src_path}/models/Document_Classifier.joblib")
        self.vectorizer = load_tfidf_vectorizer(f"{src_path}/models/TFIDF_Vectorizer.joblib")
        self.src_path = src_path
        self.repo_root = repo_root
        print("Classifier loaded ✓")

    @modal.method()
    def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        neo4j_uri: str = "",
        neo4j_username: str = "",
        neo4j_password: str = "",
        webhook_url: str = "",
    ) -> dict:
        """
        Full MarketFoundry pipeline.
        If neo4j_* credentials are provided, triples are written to the
        user's own Neo4j instance. Otherwise, results are returned as JSON only.
        """
        import tempfile, yaml, importlib, json as _json, re
        from utils.document_classification import (
            load_file, extract_text, clean_texts,
            classify_document_types, output_classifications,
        )
        from utils.document_sectioning import section_documents

        # Write extraction config with literal values (no ${} substitution)
        # MUST happen before importing document_extraction — it reads the config at import time
        config_path = f"{self.repo_root}/src/utils/extraction_config.yaml"
        config_lines = [
            "model:\n",
            "  category: Qwen\n",
            "  model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct\n",
            '  api_key: ""\n',
            '  base_url: ""\n',
            "\n",
            "extraction:\n",
            "  update_case: false\n",
            "  show_trajectory: false\n",
        ]
        if neo4j_uri and neo4j_username and neo4j_password:
            config_lines += [
                "\n",
                "construct:\n",
                "  database: Neo4j\n",
                f'  url: "{neo4j_uri}"\n',
                f'  username: "{neo4j_username}"\n',
                f'  password: "{neo4j_password}"\n',
            ]
            # Also set as env vars in case OneKE resolves ${} at read time
            os.environ["NEO4J_URL"] = neo4j_uri
            os.environ["NEO4J_USERNAME"] = neo4j_username
            os.environ["NEO4J_PASSWORD"] = neo4j_password
        with open(config_path, "w") as f:
            f.writelines(config_lines)

        # Import AFTER patching the config file so reference_config picks up Neo4j credentials
        import utils.document_extraction as _doc_ext
        importlib.reload(_doc_ext)
        extract_topics_and_run_oneke_modal = _doc_ext.extract_topics_and_run_oneke_modal

        # Write file bytes to temp path (loaders need a file path)
        suffix = os.path.splitext(filename)[1] or ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            loaded = load_file(tmp_path)
            if not loaded:
                return {"error": f"Could not load file: {filename}"}

            texts = extract_text([loaded])
            text_lookup = {file: text for file, text in texts}
            cleaned_texts = clean_texts(texts)

            classifications = classify_document_types(self.model, self.vectorizer, cleaned_texts)
            output_classifications(classifications)

            sectioned_documents = section_documents(texts)

            # extract_topics_and_run_oneke_modal now returns accumulated triples from all YAML runs
            all_triples = extract_topics_and_run_oneke_modal(cleaned_texts, classifications, text_lookup) or []
            print(f"Total triples collected across all YAML runs: {len(all_triples)}")

            # Filter out malformed entries — model sometimes outputs plain strings
            # like "description", "items", "title", "type" inside triple_list
            all_triples = [t for t in all_triples if isinstance(t, dict) and t.get("head") and t.get("tail")]
            print(f"Clean triple count after filtering: {len(all_triples)}")

            # Normalize each triple to a consistent schema:
            # { head, head_type, relation, relation_type, tail, tail_type }
            all_triples = [
                {
                    "head": t.get("head", "").strip(),
                    "head_type": t.get("head_type", "").strip().lower(),
                    "relation": (t.get("relation") or "").strip(),
                    "relation_type": (t.get("relation_type") or "").strip(),
                    "tail": t.get("tail", "").strip(),
                    "tail_type": t.get("tail_type", "").strip().lower(),
                }
                for t in all_triples
            ]

            # Push triples to user's Neo4j
            neo4j_count = 0
            if neo4j_uri and neo4j_username and neo4j_password:
                print(f"Connecting to Neo4j at {neo4j_uri} as {neo4j_username}...")
                try:
                    from neo4j import GraphDatabase
                    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
                    with driver.session() as session:
                        for triple in all_triples:
                            rel = triple.get("relation_type") or triple.get("relation") or "RELATED_TO"
                            rel_label = re.sub(r"[^a-zA-Z0-9_]", "_", rel).upper() or "RELATED_TO"
                            raw_head_type = triple.get("head_type") or "entity"
                            raw_tail_type = triple.get("tail_type") or "entity"
                            head_label = re.sub(r"[^a-zA-Z0-9_]", "_", raw_head_type).strip("_") or "entity"
                            tail_label = re.sub(r"[^a-zA-Z0-9_]", "_", raw_tail_type).strip("_") or "entity"
                            cypher = (
                                f"MERGE (h:{head_label} {{name: $head}}) "
                                f"MERGE (t:{tail_label} {{name: $tail}}) "
                                f"MERGE (h)-[r:{rel_label}]->(t) "
                                f"SET r.relation = $relation"
                            )
                            session.run(
                                cypher,
                                head=triple.get("head", ""),
                                tail=triple.get("tail", ""),
                                relation=triple.get("relation", ""),
                            )
                        result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
                        record = result.single()
                        neo4j_count = record["cnt"] if record else 0
                    driver.close()
                    print(f"Neo4j write complete: {neo4j_count} relationships in graph")
                except Exception as e:
                    print(f"Neo4j write error: {e}")

            result = {
                "filename": filename,
                "document_type": list(classifications.values())[0] if classifications else "unknown",
                "sections_identified": list(list(sectioned_documents.values())[0].keys()) if sectioned_documents else [],
                "triples": all_triples,
                "triple_count": len(all_triples),
                "neo4j_relationships_written": neo4j_count,
                "neo4j_used": bool(neo4j_uri),
                "status": "success",
            }

            # Fire webhook if provided — POST full results to user's endpoint
            if webhook_url:
                try:
                    import requests as _requests
                    resp = _requests.post(webhook_url, json=result, timeout=30)
                    print(f"Webhook delivered to {webhook_url} — status {resp.status_code}")
                except Exception as e:
                    print(f"Webhook delivery failed: {e}")

            return result

        finally:
            os.unlink(tmp_path)

    @modal.method()
    def query_graph(
        self,
        cypher: str,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
    ) -> list[dict]:
        """Run a Cypher query against the user's own Neo4j instance."""
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        try:
            with driver.session() as session:
                result = session.run(cypher)
                return [dict(record) for record in result]
        finally:
            driver.close()

    @modal.method()
    def health(self) -> dict:
        return {"status": "ok", "classifier_loaded": self.model is not None}


# ---------------------------------------------------------------------------
# FastAPI web app
# ---------------------------------------------------------------------------
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional

# --- Response models ---
class ProcessResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str
    message: str
    webhook_url: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "fc-01ABC123",
                "status": "processing",
                "poll_url": "/result/fc-01ABC123",
                "message": "Job started for 'earnings.pdf'. Poll /result/fc-01ABC123 to get results."
            }
        }
    }

class Triple(BaseModel):
    head: str
    head_type: str
    relation: str
    relation_type: str
    tail: str
    tail_type: str

class ExtractionResult(BaseModel):
    filename: str
    document_type: str
    sections_identified: List[str]
    triples: List[Triple]
    triple_count: int
    neo4j_relationships_written: int
    neo4j_used: bool
    status: str

class ResultResponse(BaseModel):
    status: str
    result: Optional[ExtractionResult] = None
    message: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "complete",
                "result": {
                    "filename": "earnings.pdf",
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
                    "neo4j_used": True,
                    "status": "success"
                }
            }
        }
    }

class QueryResponse(BaseModel):
    results: List[Any]
    count: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {"s.name": "Apple Inc.", "type(r)": "REPORTED", "o.name": "record quarterly revenue"}
                ],
                "count": 1
            }
        }
    }

web_app = FastAPI(
    title="MarketFoundry API",
    description=(
        "### An open-source, API-first knowledge extraction engine that converts any financial document format into queryable knowledge graphs.\n\n"
        "---\n\n"
        "**Build Your Own Database:** Optionally pass your Neo4j AuraDB credentials and extracted triples will be written directly to your own graph instance.\n\n"
        "Get a free Neo4j instance at [https://neo4j.com/cloud/aura/](https://neo4j.com/cloud/aura/)\n\n"
        "---\n\n"
        "## Learn More\n\n"
        "- **GitHub:** [https://github.com/jessicabat/market-foundry](https://github.com/jessicabat/market-foundry)\n"
        "- **Website:** [https://jessicabat.github.io/market-foundry/](https://jessicabat.github.io/market-foundry/)\n\n"
        "---\n\n"
        "## Quick Start\n\n"
        "### 1. Submit a job with Neo4j Credentials\n\n"
        "```bash\n"
        "curl -X POST \"https://marija-vukic--market-foundry-api-fastapi-app.modal.run/process\" \\\n"
        "  -F \"file=@/path/to/your/document.pdf\" \\\n"
        "  -F \"neo4j_uri=neo4j+s://xxxx.databases.neo4j.io\" \\\n"
        "  -F \"neo4j_username=neo4j\" \\\n"
        "  -F \"neo4j_password=yourpassword\"\n"
        "```\n\n"
        "**Response:**\n"
        "```json\n"
        "{\"job_id\": \"fc-JobID\", \"status\": \"processing\", \"poll_url\": \"/result/fc-JobID\", \"message\": \"Job started for 'file.pdf'. Poll /result/fc-JobID to get results.\"}\n"
        "```\n\n"
        "### 2. Poll for results\n\n"
        "```bash\n"
        "curl \"https://marija-vukic--market-foundry-api-fastapi-app.modal.run/result/fc-JobID\"\n"
        "```\n\n"
        "While processing:\n"
        "```json\n"
        "{\"status\":\"processing\",\"message\":\"Still running, check back soon.\"}\n"
        "```\n\n"
        "When complete: returns the full result JSON with all extracted triples."
        "```\n\n"
        "### 3. Query your graph via API in your terminal (optional)\n\n"
        "Once triples are written to Neo4j, you can query your graph directly via the API:\n\n"
        "```bash\n"
        "curl -X POST \"https://marija-vukic--market-foundry-api-fastapi-app.modal.run/query\" \\\n"
        "  -F \"cypher=MATCH (s)-[r]->(o) RETURN s.name, type(r), o.name LIMIT 25\" \\\n"
        "  -F \"neo4j_uri=neo4j+s://xxxx.databases.neo4j.io\" \\\n"
        "  -F \"neo4j_username=neo4j\" \\\n"
        "  -F \"neo4j_password=yourpassword\"\n"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".html", ".json"}


@web_app.get("/", tags=["Info"])
async def root():
    return {
        "name": "MarketFoundry API",
        "version": "1.0.0",
        "docs": "/docs",
        "github": "https://github.com/jessicabat/market-foundry",
        "description": (
            "Upload a financial document to extract a causal knowledge graph. "
            "Optionally provide your own Neo4j credentials to persist the graph."
        ),
    }


@web_app.get("/health", tags=["Info"])
async def health():
    pipeline = MarketFoundryPipeline()
    return await pipeline.health.remote.aio()


@web_app.post("/process", tags=["Pipeline"], response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(..., description="PDF, DOCX, TXT, HTML, or JSON financial document"),
    neo4j_uri: str = Form("", description="(Optional) Your Neo4j URI — e.g. neo4j+s://xxxx.databases.neo4j.io"),
    neo4j_username: str = Form("", description="(Optional) Your Neo4j username"),
    neo4j_password: str = Form("", description="(Optional) Your Neo4j password"),
    webhook_url: str = Form("", description="(Optional) URL to POST results to when processing completes — no polling needed"),
):
    """
    **Main endpoint.** Upload a financial document to extract a causal knowledge graph.

    Returns a `job_id` immediately. Poll `/result/{job_id}` to get the output
    once processing is complete (typically 3-10 minutes).

    **Without Neo4j credentials:** Results returned as JSON only in terminal when job is POLLed.
    **With Neo4j credentials:** Triples written to your Neo4j instance + returned as JSON in terminal when job is POLLed.

    Supported formats: `.pdf`, `.docx`, `.txt`, `.html`, `.json`
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")

    # Spawn the job in the background and return a job_id immediately
    pipeline = MarketFoundryPipeline()
    call = pipeline.process_document.spawn(
        contents, file.filename, neo4j_uri, neo4j_username, neo4j_password, webhook_url
    )
    response = {
        "job_id": call.object_id,
        "status": "processing",
        "poll_url": f"/result/{call.object_id}",
    }
    if webhook_url:
        response["message"] = f"Job started for '{file.filename}'. Results will be POSTed to {webhook_url} when complete."
        response["webhook_url"] = webhook_url
    else:
        response["message"] = f"Job started for '{file.filename}'. Poll /result/{call.object_id} to get results."
    return JSONResponse(content=response)


@web_app.get("/result/{job_id}", tags=["Pipeline"], response_model=ResultResponse)
async def get_result(job_id: str):
    """
    Poll this endpoint with the `job_id` from `/process` to check if your
    extraction job is complete.

    Returns `{"status": "processing"}` while running,
    or the full result JSON when done.
    """
    from modal.functions import FunctionCall
    try:
        call = FunctionCall.from_id(job_id)
        result = call.get(timeout=0)  # non-blocking check
        return JSONResponse(content={"status": "complete", "result": result})
    except TimeoutError:
        return JSONResponse(content={"status": "processing", "message": "Still running, check back soon."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@web_app.post("/query", tags=["Graph"], response_model=QueryResponse)
async def query_graph(
    cypher: str = Form(..., description="Cypher query (MATCH/CALL only)"),
    neo4j_uri: str = Form(..., description="Your Neo4j URI"),
    neo4j_username: str = Form(..., description="Your Neo4j username"),
    neo4j_password: str = Form(..., description="Your Neo4j password"),
):
    """
    Run a read-only Cypher query against **your own** Neo4j knowledge graph.

    Example:
    ```
    MATCH (s)-[r]->(o) RETURN s.name, type(r), o.name LIMIT 25
    ```
    """
    stripped = cypher.strip().upper()
    if not (stripped.startswith("MATCH") or stripped.startswith("CALL")):
        raise HTTPException(status_code=400, detail="Only MATCH and CALL queries are permitted.")

    pipeline = MarketFoundryPipeline()
    try:
        result = await pipeline.query_graph.remote.aio(
            cypher, neo4j_uri, neo4j_username, neo4j_password
        )
        return JSONResponse(content={"results": result, "count": len(result)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
