"""
src/api/graph.py
----------------
Neo4j helper for the MarketFoundry API.

OneKE writes triples to Neo4j directly when the `construct` block is set
in extraction_config.yaml. This module provides:

  1. collect_and_push_triples() — queries Neo4j after OneKE runs to count
     what was written, so the API can report it back to the caller.

  2. push_triple() — utility to manually upsert a single triple if needed.
"""

import hashlib
import re
from neo4j import Driver


def collect_and_push_triples(driver: Driver, source_filename: str) -> int:
    """
    After OneKE has run, count the total relationships in Neo4j that
    originated from this source file.

    If OneKE's construct block is disabled (default in extraction_config.yaml),
    this returns 0 — in that case the triples are printed to stdout by OneKE
    and you'd need to parse them if you want to push manually.
    """
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH ()-[r]->() WHERE r.source = $source RETURN count(r) AS cnt",
                source=source_filename,
            )
            record = result.single()
            return record["cnt"] if record else 0
    except Exception as e:
        print(f"[graph.py] Could not count triples for {source_filename}: {e}")
        return 0


def push_triple(
    driver: Driver,
    subject: str,
    predicate: str,
    obj: str,
    citation: str = "",
    section: str = "",
    doc_type: str = "",
    source: str = "",
) -> bool:
    """
    Manually upsert a single (subject, predicate, object) triple into Neo4j.
    Returns True if the relationship was created, False otherwise.
    """
    rel_type = _predicate_to_rel_type(predicate)
    triple_hash = hashlib.md5(
        f"{subject.strip().title()}|{rel_type}|{obj.strip().title()}".encode()
    ).hexdigest()[:12]

    query = f"""
    MERGE (s:Entity {{name: $subj}})
      ON CREATE SET s.created_at = timestamp()
    MERGE (o:Entity {{name: $obj}})
      ON CREATE SET o.created_at = timestamp()
    MERGE (s)-[r:{rel_type} {{triple_hash: $hash}}]->(o)
      ON CREATE SET
        r.predicate  = $predicate,
        r.citation   = $citation,
        r.section    = $section,
        r.doc_type   = $doc_type,
        r.source     = $source,
        r.created_at = timestamp()
    RETURN r
    """

    try:
        with driver.session() as session:
            result = session.run(
                query,
                subj=subject.strip().title()[:200],
                obj=obj.strip().title()[:200],
                predicate=predicate,
                hash=triple_hash,
                citation=citation,
                section=section,
                doc_type=doc_type,
                source=source,
            )
            return len(result.data()) > 0
    except Exception as e:
        print(f"[graph.py] push_triple failed: {e}")
        return False


def _predicate_to_rel_type(predicate: str) -> str:
    """Convert a natural-language predicate to a valid Neo4j relationship type."""
    rel = re.sub(r"[^a-zA-Z0-9\s]", "", predicate)
    rel = re.sub(r"\s+", "_", rel.strip()).upper()
    if not rel:
        rel = "RELATED_TO"
    if rel[0].isdigit():
        rel = "REL_" + rel
    return rel[:50]
