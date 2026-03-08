"""
This script reads the extracted triples from the JSON file and merges them into a Neo4j graph database. Use this if you have encountered issues with the Neo4j integration in the main pipeline or if you want to perform the graph construction step separately.

It uses the `neo4j` Python driver to connect to the database and execute Cypher queries for merging nodes and relationships. The script processes the triples in batches for efficiency.
Make sure to set the Neo4j connection details (URL, username, password) in your environment variables or a .env file before running this script.
"""

import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

BATCH_SIZE = 100

def merge_market_foundry_triples(url, user, password, json_file_path):
    driver = GraphDatabase.driver(url, auth=(user, password))

    with open(json_file_path, "r") as f:
        data = json.load(f)

    triples = []
    for entry in data:
        for triple in entry.get("triple_list", []):
            triples.append({
                "head": triple["head"],
                "tail": triple["tail"],
                "relation": triple["relation"],
                "head_label": triple["head_type"].replace(" ", "_").lower(),
                "tail_label": triple["tail_type"].replace(" ", "_").lower(),
                "rel_type": triple["relation_type"].replace(" ", "_").lower()
            })

    def _batch_merge(tx, batch):
        query = """
        UNWIND $batch AS triple
        CALL (triple) {
            CALL apoc.merge.node([triple.head_label], {name: triple.head}) YIELD node AS h
            CALL apoc.merge.node([triple.tail_label], {name: triple.tail}) YIELD node AS t
            CALL apoc.merge.relationship(h, triple.rel_type, {}, {}, t) YIELD rel
            SET rel.description = triple.relation
        }
        """
        tx.run(query, batch=batch)

    with driver.session() as session:
        for i in range(0, len(triples), BATCH_SIZE):
            batch = triples[i:i + BATCH_SIZE]
            session.execute_write(_batch_merge, batch)

    driver.close()
    print("Ingestion complete.")
    
if __name__ == "__main__":
    url = os.getenv("NEO4J_URL", "") 
    user = os.getenv("NEO4J_USERNAME", "")
    password = os.getenv("NEO4J_PASSWORD", "")
    json_file_name = "extraction_result.json"
    json_file_path = os.path.join(os.path.dirname(__file__), json_file_name)
    
    # Ensure variables exist before calling
    if url and user and password:
        merge_market_foundry_triples(url, user, password, json_file_path)
    else:
        print("Error: Neo4j environment variables not set.")
    
    