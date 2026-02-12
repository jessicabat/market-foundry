import os
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

class GraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def push_triples(self, triples, doc_metadata=None):
        """
        Pushes a list of triples to Neo4j.
        Triples expected format: [{'head': 'Entity1', 'head_type': 'Type1', 'relation': 'REL', 'tail': 'Entity2', 'tail_type': 'Type2'}, ...]
        """
        if not triples:
            return

        with self.driver.session() as session:
            # 1. Create Document Node (Optional but recommended for lineage)
            doc_id = doc_metadata.get('id', 'unknown') if doc_metadata else 'unknown'
            doc_name = doc_metadata.get('filename', 'unknown') if doc_metadata else 'unknown'
            
            session.execute_write(self._create_document_node, doc_id, doc_name, doc_metadata)

            # 2. Insert Triples
            for triple in triples:
                session.execute_write(self._create_triple, triple, doc_id)

    @staticmethod
    def _create_document_node(tx, doc_id, doc_name, metadata):
        query = (
            "MERGE (d:Document {id: $doc_id}) "
            "SET d.name = $doc_name, d.type = $doc_type, d.ingested_at = datetime()"
        )
        tx.run(query, doc_id=doc_id, doc_name=doc_name, doc_type=metadata.get('document_type', 'UNKNOWN'))

    @staticmethod
    def _create_triple(tx, triple, doc_id):
        # Extract fields, handling variations in keys
        head = triple.get('head') or triple.get('subject')
        tail = triple.get('tail') or triple.get('object')
        relation = triple.get('relation') or triple.get('predicate')
        
        # Clean labels (Neo4j labels cannot have spaces usually)
        head_type = triple.get('head_type', 'Entity').replace(" ", "_")
        tail_type = triple.get('tail_type', 'Entity').replace(" ", "_")
        relation_type = triple.get('relation_type', 'RELATED_TO').replace(" ", "_").upper()

        if not head or not tail: return

        # Merge Nodes
        query = (
            f"MERGE (h:{head_type} {{name: $head}}) "
            f"MERGE (t:{tail_type} {{name: $tail}}) "
            f"MERGE (h)-[r:{relation_type}]->(t) "
            "SET r.source_doc_id = $doc_id"
        )
        tx.run(query, head=head, tail=tail, doc_id=doc_id)

# function for pipeline.py to call
def push_to_neo4j(triples, doc_metadata=None):
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    builder = GraphBuilder(uri, user, password)
    try:
        builder.push_triples(triples, doc_metadata)
    finally:
        builder.close()
