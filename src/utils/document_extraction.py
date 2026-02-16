"""
Routing to OneKE/src.run.py for Knowledge Extraction
"""

import os
from dotenv import load_dotenv
import yaml
import tempfile
import subprocess
import pandas as pd
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader, JSONLoader
import time
from utils import *
from utils.document_classification import get_basename

load_dotenv()  # Load environment variables from .env file

# Resolve repository root (two levels up from this file)
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CONFIG_DIR = os.path.join(REPO_ROOT, "Configs")
ONEKE_RUN = os.path.join(REPO_ROOT, "OneKE", "src", "run.py")

CLASS_TO_CONFIG = {
    "Earnings Call Transcript": "earnings_call.yaml",
    "News Article": "news_article.yaml",
    "Research Paper": "research_paper.yaml",
    "SEC Filing": "sec_filing.yaml",
    "Press Release": "press_release.yaml",
}

def run_oneke_from_text(file_path, text, document_type, section_name=None):
    
    output_base = os.path.join(REPO_ROOT, "new_outputs", get_basename(file_path))
    os.makedirs(output_base, exist_ok=True)
    case_dir = os.path.join(output_base, f"{document_type}_{section_name or 'full'}")

    
    start_time = time.time()
    base_config_name = CLASS_TO_CONFIG.get(document_type)
    if base_config_name is None:
        return

    base_config_path = os.path.join(CONFIG_DIR, base_config_name)

    # Load base config
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # --- SCHEMA CONFIG HERE ---
    
    # Supports:
    # - Open Source: LLaMA, Qwen, MiniCPM, ChatGLM
    # - Closed Source: ChatGPT, DeepSeek, LocalServer
    
    config['model']['category'] = "Qwen" 
    config['model']['model_name_or_path'] = "Qwen/Qwen2.5-1.5B-Instruct"
    # config['model']['api_key'] = os.getenv("LM_STUDIO_API_KEY") 
    # config['model']['base_url'] = os.getenv("LM_STUDIO_LOCAL_URL") 
    # config['model']['base_url'] = os.getenv("LM_STUDIO_NETWORK_URL") 
    
    config["extraction"]["text"] = text
    config['extraction']["update_case"] = False # Controls whether to update the case repository with new extraction results.
    config['extraction']["show_trajectory"] = True #False # Controls whether to display the intermediate steps of extraction.

    config["extraction"]["case_dir"] = f"/tmp/oneke_{document_type}_{section_name}"
    
    # config['construct']['database'] = "Neo4j"
    # config['construct']['url'] = os.getenv("NEO4J_URL")
    # config['construct']['username'] = os.getenv("NEO4J_USERNAME")
    # config['construct']['password'] = os.getenv("NEO4J_PASSWORD")
    
    # --- END SCHEMA CONFIG ---

    # Write temp config
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.safe_dump(config, tmp)
        temp_config_path = tmp.name

    # Run OneKE with safe cleanup
    try:
        subprocess.run(
            [
                "python",
                ONEKE_RUN,
                "--config",
                temp_config_path,
            ],
            check=True,
        )
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    if section_name:
        print(f"Processed {get_basename(file_path)} - Section: {section_name} in {time.time() - start_time:.2f} seconds.\n")
    else:
        print(f"Processed {get_basename(file_path)} in {time.time() - start_time:.2f} seconds.\n")
            
# Run OneKE for knowledge extraction on each section of each document
def run_oneke_pipeline(sectioned_documents, text_lookup, classifications):
    for file, sections in sectioned_documents.items():
        if not any(sections.values()):
            start_time = time.time()
            print(f"No sections identified for {get_basename(file)}.")
            print(f"Running OneKE on the entire document.\n")
            run_oneke_from_text(file, text_lookup[file], classifications[file])
            end_time = time.time()
            print(f"Time taken for {get_basename(file)}: {end_time - start_time:.2f} seconds\n")
        else:
            start_time = time.time()
            for section_name, section_text in sections.items():
                print(f"Running OneKE on {get_basename(file)} - Section: {section_name}")
                run_oneke_from_text(file, section_text, classifications[file], section_name=section_name)
            end_time = time.time()
            print(f"Time taken for {get_basename(file)}: {end_time - start_time:.2f} seconds\n")