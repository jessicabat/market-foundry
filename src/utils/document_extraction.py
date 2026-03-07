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
import topic_extractor, yaml_generator

load_dotenv()  # Load environment variables from .env file

# Resolve repository root (two levels up from this file)
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CONFIG_DIR = os.path.join(REPO_ROOT, "Configs")
ONEKE_RUN = os.path.join(REPO_ROOT, "OneKE", "src", "run.py")
extraction_config_path = os.path.join(REPO_ROOT, "src", "utils", "extraction_config.yaml")

# Load reference config for OneKE extraction
with open(extraction_config_path) as f:
        content = os.path.expandvars(f.read())
reference_config = yaml.safe_load(content)

CLASS_TO_CONFIG = {
    "Earnings Call Transcript": "earnings_call.yaml",
    "News Article": "news_article.yaml",
    "Research Paper": "research_paper.yaml",
    "SEC Filing": "sec_filing.yaml",
    "Press Release": "press_release.yaml",
}

import json as _json_mod
import glob as _glob_mod

def _read_and_clear_triples():
    """Read triples from OneKE results file, searching multiple paths."""
    candidates = _glob_mod.glob("/root/**/extraction_result.json", recursive=True)
    candidates += [
        "/root/Results/extraction_result.json",
        "/root/market-foundry/Results/extraction_result.json",
        os.path.join(os.getcwd(), "Results", "extraction_result.json"),
    ]
    print(f"[_read_and_clear_triples] cwd={os.getcwd()}")
    print(f"[_read_and_clear_triples] candidates={candidates}")
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = _json_mod.load(f)
            # Handle all formats OneKE might write:
            # 1. [{"triple_list": [...]}]  — array of extraction objects (most common)
            # 2. {"triple_list": [...]}    — single extraction object
            # 3. [{"head": ...}, ...]      — flat triple array
            if isinstance(data, list):
                triples = []
                for item in data:
                    if isinstance(item, dict) and "triple_list" in item:
                        triples.extend(item["triple_list"])
                    elif isinstance(item, dict) and item.get("head") and item.get("tail"):
                        triples.append(item)
            elif isinstance(data, dict):
                triples = data.get("triple_list", [])
            else:
                triples = []
            triples = [t for t in triples if isinstance(t, dict) and t.get("head") and t.get("tail")]
            with open(path, "w") as f:
                _json_mod.dump([], f)  # keep array shape to match OneKE append logic
            print(f"[_read_and_clear_triples] Read {len(triples)} triples from {path}")
            return triples
        except Exception as e:
            print(f"[_read_and_clear_triples] Error reading {path}: {e}")
    print("[_read_and_clear_triples] WARNING: no results file found")
    return []

# Extract topics from the documents using the topic_extractor module
def extract_topics_and_run_oneke(texts, classifications, text_lookup):
    total_files = len(texts)
    index = 1
    all_triples = []
    for file, text in texts:
        file_name = get_basename(file).split(".")[0]  # Get filename without extension for YAML naming
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                
                if reference_config.get("model", {}).get("category", "LocalServer") == "LocalServer":
                    topics = topic_extractor.extract_topics_openai(text)
                    
                    topic_configs = yaml_generator.generate_yaml_configs_openai(
                        file_name,
                        classifications[file],
                        topics
                    )
                else:
                    topics = topic_extractor.extract_topics(text)
                
                    topic_configs = yaml_generator.generate_yaml_configs(
                        file_name,
                        classifications[file],
                        topics
                    )
                
                yaml_generator.write_yaml_files(
                    topic_configs,
                    output_dir=temp_dir,
                    input_file_path=file
                )

                for temp_file in os.listdir(temp_dir):
                    run_oneke_from_text(
                        file_path=os.path.join(temp_dir, temp_file),
                        text=text_lookup[file],
                        document_type=classifications[file],
                        base_config_dir=temp_dir,
                    )
                    # Collect triples after each YAML run before file gets overwritten
                    all_triples.extend(_read_and_clear_triples())

        except Exception as e:
            print(f"Error writing YAML files for {file}")
            print(f"Running OneKE using default config for {classifications[file]} due to YAML generation failure.\n")
            run_oneke_from_text(file, text_lookup[file], classifications[file])
            all_triples.extend(_read_and_clear_triples())
        finally:
            print(f"Completed processing {index} of {total_files} files.\n")
            index += 1
    return all_triples

def run_oneke_from_text(file_path, text, document_type, section_name=None, base_config_dir=None):
    start_time = time.time()
    base_config_name = CLASS_TO_CONFIG.get(document_type)
    if base_config_name is None:
        return

    if base_config_dir:
        base_config_path = file_path
    else:
        base_config_path = os.path.join(CONFIG_DIR, base_config_name)

    # Load base config
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Merge reference config into temp config
    for key in reference_config:
        if key in config and isinstance(config[key], dict):
            config[key].update(reference_config[key])
        else:
            config[key] = reference_config[key]
            
    # Update text in temp config
    config['extraction']['text'] = text
    
    # Strip construct block before passing to OneKE subprocess.
    # We handle Neo4j writes ourselves after cleaning the triple list,
    # because OneKE's convert.py crashes on malformed triples (plain strings in triple_list).
    config_for_oneke = {k: v for k, v in config.items() if k != "construct"}

    # Write temp config
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.safe_dump(config_for_oneke, tmp)
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
        print(f"Processed {get_basename(file_path)} - Section: {section_name} in {time.time() - start_time:.2f} seconds.")
    else:
        print(f"Processed {get_basename(file_path)} in {time.time() - start_time:.2f} seconds.")
            
