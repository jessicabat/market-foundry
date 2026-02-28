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

# Extract topics from the documents using the topic_extractor module
def extract_topics_and_run_oneke(texts, classifications, text_lookup):
    total_files = len(texts)
    index = 1
    for file, text in texts:
        file_name = os.path.splitext(file)[0]
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
            
                # If you later run another pipeline stage that needs the YAMLs,
                # run it HERE inside the with-block.
        except Exception as e:
            print(f"Error writing YAML files for {file}")
            print(f"Running OneKE using default config for {classifications[file]} due to YAML generation failure.\n")
            run_oneke_from_text(file, text_lookup[file], classifications[file])
        finally:
            print(f"Completed processing {index} of {total_files} files.\n")
            index += 1

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
        print(f"Processed {get_basename(file_path)} - Section: {section_name} in {time.time() - start_time:.2f} seconds.")
    else:
        print(f"Processed {get_basename(file_path)} in {time.time() - start_time:.2f} seconds.")
            
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