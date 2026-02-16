# src/oneke_runner.py
import sys
import os
import yaml
import json

# --- Add OneKE to path if needed, or ensure these imports work ---
# Assuming you have OneKE code in your PYTHONPATH or subdirectory
ONEKE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../OneKE/src"))
if ONEKE_PATH not in sys.path:
    sys.path.insert(0, ONEKE_PATH)

from pipeline import Pipeline
import models
from utils import load_extraction_config

def run_oneke_extraction(raw_text, schema_path):
    """
    Orchestrates OneKE extraction for a single document text.
    """
    print(f"  [OneKE] Loading schema from {schema_path}...")
    
    # 1. Load Schema
    # You might need to use OneKE's load_extraction_config if it does special parsing
    # or just standard yaml load if the config is simple.
    with open(schema_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup Model (Standard OneKE logic)
    model_config = config['model']
    
    # Check for vLLM or standard model
    if model_config.get('vllm_serve', False):
        from models import LocalServer
        model = LocalServer(model_config['model_name_or_path'])
    else:
        # Dynamic import of model class (Qwen, LLaMA, etc.)
        model_cls = getattr(models, model_config['category'], None)
        if not model_cls:
            raise ValueError(f"Model category {model_config['category']} not found in OneKE models.")
        
        # Instantiate
        api_key = model_config.get('api_key', "")
        if api_key:
            model = model_cls(model_config['model_name_or_path'], api_key, model_config.get('base_url', ""))
        else:
            model = model_cls(model_config['model_name_or_path'])

    # 3. Setup Pipeline
    pipe = Pipeline(model)
    ext_cfg = config['extraction']

    # 4. Run Extraction (Override text/file inputs)
    # Note: We pass raw_text directly, bypassing file reading
    print(f"  [OneKE] Extracting triples from {len(raw_text)} chars...")
    
    result = pipe.get_extract_result(
        task=ext_cfg.get('task', 'Triple'),
        instruction=ext_cfg.get('instruction', ''),
        text=raw_text,  # <--- PASS TEXT DIRECTLY
        output_schema=ext_cfg.get('output_schema', ''),
        constraint=ext_cfg.get('constraint', []),
        use_file=False,  # We are not using a file path
        file_path="", 
        truth=ext_cfg.get('truth', ''),
        mode=ext_cfg.get('mode', 'quick'),
        update_case=ext_cfg.get('update_case', False),
        show_trajectory=ext_cfg.get('show_trajectory', False),
        construct=None, # We handle graph construction ourselves
        iskg=False
    )

    # 5. Result Parsing
    if isinstance(result, tuple):
        extraction_output = result[0]
    else:
        extraction_output = result

    triples = []
    
    # CASE A: JSON String -> Parse it first
    if isinstance(extraction_output, str):
        try:
            import json
            extraction_output = json.loads(extraction_output)
        except json.JSONDecodeError:
            print(f"  [OneKE] Warning: Could not parse output JSON. Raw: {extraction_output[:50]}...")
            return []

    # CASE B: Dictionary Wrapper ({"triple_list": [...]})
    if isinstance(extraction_output, dict):
        if "triple_list" in extraction_output:
            triples = extraction_output["triple_list"]
        else:
            # Maybe it returned a single object or wrong schema?
            # Start debugging by dumping what keys exist
            # print(f"DEBUG: Keys found: {extraction_output.keys()}")
            triples = [extraction_output] # Fallback, might be wrong format but saves data

    # CASE C: Direct List ([...])
    elif isinstance(extraction_output, list):
        triples = extraction_output
        
    return triples
