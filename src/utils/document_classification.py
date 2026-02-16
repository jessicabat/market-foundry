""" Document Classification 
Functions to process text and files for data extraction tasks. Supported file formats include .pdf, .txt, .docx, .html, and .json.
"""
import re
import os
import joblib
import pandas as pd
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader, JSONLoader
import time

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".html", ".json"}

# Basename extraction
def get_basename(file_path):
    return os.path.basename(file_path)

# Load file based on its extension
def load_file(file_path):
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".html"):
            loader = BSHTMLLoader(file_path)
        elif file_path.endswith(".json"):
            loader = JSONLoader(file_path, jq_schema=".text")
        else:
            raise ValueError("Unsupported file format")
        return loader.load()
    except Exception as e:
        print(f"Skipping file due to load error: {file_path} ({e})")
        return []

# Load multiple files given a list of file paths
def load_files(file_paths):
    start_time = time.time()
    loaded_files = []
    for file_path in file_paths:
        loaded_file = load_file(file_path)
        loaded_files.append(loaded_file)
    print(f"Loaded {len(loaded_files)} files in {time.time() - start_time:.2f} seconds.")
    return loaded_files

# Expand a given path if it is a directory to process all supported files
def expand_path(path):
    expanded = []
    if os.path.isfile(path):
        expanded.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                    expanded.append(os.path.join(root, file))
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    return expanded

# Extract raw text from loaded files
def extract_text(loaded_files):
    texts = []
    for loaded_file in loaded_files:
        combined_text = "\n".join(page.page_content for page in loaded_file)
        texts.append((loaded_file[0].metadata.get('source'), combined_text))
    return texts

def normalize_for_knn(text: str) -> str:
    # Collapse runs of 3+ newlines into 2, and 2+ spaces into 1
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

# Load the trained document classification model
def load_document_classification_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Ensure there is a model to load in the specified path.")
    return joblib.load(model_path)

# Load the TF-IDF vectorizer
def load_tfidf_vectorizer(vectorizer_path):
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}. Ensure there is a model to load in the specified path.")
    return joblib.load(vectorizer_path)

# Classify documents using the provided model and vectorizer
def classify_document_types(model, vectorizer, texts):
    vectorized_texts = vectorizer.transform([text[1] for text in texts])
    classifications = {}
    for file, text in zip([text[0] for text in texts], vectorized_texts):
        classification = model.predict(text)
        classifications[file] = classification[0]
    return classifications

# Output document classifications
def output_classifications(classifications):
    # df_classifications = pd.DataFrame(list(classifications.items()), columns=["File", "Document Type"])
    # df_classifications["File"] = df_classifications["File"].apply(get_basename)
    # print("Document Classifications:\n", df_classifications, "\n")
    
    # Extract labels and confidences for DataFrame
    files = []
    doc_types = []
    confidences = []
    
    for file, (label, score) in classifications.items():
        files.append(file)
        doc_types.append(label)
        confidences.append(f"{score:.3f}")
    
    df_classifications = pd.DataFrame({
        "File": files,
        "Document Type": doc_types,
        "Confidence": confidences
    })
    
    df_classifications["File"] = df_classifications["File"].apply(get_basename)
    
    print("Document Classifications:\n", df_classifications, "\n")

    
# """ Document Sectioning for src/run.py
# Functions to run the document sectioning pipeline.
# """

# SECTION_HEADERS = {
#     "overview": [
#         "overview", "business overview", "company overview",
#         "about the company", "item 1.", "item 1 ", "corporate overview",
#         "company description", "business description"
#     ],
#     "financials": [
#         "financial statements", "consolidated statements",
#         "balance sheet", "income statement",
#         "statement of operations", "cash flow", "item 8.",
#         "statement of financial position", "statement of comprehensive income",
#         "statements of cash flows", "financial position"
#     ],
#     "mdna": [
#         "management's discussion", "md&a", "results of operations", 
#         "item 7.", "management discussion and analysis",
#         "operating performance", "financial performance", "performance analysis"
#     ],
#     "risk_factors": [
#         "risk factors", "risks and uncertainties",
#         "forward-looking statements", "item 1a.", "risk management",
#         "potential risks", "business risks", "market risks"
#     ],
#     "notes": [
#         "notes to financial statements", "accounting policies",
#         "significant accounting", "footnotes", "note disclosures",
#         "accounting standards", "financial note"
#     ],
#     "outlook": [
#         "outlook", "guidance", "future outlook",
#         "expectations", "forecast", "future prospects", "forward guidance"
#     ],
#     "legal": [
#         "legal proceedings", "regulatory matters",
#         "compliance", "litigation", "legal notices", "regulatory compliance",
#         "legal issues", "court proceedings"
#     ],
#     "introduction": [
#         "introduction", "executive summary", "summary",
#         "background", "item 1.", "related work", "preface", "prologue", "abstract"
#     ],
#     "methodology": [
#         "methodology", "approach", "data and methods",
#         "research design", "experimental setup", "experiments",
#         "methods and materials", "research methodology", "techniques"
#     ],
#     "results": [
#         "results", "findings", "outcomes", "item 2.", "empirical results", "key findings"
#     ],
#     "discussion": [
#         "discussion", "interpretation", "implications", "item 3.",
#         "analysis and discussion", "interpretation of results"
#     ],
#     "conclusion": [
#         "conclusion", "conclusions", "summary of findings",
#         "final thoughts", "item 4.", "concluding remarks", "summary"
#     ],
# }

# # Heuristic to determine if a line is likely a section header based on formatting and content cues
# def is_header_like(line):
#     stripped = line.strip()
#     return (
#         5 < len(stripped) < 80 and
#         not stripped.endswith(".") and
#         (
#             stripped.isupper() or
#             stripped[0].isdigit() or
#             stripped.istitle()
#         )
#     )

# # Section a single document based on the defined headers and heuristics
# def section_document(text):
#     sections = {key: [] for key in SECTION_HEADERS.keys()}
#     current_section = None

#     for line in text.splitlines():
#         line_lower = line.lower()

#         if is_header_like(line):
#             for section, headers in SECTION_HEADERS.items():
#                 if any(h in line_lower for h in headers):
#                     current_section = section
#                     break

#         if current_section:
#             sections[current_section].append(line)

#     return {k: "\n".join(v) for k, v in sections.items()}

# # Section documents based on their classifications
# def section_documents(texts):
#     sectioned_documents = {}
#     # For each document, run the sectioning algorithm to identify sections based on the defined headers and heuristics. Store the identified sections in a dictionary for each document.
#     for file, text in texts:
#         sections = section_document(text)
#         # keep only filled sections
#         filled_sections_only = {
#             section: content
#             for section, content in sections.items()
#             if content.strip()
#         }
#         # List the identified sections for each document
#         # print(
#         #     f"Sections for {get_basename(file)}:\n",
#         #     list(filled_sections_only.keys())
#         # )
#         sectioned_documents[file] = filled_sections_only
#     return sectioned_documents

# """
# Routing to OneKE/src.run.py for Knowledge Extraction
# """

# # Resolve repository root (two levels up from this file)
# REPO_ROOT = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..")
# )

# CONFIG_DIR = os.path.join(REPO_ROOT, "Configs")
# ONEKE_RUN = os.path.join(REPO_ROOT, "OneKE", "src", "run.py")

# CLASS_TO_CONFIG = {
#     "Earnings Call Transcript": "earnings_call.yaml",
#     "News Article": "news_article.yaml",
#     "Research Paper": "research_paper.yaml",
#     "SEC Filing": "sec_filing.yaml",
#     "Press Release": "press_release.yaml",
# }

# def run_oneke_from_text(file_path, text, document_type, section_name=None):
#     start_time = time.time()
#     base_config_name = CLASS_TO_CONFIG.get(document_type)
#     if base_config_name is None:
#         return

#     base_config_path = os.path.join(CONFIG_DIR, base_config_name)

#     # Load base config
#     with open(base_config_path, "r") as f:
#         config = yaml.safe_load(f)
        
#     # For model config, we use LocalServer (LM Studio)
#     config['model']['category'] = "LocalServer"
#     config['model']['model_name_or_path'] = "qwen3-4b-instruct-2507"
#     config['model']['api_key'] = os.getenv("LM_STUDIO_API_KEY") 
#     # config['model']['base_url'] = os.getenv("LM_STUDIO_LOCAL_URL")
#     config['model']['base_url'] = os.getenv("LM_STUDIO_NETWORK_URL") 
    
#     '''For extraction config, we set the text to be the section text, and set use_file to false since we are directly passing the text. We also set mode to "customized" to use the customized 3-agent process for better performance on financial documents. You can further customize the process by modifying the "customized" mode in src/config.yaml.
#     '''
    
#     config["extraction"]["use_file"] = False
#     config["extraction"]["text"] = text
#     config['extraction']["mode"] = "customized"
#     config['extraction']["update_case"] = False
#     config['extraction']["show_trajectory"] = False
    
#     # config['construct']['database'] = "Neo4j"
#     # config['construct']['url'] = os.getenv("NEO4J_URL")
#     # config['construct']['username'] = os.getenv("NEO4J_USERNAME")
#     # config['construct']['password'] = os.getenv("NEO4J_PASSWORD")

#     # Write temp config
#     with tempfile.NamedTemporaryFile(
#         mode="w", suffix=".yaml", delete=False
#     ) as tmp:
#         yaml.safe_dump(config, tmp)
#         temp_config_path = tmp.name

#     # Run OneKE with safe cleanup
#     try:
#         subprocess.run(
#             [
#                 "python",
#                 ONEKE_RUN,
#                 "--config",
#                 temp_config_path,
#             ],
#             check=True,
#         )
#     finally:
#         if os.path.exists(temp_config_path):
#             os.remove(temp_config_path)
#     if section_name:
#         print(f"Processed {get_basename(file_path)} - Section: {section_name} in {time.time() - start_time:.2f} seconds.\n")
#     else:
#         print(f"Processed {get_basename(file_path)} in {time.time() - start_time:.2f} seconds.\n")
            
# # Run OneKE for knowledge extraction on each section of each document
# def run_oneke_pipeline(sectioned_documents, text_lookup, classifications):
#     for file, sections in sectioned_documents.items():
#         if not any(sections.values()):
#             start_time = time.time()
#             print(f"No sections identified for {get_basename(file)}.")
#             print(f"Running OneKE on the entire document.\n")
#             run_oneke_from_text(file, text_lookup[file], classifications[file])
#             end_time = time.time()
#             print(f"Time taken for {get_basename(file)}: {end_time - start_time:.2f} seconds\n")
#         else:
#             start_time = time.time()
#             for section_name, section_text in sections.items():
#                 print(f"Running OneKE on {get_basename(file)} - Section: {section_name}")
#                 run_oneke_from_text(file, section_text, classifications[file], section_name=section_name)
#             end_time = time.time()
#             print(f"Time taken for {get_basename(file)}: {end_time - start_time:.2f} seconds\n")