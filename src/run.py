"""
Module for running the market-foundry extraction framework.

This module serves as the main entry point for processing input files through
the extraction pipeline. It handles command-line argument parsing, document
classification, and text vectorization using pre-trained models.

Main functionality:
- Parses command-line arguments for input file or folder paths
- Loads files using the utility load_file function
- Classifies documents using a trained TF-IDF + model pipeline
- Outputs document type classifications for each file

Usage:
    python run.py --file <file_path> or --folder <folder_path>

Example:
    python run.py --file data/file1.txt
    python run.py --folder data/
"""

import os
import argparse
import yaml
import array
from annotated_types import doc
from utils import *
from models import *
from utils.process import *

def main():
    # Create command-line argument parser
    parser = argparse.ArgumentParser(description='Pass in file or folder path to process.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str,
                       help='Path to a single input file.')
    group.add_argument('--folder', type=str,
                       help='Path to a folder containing input files.')

    # Parse command-line arguments
    args = parser.parse_args()
    
    # Load and process the files
    if args.file:
        file_paths = [args.file]
    else:
        file_paths = expand_path(args.folder)
    
    loaded_files = load_files(file_paths)
        
    # Load the trained document classification model from the models folder
    model_path = os.path.join(os.path.dirname(__file__), "models", "Document_Classifier.joblib")
    model = load_document_classification_model(model_path)
        
    # Load the TF-IDF vectorizer from the models folder
    vectorizer_path = os.path.join(os.path.dirname(__file__), "models", "TFIDF_Vectorizer.joblib")
    vectorizer = load_tfidf_vectorizer(vectorizer_path)
        
    # Extract raw text from loaded files
    texts = extract_text(loaded_files)
        
    # Classify documents
    classifications = classify_document_types(model, vectorizer, texts)
        
    # Output the classifications
    output_classifications(classifications)
    
    # Run OneKE for knowledge extraction
    run_oneke_extraction(classifications)
    
if __name__ == "__main__":
    main()