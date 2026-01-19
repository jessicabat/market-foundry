"""
Module for running the market-foundry extraction framework.

This module serves as the main entry point for processing input files through
the extraction pipeline. It handles command-line argument parsing, document
classification, and text vectorization using pre-trained models.

Main functionality:
- Parses command-line arguments for input file paths
- Loads files using the utility load_file function
- Classifies documents using a trained TF-IDF + model pipeline
- Outputs document type classifications for each file

Usage:
    python run.py --files <file_path1> [<file_path2> ...]

Example:
    python run.py --files data/file1.txt data/file2.txt
"""

import array
import os
import pandas as pd
import argparse
from annotated_types import doc
import yaml
import joblib
from utils import *
from models import *
from utils.process import load_file, get_basename, expand_path

def main():
    # Create command-line argument parser
    parser = argparse.ArgumentParser(description='Run the extraction framework.')
    parser.add_argument('--file', type=str, required=True, 
                        help='Path to the input file or directory.')

    # Parse command-line arguments
    args = parser.parse_args()
    
    # Load and process the files
    file_paths = expand_path(args.file)
    
    loaded_files = []
    for file_path in file_paths:
        loaded_file = load_file(file_path)
        loaded_files.append(loaded_file)
        print("File loaded:", get_basename(file_path))
        
    # Load the trained document classification model from the models folder
    model_path = os.path.join(os.path.dirname(__file__), "models", "Document_Classifier.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Ensure there is a model to load in the specified path.")
    model = joblib.load(model_path)
        
    # Load the TF-IDF vectorizer from the models folder
    vectorizer_path = os.path.join(os.path.dirname(__file__), "models", "TFIDF_Vectorizer.joblib")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}. Ensure there is a model to load in the specified path.")
    vectorizer = joblib.load(vectorizer_path)
        
    # Extract raw text from loaded files
    texts = []
    # Note: loaded_file may itself be a list of pages
    for loaded_file in loaded_files: 
        # Combine all pages into a single string
        combined_text = " ".join(page.page_content for page in loaded_file)
        texts.append((loaded_file[0].metadata.get('source'), combined_text))
        
    # Transform texts using the loaded TF-IDF vectorizer    
    vectorized_texts = vectorizer.transform([text[1] for text in texts])
            
    # Classify the document types of each loaded file
    classifications = {}
    for file, text in zip([text[0] for text in texts], vectorized_texts):
        classification = model.predict(text)
        classifications[file] = classification[0]
        
    # # Output the classifications
    # for file, classification in classifications.items():
    #     print(f"File: {get_basename(file)} => Document Type: {classification}")
    
    df_classifications = pd.DataFrame(list(classifications.items()), columns=["File", "Document Type"])
    df_classifications["File"] = df_classifications["File"].apply(get_basename)
    print("Document Classifications:\n", df_classifications)
    
if __name__ == "__main__":
    main()