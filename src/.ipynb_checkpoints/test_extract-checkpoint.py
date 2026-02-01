import argparse
import os
import yaml
from nltk.tokenize import sent_tokenize
from utils import *
from utils.process import load_file

def main():
    # Create command-line argument parser
    parser = argparse.ArgumentParser(description='Run the extraction framefork.')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the input file.')

    # Parse command-line arguments
    args = parser.parse_args()
    # Load and process the file
    loaded_file = load_file(args.file)
    
    print(loaded_file)


if __name__ == "__main__":
    main()