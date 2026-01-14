# causal-nlp-extraction
Authors: Jessica Batbayar, Matthew Wong, Marija Vukic 

### Contributions
All authors worked together through continuous pair programming. We jointly participated in planning the workflow, implementing the extraction pipeline, configuring models, debugging code, setting up Neo4j, running experiments, and writing all documentation. Every component of the project was developed collaboratively, and all authors contributed equally to its design, execution, and analysis.

## Acknowledgements
We would like to acknowledge the following repositories and their authors for their contributions to this project. Specifically, we utilized **OneKE** for knowledge extraction and **Causal Copilot** for causal analysis.
- [OneKE](https://github.com/OpenSPG/OneKE?files=1)
- [Causal Copilot](https://github.com/Lancelot39/Causal-Copilot)

# Table of Contents
- [Workflow Overview](#workflow-overview)
- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
- [Running Knowledge Extraction](#running-knowledge-extraction)
- [Future Plans: Postprocessing and Integration with Causal Copilot](#future-plans-postprocessing-and-integration-with-causal-copilot)

### Workflow Overview
PDF Paper/Text Data → OneKE Triple Extraction → Knowledge Graph → CSV Conversion → Causal Copilot → Causal Analysis

## Introduction
We aim to use **OneKE** to extract knowledge from a paper of our choice. Using the papers located in the `FinancialPapers` directory, we perform knowledge extraction using **OneKE** to create a knowledge graph in `Neo4j`.

To replicate our results, we offer three methods: 
- `local` environment setup  
- `conda` environment setup
- `Docker` setup

First, clone our repository and ensure you have Docker Desktop or conda installed if you want to use those methods. Edit the `.yaml` files found in `FinancialConfigs` to set your desired model, extraction mode, and constraints (defined in `OneKE/src/config.yaml`). To construct a knowledge graph, ensure you have your own instance of `Neo4j` running either locally or remotely through `Neo4j AuraDB`. Then, in the `construct` section of the schema, specify your databse url, username, and password. Neo4j can be run through Docker (locally) or online (remotely). 

## Setup Instructions
To set up the environment for running knowledge extraction using **OneKE**, you have three options: local setup using `pip`, `conda` environment setup, or `Docker` setup.

#### Local Setup:
The `requirements.txt` file lists all necessary Python packages. Use this file if you want a lightweight setup without using conda.
Navigate to the `root` directory of our repo and run the command below to install the required packages:
```bash
pip install -r requirements.txt
```

#### Conda Environment Setup:
The `environment.yml` file specifies all dependencies (including PyTorch, Transformers, Neo4j driver, and OneKE requirements). Use this file if you want the exact same environment we used for our experiments.

Navigate to the `root` directory of our repo and run the commands below to create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate <environment_name>
```

#### Docker Setup:
Enter the `root` directory of our repo and run the commands below to pull our docker image and create a container for processing:

> ⚠️ **Important:** When pulling the image from Github, there are two versions: One that is compatible for **Windows** (`amd64`) and one that is compatible for **Mac** (`arm64`). To specify which version you want to pull, ensure the **tag** is either: `:latest` for `arm64`, or `:amd64` for `amd64`.

```bash
docker pull ghcr.io/mathyoutw/causal-nlp-extraction:<tag>
docker run -it \
  -v <path_to_causal-nlp-extraction>:/app/causal-nlp-extraction \
  causal-nlp-extraction
```

#### For users with an NVIDIA GPU, you can use the NVIDIA runtime to leverage GPU acceleration:

```bash
docker pull ghcr.io/mathyoutw/causal-nlp-extraction:<tag>
docker run -it --gpus all \
  -v <path_to_causal-nlp-extraction>:/app/causal-nlp-extraction \
  causal-nlp-extraction
```

#### Neo4j Database Setup:
To set up a `Neo4j` database, you have two options: using `Neo4j Aura` (remote) or `Neo4j Local`.
1. **Neo4j Aura (Remote)**:
   - Sign up for a free account at [Neo4j Aura](https://neo4j.com/cloud/aura/).
   - Create a new database instance and note down the `database ID`, `username`, and `password`.
2. **Neo4j Local**:
   - (Optional) Install `Neo4j` locally by following the instructions at [Neo4j Downloads](https://neo4j.com/download/).
   - Start the `Neo4j` server and note down the `username` and `password`.

Below are two example `construct` sections for connecting to a `Neo4j` database, either through `Neo4j Aura` (remote) or `Neo4j Local`. Choose the one that fits your setup and replace the placeholder values with your actual database credentials.

#### Neo4j Aura Example Construct Section:
```yaml
construct: # Need this for constructing Knowledge Graph
  database: Neo4j 
  url: neo4j+s://<database-id>.databases.neo4j.io
  username: neo4j # your database username.
  password: "<database_password>" # your database password.
```

#### Neo4j Local Example Construct Section:
```yaml
construct: # Need this for constructing Knowledge Graph
  database: Neo4j 
  url: bolt://localhost:7687 # your database URL，Neo4j's default port is 7687.
  username: neo4j # your database username.
  password: "<database_password>" # your database password.
```

## Running Knowledge Extraction

After opening the container or activating the environment you chose and entering the `causal-nlp-extraction` root directory, run the **`run.py`** file in the `OneKE/src/` folder. The model we chose is [**Qwen2.5-VL-7B-Instruct**](https://huggingface.co/Qwen2.5-VL-7B-Instruct) from the Hugging Face API, selected because it is open source, easy to access, and efficient for extraction. 

> ⚠️ **Important:** Some models on Hugging Face may require an access token. To do so, log in to Hugging Face, navigate to the settings, create an access token with **read** permissions. Use the command below and enter in your key to gain access to Hugging Face:

```bash
huggingface-cli login
```

After Hugging Face recognizes your key, run this command below to start knowledge extraction:

```bash
python OneKE/src/run.py --config <FinancialConfigs/path_to_yaml_file>
```

When the process finishes, you should see the resulting knowledge extracted in your terminal, and further pushed to `Neo4j` to create a knowledge graph in the explore tab of your instance.

## Future Plans: Postprocessing and Integration with Causal Copilot
Because integration with **Causal Copilot** requires `.csv` files as input rather than `.json` data, we will export the extracted data from our `Neo4j` database to CSV format that can be passed to **Causal Copilot** for causal discovery and inference. The resulting CSV can then be uploaded to **Causal Copilot** to explore causal relationships within the extracted knowledge.
