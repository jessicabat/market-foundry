# Market Foundry: From Documents to Intelligence – Built Your Way  
**Authors:** Jessica Batbayar, Matthew Wong, Marija Vukic  
**[Project Website](https://jessicabat.github.io/market-foundry/)**

## Contributions  
All authors collaborated continuously through pair programming. Together, we designed the workflow, implemented the extraction pipeline, configured models, debugged code, set up Neo4j, ran experiments, and authored all documentation. Every component of the project was developed collaboratively, with equal contributions from each author in terms of design, execution, and analysis.

## Acknowledgements  
We gratefully acknowledge the **OneKE** repository and its authors for enabling this work. We leveraged **OneKE** to extract structured knowledge from financial papers—making it possible to build a comprehensive knowledge graph.

-   [OneKE](https://github.com/OpenSPG/OneKE)

---

## Table of Contents  
-   [Workflow Overview](#workflow-overview)  
-   [Introduction](#introduction)  
-   [Setup Instructions](#setup-instructions)  
-   [Running Market Foundry Pipeline](#running-market-foundry-pipeline)  

### Workflow Overview  
Document (`.pdf`, `.txt`, `.docx`, `.html`, `.json`) → Document Classification → Sectioning → Knowledge Extraction → Neo4j Knowledge Graph Construction  

---

## Introduction  

Our project focuses on transforming unstructured financial documents into a structured knowledge graph using **Neo4j**. We use **OneKE**, an open-source framework, to extract entities and relationships from these documents through a pipeline that includes classification, sectioning, and semantic understanding.

We designed the system to process selected financial papers through OneKE, producing a structured knowledge graph stored in Neo4j.

To reproduce our results, users can choose between:  
-   A `conda` environment setup  
-   A `Docker` containerized setup  

First, clone the repository and ensure Docker Desktop or conda is installed. Then, follow the instructions below to set up your environment and run the pipeline.

---

## Setup Instructions  

### Two Options for Environment Setup  
We offer two methods to set up your environment: a **conda** environment or a **Docker** container.

#### Conda Environment Setup  

The `environment.yml` file defines all required dependencies. Use this if you want an identical setup to our experimental environment.

Navigate to the root directory of the repository and run:

```bash
conda env create -f environment.yml
conda activate <environment_name>
```

By default, the environment is named `market-foundry`. You can customize this name in the `environment.yml` file if preferred.

#### Docker Setup  

Go to the root directory of the repo and execute the following commands to pull our image and launch a container:

> ⚠️ **Note:** The GitHub image includes two versions—**Windows-compatible (`amd64`)** and **Mac-compatible (`arm64`)**. To select the correct version, ensure your tag is either:  
> - `:latest` for `arm64` (macOS)  
> - `:amd64` for `amd64` (Windows)

```bash
docker pull ghcr.io/mathyoutw/causal-nlp-extraction:<tag>
docker run -it \
  -v <path_to_causal-nlp-extraction>:/app/causal-nlp-extraction \
  causal-nlp-extraction
```

#### For Users with an NVIDIA GPU  

To enable GPU acceleration, use the NVIDIA runtime:

```bash
docker pull ghcr.io/mathyoutw/causal-nlp-extraction:<tag>
docker run -it --gpus all \
  -v <path_to_causal-nlp-extraction>:/app/causal-nlp-extraction \
  causal-nlp-extraction
```

#### Neo4j Database Setup  

To build the knowledge graph, you must have a running instance of **Neo4j**. You can set up your **Neo4j** instance in two ways: via **Neo4j Aura (remote)** or **Neo4j (Local)**.

1. **Neo4j Aura (Remote)**  
   - Sign up for a free account at [Neo4j Aura](https://neo4j.com/cloud/aura/).  
   - Create a new database instance and note down your **database ID**, **username**, and **password**.  

2. **Neo4j (Local)**  
   - (Optional) Install Neo4j locally by following the instructions at [Neo4j Downloads](https://neo4j.com/download/).  
   - Start the Neo4j server and record your **username** and **password**.

In the `construct` section of the schema configuration found in `src/document_extraction.py`, provide your database URL, username, and password. Neo4j can be launched locally (via Docker) or accessed remotely through the cloud.

Below are two example `construct` configuration blocks for connecting to a Neo4j database—choose one based on your setup, and replace placeholder values with actual credentials.

#### Neo4j Aura Example (Remote)  

```yaml
construct: # Required for knowledge graph construction  
  database: Neo4j  
  url: neo4j+s://<database-id>.databases.neo4j.io  
  username: neo4j  
  password: "<database_password>"
```

#### Neo4j Local Example (On-Device)  

```yaml
construct: # Required for knowledge graph construction  
  database: Neo4j  
  url: bolt://localhost:7687  # Default port is 7687  
  username: neo4j  
  password: "<database_password>"
```

---

## Customizations

### Model Selection 
OneKE by default supports the following model APIs:
1. **LocalServer** (e.g. LM Studio, Ollama, vLLM)
2. **OpenAI** 
3. **DeepSeek**

> **Note**: The chosen **LocalServer** *must* have an OpenAI-compatible API for seamless integration with OneKE. We used **LM Studio** in our experiments, but you can use any LocalServer that meets this requirement.

In addition to the default API's, OneKE also supports the following model categories from Hugging Face:
1. **LLaMA**
2. **Qwen**	
3. **ChatGLM**
4. **MiniCPM**
6. **DeepSeek-R1**  (Category will depend on distilled version)

> **Note**: OneKE's own model does *not* support Triple Extraction, so we did not include it in the list of supported models for our case. However, you can still use it for the other tasks OneKE supports such as: Named Entity Recognition (NER), Relation Extraction (RE) and Event Extraction (EE).

In `src/utils/document_extraction.py`, you can specify any Hugging Face model from the supported model categories listed above. Just update the `category` and `model_name_or_path` fields in the `model` portion of the configuration. If you choose a API-based model (e.g. LocalServer, OpenAI, DeepSeek), make sure to also include the necessary authentication credentials (e.g. API key) and base URL if applicable.

### Example Model Configuration Using LocalServer

```python
# For a LocalServer config we use an LM Studio server on our local network
config['model']['category'] = "LocalServer"
config['model']['model_name_or_path'] = "qwen/qwen3-4b-2507" # Use the API identifier for your LocalServer model
config['model']['api_key'] = os.getenv("LM_STUDIO_API_KEY") # API key may be required for authentication, depending on your setup
config['model']['base_url'] = os.getenv("LM_STUDIO_URL")
```

For safety, we recommend using environment variables to store sensitive information like API keys and database credentials. You can set these in your terminal or include them in a `.env` file (and ensure it’s added to `.gitignore` to prevent accidental commits).

### Example Model Configuration Using Hugging Face

```python
# For a Hugging Face config we use Qwen3-4B-Instruct-2507
config['model']['category'] = "Qwen"
config['model']['model_name_or_path'] = "Qwen/Qwen3-4B-Instruct-2507" # Use the Hugging Face model identifier, which typically includes the username and model name
```

## Running Market Foundry Pipeline  

After activating your environment or launching the Docker container, navigate to the root directory of the project and run the pipeline via `run.py` in the `src` folder.

We used **Qwen3-4B-Instruct-2507** from Hugging Face — an open-source, instruction-tuned model ideal for financial document understanding. This model excels at interpreting complex text and extracting relevant entities and relationships.

> ⚠️ **Important:** Some models on Hugging Face require authentication. To access them:  
> 1. Log in to your [Hugging Face](https://huggingface.co) account.  
> 2. Go to Settings → Access Tokens → Create a new token with *read* permissions.  
> 3. Run the following command and enter your token when prompted:

```bash
huggingface-cli login
```

Once authenticated, you’re ready to run the pipeline.

### Two Execution Methods  

1. **Run with a single file**  
   Use this if you want to process one specific document. Replace `<path_to_document>` with the actual path:  

   ```bash
   python src/run.py --file <path_to_document>
   ```

2. **Run with all files in a folder**  
   This option processes every file inside a directory, which is useful for bulk processing. However, it may take significantly longer—especially if you have many documents—since each one is processed sequentially.

   Use this method only if:  
   - You have a small number of files, or  
   - You're processing many smaller documents.

   Replace `<path_to_folder>` with your folder path:

   ```bash
   python src/run.py --folder <path_to_folder>
   ```

> ✅ **Note:** When the pipeline completes, you’ll see extracted knowledge printed in the terminal. The results will optionally be pushed to Neo4j and visualized in the **Explore** tab of your instance.

---

## Final Thoughts  
This workflow enables seamless transformation of unstructured financial documents into a rich, interconnected knowledge graph—powered by open-source tools, accessible models, and modular configuration. Whether using conda or Docker, users can replicate our pipeline with minimal friction and full control over model selection, environment setup, and database integration.
