from sympy import content

from transformers import AutoTokenizer, AutoModelForCausalLM
from topic_extractor import load_model_hf, load_model_openai
from openai import OpenAI
import torch
import gc
import json
import yaml
import os
import copy
from dotenv import load_dotenv

load_dotenv()

# MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL,
#     torch_dtype=torch.float32,
#     device_map="auto"
# )

# MODEL = "lfm2-8b-a1b"
# client = OpenAI(
#     api_key=os.getenv("LM_STUDIO_API_KEY"),
#     base_url=os.getenv("LM_STUDIO_NETWORK_URL")
# )

BASE_YAML_TEMPLATE = {
    "model": {
        "category": "Qwen",
        "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "api_key": "",
        "base_url": "",
        "device": "auto"
    },
    "extraction": {
        "task": "Triple",
        "constraint": [[], []],
        "use_file": False,
        "file_path": "",
        "mode": "customized",
        "update_case": False,
        "show_trajectory": False
    }
}


def safe_json_parse(text):

    # Try direct parse first
    try:
        print("\nObserved Topics:\n", text)
        json_response = json.loads(text)
        return json_response
    except:
        print("Failed to parse Extracted Topics as JSON")

    # Attempt to extract FIRST valid JSON object
    brace_stack = 0
    start_idx = None

    for i, ch in enumerate(text):

        if ch == "{":
            if start_idx is None:
                start_idx = i
            brace_stack += 1

        elif ch == "}":
            brace_stack -= 1

            if brace_stack == 0 and start_idx is not None:
                candidate = text[start_idx:i+1]

                try:
                    return json.loads(candidate)
                except:
                    start_idx = None
                    brace_stack = 0
                    continue

    print("⚠️ Could not parse JSON — returning empty fallback")
    return {"yaml_files": []}

def generate_yaml_configs_openai(document_type, topic_map):
    MODEL, client = load_model_openai()

    messages = [
    {
        "role": "system",
        "content": """
        You are a structured JSON generator.

        CRITICAL RULES:

        - Output ONLY valid JSON.
        - NEVER output explanations.
        - NEVER output markdown.
        - NEVER output text before or after JSON.
        - If unsure, output:

        {"yaml_files":[]}
        """
            },
            {
                "role": "user",
                "content": f"""
        You are designing knowledge extraction constraints for a knowledge graph.

        INPUTS:

        Document Type:
        {document_type}

        Topic Map:
        {json.dumps(topic_map, indent=2)}

        TASK:

        Generate YAML extraction plans grounded in the topic map.

        OUTPUT FORMAT:

        {{
        "yaml_files": [
            {{
            "file_name": "string.yaml",
            "instruction_focus": "short semantic focus",
            "entity_types": ["entity1","entity2"],
            "relation_types": ["relation1","relation2"]
            }}
        ]
        }}

        RULES (CRITICAL):

        - Generate EXACTLY 1 YAML file.
        - The YAML must cover ALL main topics from the topic map.

        ONTOLOGY DISCIPLINE RULES:

        - Entity types MUST represent stable real-world classes.
        - Entity types must be literal ontological categories (e.g., "Company", "Product", "Technology", "Stock", "Index", "Financial Metric", "Event").
        - Do NOT create abstract interpretive categories (e.g., "Corporate Strategy", "Market Reaction", "Stock Performance", "Technical Capability", "Investment Entity").
        - Do NOT convert roles or behaviors into entity types.
        - Do NOT encode outcomes, sentiments, reactions, or states as entity types.
        - Entity types must be reusable across many documents in the same domain.

        RELATION DISCIPLINE RULES:

        - Relation types must describe observable relationships stated in text.
        - Do NOT generate interpretive or inferred causal relations unless explicitly stated in text.
        - Do NOT create meta-level reporting relations (e.g., "claims", "states", "asserts") unless speech attribution is a core topic.
        - Prefer concrete semantic relations (e.g., "acquires", "partners_with", "drops_by", "uses", "competes_with").
        - Merge semantically similar relations into one reusable category.

        SCOPE CONTROL RULES:

        - Avoid fabricating economic interpretation (e.g., "market sentiment shift", "pricing power erosion") unless explicitly described.
        - Avoid encoding analysis, inference, or narrative framing.
        - Focus strictly on extractable factual structure.
        
        DOMAIN ADAPTATION RULE:

        - Entity and relation types must reflect the primary domain of the document.
        - Do not reuse finance-oriented types unless the document is about finance.
        - Infer the domain from the topic map before generating types.

        TYPE COUNT RULES:

        - Create 10–15 entity types maximum.
        - Create 10–15 relation types maximum.
        - Types must be reusable across multiple triples.
        - Avoid narrow variants of the same concept.

        file_name MUST be:
        {document_type}.yaml

        Return ONLY valid JSON.
        """
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    content = response.choices[0].message.content.strip()
    # print("\nRAW MODEL RESPONSE:\n", content)
    return safe_json_parse(content)

def generate_yaml_configs(document_type, topic_map):

    messages = [
    {
        "role": "system",
        "content": """
        You are a structured JSON generator.

        CRITICAL RULES:

        - Output ONLY valid JSON.
        - NEVER output explanations.
        - NEVER output markdown.
        - NEVER output text before or after JSON.
        - If unsure, output:

        {"yaml_files":[]}
        """
            },
            {
                "role": "user",
                "content": f"""
        You are designing knowledge extraction constraints for a knowledge graph.

        INPUTS:

        Document Type:
        {document_type}

        Topic Map:
        {json.dumps(topic_map, indent=2)}

        TASK:

        Generate YAML extraction plans grounded in the topic map.

        OUTPUT FORMAT:

        {{
        "yaml_files": [
            {{
            "file_name": "string.yaml",
            "instruction_focus": "short semantic focus",
            "entities": ["entity1","entity2"],
            "relations": ["relation1","relation2"]
            }}
        ]
        }}

        RULES (CRITICAL):

        - Generate EXACTLY 3 YAML files.
        - Each YAML must focus on ONE main topic from the topic map.
        - file_name MUST be unique:
        {document_type}_focus_1.yaml
        {document_type}_focus_2.yaml
        {document_type}_focus_3.yaml

        ENTITY GENERATION PROCESS (FOLLOW STRICTLY):

        Step 1:
        Identify the CORE concept of the topic.

        Step 2:
        List related BUSINESS OBJECTS discussed under this concept.
        Examples of object types:
        - financial metrics
        - business operations
        - costs
        - forecasts
        - stakeholders
        - investments
        - risks

        Step 3:
        Convert those objects into canonical nouns.

        Step 4:
        Return 12–18 DISTINCT entities.

        IMPORTANT:
        - Entities must be DIFFERENT concepts, not paraphrases.
        - Do NOT repeat the topic words directly.
        - Expand outward from the topic into related domain concepts.

        Return ONLY valid JSON.
        """
        }
    ]


    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1200,
        temperature=0.2
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # print("\nRAW MODEL RESPONSE:\n", response)

    return safe_json_parse(response)


def write_yaml_files(config_json, output_dir="generated_yamls", input_file_path=""):

    os.makedirs(output_dir, exist_ok=True)

    seen_files = set()

    for i, config in enumerate(config_json.get("yaml_files", [])[:3]):

        yaml_obj = copy.deepcopy(BASE_YAML_TEMPLATE)

        entities = config.get("entity_types", [])
        relations = config.get("relation_types", [])

        yaml_obj["extraction"]["constraint"] = [entities, relations]
        yaml_obj["extraction"]["file_path"] = input_file_path

        file_name = config.get("file_name", f"generated_{i}.yaml")

        if file_name in seen_files:
            file_name = file_name.replace(".yaml", f"_{i}.yaml")

        seen_files.add(file_name)

        file_path = os.path.join(output_dir, file_name)

        # ---------- CUSTOM YAML WRITE ----------
        with open(file_path, "w") as f:

            f.write("model:\n")
            f.write("  category: Qwen\n")
            f.write("  model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct\n")
            f.write("  api_key: \"\"\n")
            f.write("  base_url: \"\"\n")
            f.write("  device: auto\n\n")

            f.write("extraction:\n")
            f.write("  task: Triple\n")

            # INLINE constraint formatting
            f.write("  constraint: [\n")
            f.write("  " + json.dumps(entities) + ",\n")
            f.write("  " + json.dumps(relations) + ",\n")
            f.write("  ]\n")

            f.write("  use_file: false\n")
            f.write(f"  file_path: {input_file_path}\n")
            f.write("  mode: customized\n")
            f.write("  update_case: false\n")
            f.write("  show_trajectory: false\n")

        print(f"Created YAML → {file_path}")

#testing 
if __name__ == "__main__":

    topic_map = {
        "Revenue Growth": ["Enterprise AI Adoption", "Infrastructure Costs"],
        "Operating Margins": ["Infrastructure Costs"],
        "Growth Expectations": ["Continued Growth"]
    }

    document_type = "earnings_call_transcript"

    configs = generate_yaml_configs(document_type, topic_map)

    write_yaml_files(
        configs,
        output_dir="generated_yamls",
        input_file_path="example_document.pdf"
    )