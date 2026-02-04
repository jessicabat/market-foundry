from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import yaml
import os
import copy

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map="auto")


BASE_YAML_TEMPLATE = {
    "model": {
        "category": "Qwen",
        "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "device": "auto"
    },
    "extraction": {
        "task": "Triple",
        "instruction": "",
        "constraint": [],
        "use_file": True,
        "file_path": "",
        "mode": "customized"
    }
}


def generate_yaml_configs(document_type, topic_map):

    messages = [
        {
            "role": "system",
            "content": """
You are a structured configuration generator.

You MUST output valid JSON.
You MUST NOT output explanations.
You MUST NOT output markdown.
If unsure, output {"yaml_files": []}.
"""
        },
        {
            "role": "user",
            "content": f"""
You are designing knowledge extraction YAML configurations.

INPUTS:
Document Type: {document_type}

Topics:
{json.dumps(topic_map, indent=2)}

TASK:
Design multiple YAML extraction configurations optimized for knowledge graph extraction.

OUTPUT FORMAT:
{{
  "yaml_files": [
    {{
      "file_name": "string.yaml",
      "instruction_focus": "What this YAML focuses on extracting",
      "entity_constraints": ["Entity_Type_1", "Entity_Type_2"],
      "relation_constraints": ["Relation_1", "Relation_2"]
    }}
  ]
}}

RULES:
- Generate 2 to 4 YAML configs
- Each config must focus on a different semantic extraction angle
- Consider document structure + topic coverage
- Think like a knowledge graph engineer
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
        max_new_tokens=1600,
        temperature=0.2
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    print("\nRAW YAML MODEL RESPONSE:")
    print("------------------------------------------------")
    print(response)
    print("------------------------------------------------")

    if not response:
        raise ValueError("Model returned empty response")

    return json.loads(response)


def write_yaml_files(config_json, output_dir="generated_yamls", input_file_path=""):

    os.makedirs(output_dir, exist_ok=True)

    for config in config_json.get("yaml_files", []):

        yaml_obj = copy.deepcopy(BASE_YAML_TEMPLATE)

        yaml_obj["extraction"]["instruction"] = config.get(
            "instruction_focus",
            "Extract factual triples grounded in the text."
        )

        yaml_obj["extraction"]["constraint"] = [
            config.get("entity_constraints", []),
            config.get("relation_constraints", [])
        ]

        yaml_obj["extraction"]["file_path"] = input_file_path

        file_name = config.get("file_name", "generated_config.yaml")
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w") as f:
            yaml.dump(yaml_obj, f, sort_keys=False)

        print(f"✅ Created YAML → {file_path}")


if __name__ == "__main__":

    topic_map = {
        "Revenue Growth": ["Enterprise AI Adoption", "Infrastructure Costs"],
        "Risk Factors": ["Macroeconomic Risks"]
    }

    document_type = "earnings_call_transcript"

    configs = generate_yaml_configs(document_type, topic_map)

    write_yaml_files(
        configs,
        output_dir="generated_yamls",
        input_file_path="example_document.pdf"
    )
