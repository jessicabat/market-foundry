import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import torch
import json
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_model_hf():
    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model
    
def load_model_openai():
    with open(os.path.join(os.path.dirname(__file__), "utils", "extraction_config.yaml")) as f:
        config = yaml.safe_load(f)
    model_info = config.get("model", {})
    
    MODEL = model_info.get("model_name_or_path", "")
    
    client = OpenAI(
        api_key=os.getenv("LM_STUDIO_API_KEY"),
        base_url=os.getenv("LM_STUDIO_LOCAL_URL"),
    )
    return MODEL, client

def extract_topics_openai(document):
    MODEL, client = load_model_openai()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                You are a strict JSON generator.
                
                You must output ONLY valid JSON.
                If any non-JSON text is generated, the program will crash.
                
                You MUST ensure that:
                - EVERY [ has a matching ]
                - EVERY { has a matching }
                - The JSON must be syntactically valid and parseable.
                If it is not valid JSON, regenerate it completely.

                Do not summarize.
                Do not explain.
                Do not format.
                Do not use markdown.
                Do not use headings.
                Do not use bullet points.

                TASK:
                Analyze the document and identify ALL major topics discussed.

                For each major topic, identify related subtopics.

                OUTPUT FORMAT (flat dictionary only):

                {
                "Main Topics": ["Topic 1", "Topic 2"],
                "Topic 1": ["Subtopic A", "Subtopic B"],
                "Topic 2": ["Subtopic C"]
                }

                RULES:
                - Topics must represent major themes
                - Subtopics must represent supporting ideas
                - No nested dictionaries
                - No wrapper keys
                - Use natural topic names
                """
            },
            {
                "role": "user",
                "content": f"""
                Document:
                {document}
                """
            }
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    content = response.choices[0].message.content.strip()

    if not content:
        raise ValueError("Model returned empty response")
    
    # Debugging in case of JSON parsing issues
    # print("\nObserved Topics:\n", repr(content))

    # Remove markdown fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    return json.loads(content)


def extract_topics(document):
    tokenizer, model = load_model_hf()
    messages=[
            {
                "role": "system",
                "content": """
                You are a strict JSON generator.
                
                You must output ONLY valid JSON.
                If any non-JSON text is generated, the program will crash.
                
                You MUST ensure that:
                - EVERY [ has a matching ]
                - EVERY { has a matching }
                - The JSON must be syntactically valid and parseable.
                If it is not valid JSON, regenerate it completely.

                Do not summarize.
                Do not explain.
                Do not format.
                Do not use markdown.
                Do not use headings.
                Do not use bullet points.

                TASK:
                Analyze the document and identify ALL major topics discussed.

                For each major topic, identify related subtopics.

                OUTPUT FORMAT (flat dictionary only):

                {
                "Main Topics": ["Topic 1", "Topic 2"],
                "Topic 1": ["Subtopic A", "Subtopic B"],
                "Topic 2": ["Subtopic C"]
                }

                RULES:
                - Topics must represent major themes
                - Subtopics must represent supporting ideas
                - No nested dictionaries
                - No wrapper keys
                - Use natural topic names
                """
            },
            {
                "role": "user",
                "content": f"""
                Document:
                {document}
                """
            }
        ]
        
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        max_length=8192,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    if not response:
        raise ValueError("Model returned empty response")
    
    del model
    del tokenizer
    gc.collect()

    # return json.loads(response)
    print("\nObserved Topics:\n", response)
    try:
        response = response.strip()

        # Remove markdown fences if present
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()

        return json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("Model response is not valid JSON")

if __name__ == "__main__":
    doc = """
    The company reported strong revenue growth driven by enterprise AI adoption.
    Operating margins declined due to higher infrastructure costs.
    Management expects continued growth next year but warns of macroeconomic risks.
    """
    topics = extract_topics(doc)
    print("\nExtracted Topics:\n")
    print(json.dumps(topics, indent=2))