from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    )


def extract_topics(document):
    messages = [
        {
            "role": "system",
            "content": "You output ONLY valid JSON. No explanation. No markdown."
        },
        {
            "role": "user",
            "content": f"""
                        Analyze the document and identify ALL major topics discussed.

                        For each major topic, identify related subtopics or supporting ideas.

                        OUTPUT REQUIREMENTS:

                        Return ONLY valid JSON.
                        Return ONLY a flat dictionary.

                        FORMAT:
                        {{
                        "Main Topics": ["Topic 1", "Topic 2", ...],
                        "Topic 1": ["Subtopic A", "Subtopic B", ...],
                        "Topic 2": ["Subtopic C", "Subtopic D", ...],
                        ...
                        }}

                        RULES:
                        - Topics should represent major themes in the document
                        - Subtopics should represent supporting ideas, drivers, risks, or related concepts
                        - Do NOT restrict to financial metrics only
                        - Do NOT create wrapper keys like "Topics" or "Subtopics"
                        - Do NOT create nested dictionaries
                        - Use natural, human-readable topic names
                        - Prefer broader parent topics over highly similar ones.

                        Focus on capturing the full semantic coverage of the document.

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