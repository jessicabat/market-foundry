from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map="auto")


def safe_json_load(response):

    raw = response.strip()

    # Extract JSON region
    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end != -1:
        candidate = raw[start:end+1]
    else:
        candidate = raw

    # Remove incomplete trailing topic block (if cut mid-object)
    candidate = re.sub(
        r',\s*\{\s*"topic_name"[^}]*$',
        '',
        candidate
    )

    # Balance square brackets
    while candidate.count("[") > candidate.count("]"):
        candidate += "]"

    # Balance curly braces
    while candidate.count("{") > candidate.count("}"):
        candidate += "}"

    # Remove trailing commas
    candidate = candidate.replace(",]", "]")
    candidate = candidate.replace(",}", "}")

    return json.loads(candidate)


def normalize_structure(data):

    MAX_TOPIC_WORDS = 5
    MAX_ENTITY_WORDS = 3
    MAX_RELATION_WORDS = 3
    MAX_TOPICS = 5

    def word_count_ok(text, max_words):
        return len(text.split()) <= max_words

    if "topics" not in data:
        return data

    cleaned_topics = []

    for topic in data["topics"]:

        if "topic_name" not in topic:
            continue

        topic_name = topic["topic_name"]

        # Drop topic if topic name too long
        if not word_count_ok(topic_name, MAX_TOPIC_WORDS):
            continue

        # Filter entities (NO trimming)
        entities = [
            e for e in topic.get("entities", [])
            if isinstance(e, str) and word_count_ok(e, MAX_ENTITY_WORDS)
        ]

        # Filter relations (NO trimming)
        relations = [
            r for r in topic.get("relations", [])
            if isinstance(r, str) and word_count_ok(r, MAX_RELATION_WORDS)
        ]

        cleaned_topics.append({
            "topic_name": topic_name,
            "entities": list(set(entities)),
            "relations": list(set(relations))
        })

        # Stop at topic cap
        if len(cleaned_topics) >= MAX_TOPICS:
            break

    data["topics"] = cleaned_topics
    return data

def enforce_semantic_uniqueness(data):

    topic_names = set()
    entity_global = set()
    relation_global = set()

    # Collect topic names
    for topic in data.get("topics", []):
        topic_names.add(topic["topic_name"].lower())

    cleaned_topics = []

    for topic in data.get("topics", []):

        topic_name = topic["topic_name"]
        topic_lower = topic_name.lower()

        entities = []
        relations = []

        # -------- Entities --------
        for e in topic.get("entities", []):
            e_lower = e.lower()

            # Remove if duplicate of topic anywhere
            if e_lower in topic_names:
                continue

            if e_lower not in entity_global:
                entities.append(e)
                entity_global.add(e_lower)

        # -------- Relations --------
        for r in topic.get("relations", []):
            r_lower = r.lower()

            # Remove if overlaps entity or topic
            if r_lower in topic_names or r_lower in entity_global:
                continue

            if r_lower not in relation_global:
                relations.append(r)
                relation_global.add(r_lower)

        cleaned_topics.append({
            "topic_name": topic_name,
            "entities": entities,
            "relations": relations
        })

    data["topics"] = cleaned_topics
    return data


def extract_semantic_structure(document):

    messages = [
        {
            "role": "system",
            "content": """
You output ONLY valid JSON.
No explanation.
No markdown.
If unsure return {}.

You are extracting SEMANTIC TOPICS grounded in business / financial meaning.
Topics MUST be real semantic themes, NEVER placeholders like "Main Topic".
"""
        },
        {
            "role": "user",
            "content": f"""
Analyze the document and extract semantic topic structure.

GOAL:
Extract REAL semantic topics and map supporting entities and relations to each topic.

IMPORTANT CONSTRAINTS:
- Topics must be meaningful business themes
- NEVER output "Main Topic"
- Each topic must be reusable across documents
- Entities must be nouns or noun phrases
- Relations must be verbs or action phrases
- Prefer canonical financial / business wording
- Keep output flat: topic → entities → relations ONLY

TOKEN LENGTH RULES (CRITICAL):
- topic_name ≤ 5 words
- entities ≤ 3 words each
- relations ≤ 3 words each

If longer → compress to core semantic meaning.

OUTPUT FORMAT (STRICT):
{{
  "topics": [
    {{
      "topic_name": "Semantic topic name",
      "entities": ["entity1", "entity2"],
      "relations": ["relation1", "relation2"]
    }}
  ]
}}

RULES:
- Return ONLY valid JSON
- Do NOT add wrapper keys beyond "topics"
- Do NOT nest deeper than topics → topic_name/entities/relations
- Avoid duplicates
- Return at most 5 topics
- Merge similar ideas into one topic when possible

DOCUMENT:
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
        max_new_tokens=500,
        temperature=0.1
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    if not response:
        raise ValueError("Model returned empty response")

    parsed = safe_json_load(response)
    normalized = normalize_structure(parsed)
    deduped = enforce_semantic_uniqueness(normalized)

    return deduped


if __name__ == "__main__":

    doc = """
    \n\n\tMeta - Meta Announces Joint Venture with Funds Managed by Blue Owl Capital to Develop Hyperion Data Center\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nSkip to main content\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n  Meta Investor Relations\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nHome\nCompany InfoFinancials\nQuarterly EarningsAnnual ReportsSEC Filings\nAnnual MeetingPress ReleasesInvestor Events\nUpcoming EventsPast Events\nStock Info\nStock DataStock ChartHistorical Price Look UpInvestment CalculatorAnalyst Coverage\nResponsible Business PracticesLeadership & Governance\nGovernance DocumentsManagementBoard of DirectorsCommittee CompositionContact the Board\n\n\n\n\n\nHome\nCompany InfoFinancials\nQuarterly EarningsAnnual ReportsSEC Filings\nAnnual MeetingPress ReleasesInvestor Events\nUpcoming EventsPast Events\nStock Info\nStock DataStock ChartHistorical Price Look UpInvestment CalculatorAnalyst Coverage\nResponsible Business PracticesLeadership & Governance\nGovernance DocumentsManagementBoard of DirectorsCommittee CompositionContact the Board\n\n\n\n\nHome\nCompany InfoFinancials\nQuarterly EarningsAnnual ReportsSEC Filings\nAnnual MeetingPress ReleasesInvestor Events\nUpcoming EventsPast Events\nStock Info\nStock DataStock ChartHistorical Price Look UpInvestment CalculatorAnalyst Coverage\nResponsible Business PracticesLeadership & Governance\nGovernance DocumentsManagementBoard of DirectorsCommittee CompositionContact the Board\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nView all Press Releases\n\n\n\nMeta Announces Joint Venture with Funds Managed by Blue Owl Capital to Develop Hyperion Data Center\n\n\n\nOctober 21, 2025 \n\n\n\n\n\n\nDownload this Press Release\n(opens in new window)\n\n\n\n\n\n\nMeta announces joint venture with funds managed by Blue Owl Capital to finance the development and conduct the operations of the Hyperion data center campus in Richland Parish, Louisiana\n\n\nMENLO PARK, Calif., Oct. 21, 2025 /PRNewswire/ -- Meta Platforms, Inc. (NASDAQ: META) and funds managed by Blue Owl Capital entered into a joint venture agreement which will develop and own the Hyperion data center campus. Meta will provide construction management and property management services for the project.\n\n\n\n\n\n\n\nThis innovative partnership was designed to support the speed and flexibility required for Meta\'s data center projects and long-term AI ambitions. Meta has 15 years of experience developing, constructing and operating world class data center facilities. Blue Owl Capital complements this joint venture with its ability to deliver substantial capital at scale, along with deep expertise in digital infrastructure investment—enabling the rapid execution of mission-critical data center projects for hyperscalers.\nFunds managed by Blue Owl Capital will own an 80% interest in the joint venture, while Meta will retain the remaining 20% ownership. The parties have committed to fund their respective pro rata share of the approximately $27 billion in total development costs for the buildings and long-lived power, cooling, and connectivity infrastructure at the campus.\nIn connection with the creation of the joint venture, Meta contributed certain land and construction-in-progress assets relating to the campus development which were previously classified as held-for-sale. The funds managed by Blue Owl Capital made a cash contribution of approximately $7 billion to the joint venture, and Meta received a one-time distribution from the joint venture in the amount of approximately $3 billion.\n\nDoug Ostrover and Marc Lipschultz, Co-CEOs of Blue Owl Capital\xa0said, "We\'re proud that our funds are partnering with Meta on the development of the Hyperion data center campus—an ambitious project that reflects the scale and speed required to power the next generation of AI infrastructure. Blue Owl\'s ability to deliver substantial capital at scale, combined with our deep experience supporting hyperscalers, makes us uniquely positioned to help bring mission-critical digital infrastructure to life. We look forward to continuing our work with Meta and contributing to the long-term growth of the Richland Parish community."\n\nSusan Li, CFO, Meta said, "Our AI ambitions will be realized through our ability to deliver the infrastructure to support it. Our partnership with Blue Owl Capital to develop the Hyperion Data Center is a bold step forward—combining Meta\'s deep expertise in building and operating world-class data centers with Blue Owl\'s strength in infrastructure investment."\n\nRachel Peterson, VP, Data Centers, Meta, said, "We are proud to be part of the Richland Parish community and we look forward to continuing to strengthen our partnership for years to come. Construction is well underway with thousands of construction workers on site and, once online, the project will support over 500 operational jobs."\nMeta entered into operating lease agreements with the joint venture for use of all of the facilities of the campus once construction is complete. These lease agreements will have a four-year initial term with options to extend, providing Meta with long-term strategic flexibility.\nTo balance this optionality in a cost-efficient manner, Meta also provided the joint venture with a residual value guarantee for the first 16 years of operations whereby Meta would make a capped cash payment to the joint venture based on the then-current value of the campus if certain conditions are met following a non-renewal or termination of a lease.\nA portion of capital raised by Blue Owl will be funded by debt issued to PIMCO and select other bond investors through a private securities offering.\nMorgan Stanley & Co. LLC served as exclusive financial advisor to Meta in connection with this transaction and served as sole bookrunner in connection with the private securities offering. Latham & Wakins LLP served as legal counsel to Meta on the transaction and Eversheds Sutherland advised Meta on leasing matters. Arthur D. Little LLC acted as commercial due diligence advisor to Meta. Marsh provided Meta project risk analysis and insurance services. Arup provided technical and environmental independent engineer services to Meta. Kirkland & Ellis LLP served as legal counsel to the Blue Owl Capital funds on the transaction. Milbank LLP served as legal counsel to Morgan Stanley on the securities offering.\n\nAbout Meta\xa0Meta is building the future of human connection, powered by artificial intelligence and immersive technologies. When Facebook launched in 2004, it changed the way people connect. Apps like Messenger, Instagram, and WhatsApp further empowered billions around the world. Now, Meta is moving beyond 2D screens toward experiences that foster deeper connections and unlock new possibilities.\n\nForward-Looking StatementsThis press release contains forward-looking statements regarding the transaction and our business. These forward-looking statements are only predictions and may differ materially from actual results due to a variety of factors. Because some of these risks and uncertainties cannot be predicted or quantified and some are beyond our control, you should not rely on our forward-looking statements as predictions of future events. More information about potential risks and uncertainties that could affect our business and financial results is more fully detailed under the caption "Risk Factors" in our Quarterly Report on Form 10-Q filed with the Securities and Exchange Commission on July 31, 2025, which is available on our Investor Relations website at investor.atmeta.com and on the SEC website at www.sec.gov. In addition, please note that the date of this press release is October 21, 2025, and any forward-looking statements contained herein are based on assumptions that we believe to be reasonable as of this date. We undertake no obligation to update these statements as a result of new information or future events.\n\nContacts\xa0\nInvestors:Kenneth Dorellinvestor@meta.com /\u202finvestor.atmeta.com\xa0\nPress:Ashley Zandypress@meta.com\u202f/\u202fmeta.com/news\n\n View original content to download multimedia:https://www.prnewswire.com/news-releases/meta-announces-joint-venture-with-funds-managed-by-blue-owl-capital-to-develop-hyperion-data-center-302590584.html\nSOURCE Meta\n\n\nView all Press Releases\n\n\n\n\n\n\n\n\n\n\n\n\n\nLearn More\n\n\n\n\n\n\n\n\n\n\n\nAbout\nCreate a page\nCareers\nPrivacy\nTerms\nHelp\n\n\n\n\n\n\n\n        © Meta 2025\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
    """

    structure = extract_semantic_structure(doc)

    print("\nExtracted Semantic Structure:\n")
    print(json.dumps(structure, indent=2))
