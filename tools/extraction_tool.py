import csv
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM # Added AutoTokenizer, AutoModelForCausalLM
import torch 
from langchain_core.tools import tool
import re


llm_eval_pipeline = None 

def initialize_llm_eval_pipeline(model_name):
    global llm_eval_pipeline
    if llm_eval_pipeline is None:
        print(f"Loading LLM for evaluation: {model_name} ... (first time only)")
        try:
            llm_eval_pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,  # use GPU if available
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            print("LLM for evaluation loaded!")
        except Exception as e:
            print(f"Error loading LLM for evaluation: {e}")
            print(f"Please check if the model name '{model_name}' is correct and accessible on Hugging Face.")
            raise



def extract_json_safely(text: str):
    """
    Extract the first valid JSON object from LLM output.
    """
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def normalize_llm_output(parsed_json: dict):
    relevance = parsed_json.get("relevance", "NOT_RELEVANT").upper()
    judgement = parsed_json.get("judgement", "NOT_APPLICABLE").upper()
    explanation = parsed_json.get("explanation", "")

    if relevance not in {"RELEVANT", "NOT_RELEVANT"}:
        relevance = "NOT_RELEVANT"

    if relevance == "RELEVANT":
        if judgement not in {"ACCEPT", "REJECT", "PENDING"}:
            judgement = "REJECT"
            explanation += " (Judgement defaulted due to invalid LLM output)"
    elif relevance == "NOT_RELEVANT":
        judgement = "NOT_APPLICABLE"

    return relevance, judgement, explanation


def extract_prior_auth_flags(raw_text: str):
    text_lower = raw_text.lower()

    prior_auth = bool(re.search(r"prior (?:auth|authorization)\s*[:#]?\s*(y|yes|required)", text_lower))
    med_necess = bool(re.search(r"med(?:ical)? necess(?:ity)? review\s*[:#]?\s*(y|yes|required)", text_lower))

    return prior_auth, med_necess


@tool
def evaluate_with_mistral_and_export_csv(
    model_output,
    filtered_policies,
    csv_path,
    model_name="meta-llama/Llama-3.2-1b-Instruct"
):
    """
    Evaluate policy relevance using Mistral and enforce prior_auth / med_necess flags.
    Export results as CSV.
    """
    global llm_eval_pipeline

    initialize_llm_eval_pipeline(model_name)

    if not filtered_policies:
        print("⚠️ No policies provided to LLM evaluation.")

    canonical_request = model_output["canonical_request"]

    # Enforce prior auth / med necessity from raw text
    raw_text = model_output.get("raw_text", "")
    prior_auth_flag, med_necess_flag = extract_prior_auth_flags(raw_text)
    canonical_request["prior_auth_required"] = prior_auth_flag
    canonical_request["medical_necessity_review"] = med_necess_flag

    results = []

    for policy in filtered_policies:
        policy_text = policy["rule_text"]

        prompt =  f"""
You are a healthcare prior authorization reviewer with knowledge of oncology and chemotherapy workflows.

Patient request:
{json.dumps(canonical_request, indent=2)}

Policy rule:
{policy_text}

Instructions:

1️⃣ Determine relevance:
- RELEVANT only if policy explicitly applies to:
    - Oncology treatments
    - Chemotherapy sessions
    - Prior authorization requirements
    - Peer-to-peer review for oncology/chemo
    - Medical necessity or treatment exclusions relevant to chemotherapy
- NOT_RELEVANT otherwise
- Do not mark NOT_RELEVANT just because diagnosis or notes are missing

2️⃣ Determine judgement:
- If relevance is NOT_RELEVANT → judgement MUST be NOT_APPLICABLE
- If relevance is RELEVANT → judgement MUST be ACCEPT or REJECT
- Mention missing fields in explanation

3️⃣ Provide explanation linking policy to patient request

Return STRICT JSON only:
{{
  "relevance": "RELEVANT" or "NOT_RELEVANT",
  "judgement": "ACCEPT" | "REJECT" | "NOT_APPLICABLE",
  "explanation": "short reasoning referencing the policy"
}}
"""

        output = llm_eval_pipeline(
            prompt,
            max_new_tokens=250,
            do_sample=False
        )[0]["generated_text"]

        generated_text = output.split(prompt)[-1].strip()

        # 🔍 DEBUG
        print("\n================ RAW LLM OUTPUT ================\n")
        print(generated_text)
        print("\n================================================\n")

        parsed = extract_json_safely(generated_text)

        if parsed is None:
            relevance = "NOT_RELEVANT"
            judgement = "NOT_APPLICABLE"
            explanation = "Invalid or missing JSON returned by LLM"
        else:
            relevance, judgement, explanation = normalize_llm_output(parsed)

        results.append({
            "policy_id": policy["policy_id"],
            "rule_text": policy_text,
            "relevance": relevance,
            "judgement": judgement,
            "explanation": explanation
        })

    # --------------------------------------------------
    # CSV EXPORT
    # --------------------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["policy_id", "rule_text", "relevance", "judgement", "explanation"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ CSV exported: {csv_path}")
    return results
