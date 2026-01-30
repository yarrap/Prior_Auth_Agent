import json
import re
import os
from langchain_core.tools import tool


# Load model once (don't reload every time)
# Use Llama 3.2 1B instruct model as requested
MODEL_NAME = os.getenv("ENTITY_MODEL", "meta-llama/Llama-3.2-1b-Instruct")
generator = None

# Canonical plan names
CANONICAL_PLANS = [
    "PPO Basic",
    "PPO Gold",
    "HMO Silver",
    "HMO Platinum",
    "EPO Standard"
]

def load_model():
    """Lazily load the configured LLaMA model.

    This function will raise a clear exception if the model cannot be loaded,
    with a hint to install required packages (`transformers`, `accelerate`,
    and optionally `bitsandbytes`) and configure GPU if needed.
    """
    global generator
    if generator is None:
        try:
            from transformers import pipeline
            import torch
        except Exception as e:
            raise ImportError(
                "Required packages for model loading are missing.\n"
                "Run: pip install transformers accelerate safetensors sentencepiece"
            ) from e

        print("Loading LLaMA model... (first time only)")
        device = 0 if torch.cuda.is_available() else -1

        # For LLaMA-family instruct models use text-generation pipeline
        try:
            generator = pipeline(
                "text-generation",
                model=MODEL_NAME,
                tokenizer=MODEL_NAME,
                device=device,
                trust_remote_code=True,
            )
            print(f"Loaded model: {MODEL_NAME}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{MODEL_NAME}': {e}\n"
                "If you intended to run locally, install/upgrade: \n"
                "  pip install -U transformers accelerate bitsandbytes safetensors sentencepiece\n"
                "And ensure a compatible CUDA toolkit is available for GPU usage.\n"
                "Alternatively set ENTITY_MODEL to a local path or a smaller model."
            ) from e

def normalize_plan_name(plan_name: str) -> str:
    """Try to match LLM output to canonical plan names"""
    if not plan_name:
        return None
    plan_name_lower = plan_name.strip().lower()
    for canonical in CANONICAL_PLANS:
        canonical_lower = canonical.lower()
        # match if all words in plan_name exist in canonical
        if all(word in canonical_lower for word in plan_name_lower.split()):
            return canonical
    return plan_name  # fallback if no match





@tool
def extract_entities(raw_text: str) -> dict:
    """
    Clean messy OCR text and extract entities using Mistral 3B Instruct.
    Adds deterministic intent, authorization conditions, exclusions, and normalizes plan names.
    """
    load_model()

    # 1′️ Determine intent deterministically
    if re.search(r"Prior\s*Auth[:\s]*Y|Prior\s*Authorization|Prior\s*Auth\s*Required", raw_text, re.I):
        intent = "PriorAuth"
    elif re.search(r"\bAppeal\b", raw_text, re.I):
        intent = "Appeal"
    elif re.search(r"\bClaim\b", raw_text, re.I):
        intent = "Claim"
    else:
        intent = "PriorAuth"  # fallback default

    # 2′️ Identify authorization conditions
    authorization_conditions = []
    if re.search(r"peer[- ]?to[- ]?peer", raw_text, re.I):
        authorization_conditions.append("Peer-to-Peer Review Required")

    # 3′️ Identify coverage exclusions
    exclusions = []
    if re.search(r"Experimntl|Experimental|Non[- ]?FDA[- ]?Approved", raw_text, re.I):
        exclusions.append("Experimental Therapy")

    # 4′️ Use LLM to extract remaining structured info
    prompt = f"""You are a medical document parser. Extract structured JSON from messy medical text. Fix typos in patient names, procedure names, provider names, and all other fields if needed.

Extract as JSON with these fields:
- patient_name
- age (number)
- plan_type
- provider
- network_status (In-Network, Out-of-Network)
- procedure
- diagnosis
- service_type (Emergency, Elective, Routine, Oncology)
- cpt_code (5 digits)
- prior_auth_required (true if Y, Yes, Required)
- medical_necessity_review (true if Required, Reqrd, Y)
- procedure_category (Elective, Emergency, Urgent)

Text: {raw_text}
Return ONLY valid JSON, no explanation."""

    # Generate text from the model. Use explicit generation args compatible with
    # transformers; avoid mixing generation_config with kwargs.
    gen_result = generator(prompt, max_new_tokens=300, do_sample=False, return_full_text=False)

    # Extract text depending on pipeline return type
    if isinstance(gen_result, list) and gen_result:
        first = gen_result[0]
        if isinstance(first, dict):
            raw_output = first.get("generated_text") or first.get("text") or str(first)
        else:
            raw_output = str(first)
    elif isinstance(gen_result, dict):
        raw_output = gen_result.get("generated_text") or gen_result.get("text") or str(gen_result)
    else:
        raw_output = str(gen_result)

    # Try multiple parsing strategies to recover JSON
    # 1) Remove markdown fences and code blocks
    cleaned = re.sub(r"```json|```", "", raw_output, flags=re.IGNORECASE).strip()

    # 2) Find all {...} JSON-like blocks
    candidates = re.findall(r"\{[\s\S]*?\}", cleaned)
    parsed = None
    parsed_candidates = []

    import ast

    for candidate in candidates:
        # try normal JSON
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                parsed_candidates.append(obj)
                continue
        except Exception:
            pass

        # try ast.literal_eval for python-style dicts / single quotes
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                parsed_candidates.append(json.loads(json.dumps(obj)))
                continue
        except Exception:
            pass

    # pick the candidate with the most keys (most complete)
    if parsed_candidates:
        parsed = max(parsed_candidates, key=lambda d: len(d.keys()))

    if parsed is None:
        print("Could not parse LLM response as JSON. Falling back to deterministic extraction.")

        # Fallback: extract simple fields from the raw_text so downstream
        # filtering can operate even when the LLM didn't return strict JSON.
        plan_match = re.search(r"Plan[:\s]*([A-Za-z0-9 \-]+)", raw_text, re.I)
        plan_raw = plan_match.group(1).strip() if plan_match else ""

        prior_auth_flag, med_necess_flag = extract_prior_auth_flags(raw_text)

        plan_type_normalized = normalize_plan_name(plan_raw) if plan_raw else None

        # Construct a minimal canonical_request usable by the policy filter
        canonical = {
            "patient_name": None,
            "age": None,
            "plan_type": plan_type_normalized or plan_raw or "",
            "provider": None,
            "network_status": None,
            "procedure": None,
            "diagnosis": None,
            "service_type": None,
            "cpt_code": None,
            "prior_auth_required": prior_auth_flag,
            "medical_necessity_review": med_necess_flag,
            "coverage_exclusions": None,
            "notes": None
        }

        return {
            "intent": intent,
            "canonical_request": canonical,
            "policy_entities": {
                "procedure_name": None,
                "procedure_category": None,
                "service_type": None,
                "requires_prior_auth": prior_auth_flag,
                "medical_necessity_required": med_necess_flag,
                "provider": None,
                "network_status": None,
                "authorization_conditions": authorization_conditions,
                "exclusions": exclusions,
            },
            "raw_output": raw_output,
        }
    data = parsed

    # 5′️ Normalize plan name
    plan_type_normalized = normalize_plan_name(data.get("plan_type"))

    # 6′️ Construct final structured output
    return {
        "intent": intent,
        "canonical_request": {
            "patient_name": data.get("patient_name"),
            "age": data.get("age"),
            "plan_type": plan_type_normalized,
            "provider": data.get("provider"),
            "network_status": data.get("network_status"),
            "procedure": data.get("procedure"),
            "diagnosis": data.get("diagnosis"),
            "service_type": data.get("service_type"),
            "cpt_code": data.get("cpt_code"),
            "prior_auth_required": data.get("prior_auth_required", False),
            "medical_necessity_review": data.get("medical_necessity_review", False),
            "coverage_exclusions": data.get("coverage_exclusions"),
            "notes": data.get("notes")
        },
        "policy_entities": {
            "procedure_name": data.get("procedure"),
            "procedure_category": data.get("procedure_category"),
            "service_type": data.get("service_type"),
            "requires_prior_auth": data.get("prior_auth_required", False),
            "medical_necessity_required": data.get("medical_necessity_review", False),
            "provider": data.get("provider"),
            "network_status": data.get("network_status"),
            "authorization_conditions": authorization_conditions,
            "exclusions": exclusions
        }
    }



