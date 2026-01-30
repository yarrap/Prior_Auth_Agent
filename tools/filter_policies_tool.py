import json
from langchain_core.tools import tool

VALID_PLANS = {"ppo basic", "ppo gold", "hmo silver", "hmo platinum", "epo standard"}

@tool
def filter_policies_simple(intent: str, canonical_request: dict, all_policies: list) -> list:
    """
    Filter policies based only on:
    1️⃣ Intent (PriorAuth, Claim, Appeal)
    2️⃣ Plan type matches OR is 'All Plans'
    Only valid plans from the policy document are considered.
    """
    intent = intent.strip().lower()
    plan_type = canonical_request.get("plan_type", "").strip().lower()

    # Skip if plan is not valid
    if plan_type not in VALID_PLANS:
        print(f"Warning: plan '{plan_type}' is not in valid plans. Skipping filtering.")
        return []

    applicable_policies = []

    for policy in all_policies:
        policy_intent = policy.get("intent", "").strip().lower()
        policy_plan = policy.get("plan_type", "").strip().lower()

        # 1️⃣ Intent must match
        if policy_intent != intent:
            continue

        # 2️⃣ Plan type must match OR be global
        if policy_plan != "all plans" and policy_plan != plan_type:
            continue

        applicable_policies.append(policy)

    return applicable_policies
