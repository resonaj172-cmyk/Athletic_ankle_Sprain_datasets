import os
import json
import requests
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

SYSTEM_PROMPT = (
    "You are the assistant inside an athlete fatigue monitoring app. "
    "Answer only from the context you are given. Use simple language. "
    "Fatigue is predicted as Low, Moderate or High by the selected model from the seven values of the current session; "
    "the model name and its accuracy are given in the context. "
    "Previous-session workload is shown for context and is not a model input. "
    "Sessions are ordered by session number, not dates. "
    "Never give medical advice or diagnoses. "
    "If the context does not contain the answer, say you do not have that information. "
    "Keep answers short."
)

MEDICAL_KEYWORDS = [
    "diagnose", "diagnosis", "injury", "hurt", "pain", "torn", "fracture",
    "sprain", "swollen", "treatment", "medicine", "pill", "doctor", "cure", "medical"
]


def generate_template_response(prompt: str, context: Dict[str, Any]) -> str:
    """Provides dynamic, deterministic template answers when no LLM key is configured."""
    p_lower = prompt.lower().strip()

    # Safety check for medical advice
    if any(k in p_lower for k in MEDICAL_KEYWORDS):
        return (
            "I cannot give medical advice or clinical diagnoses. "
            "If you are experiencing pain, persistent fatigue, or symptoms of injury, "
            "please consult a qualified doctor, physiotherapist, or sports medicine professional."
        )

    latest = context.get("latest_session")
    expl = context.get("latest_explanation")
    workload = context.get("workload_context")
    model_name = context.get("model_name", "Selected Model")
    acc_pct = context.get("accuracy_pct", 0.0)

    # Question 1: Why is fatigue predicted High / Moderate / Low?
    if "why" in p_lower and ("fatigue" in p_lower or "prediction" in p_lower or "predicted" in p_lower):
        if not latest or not latest.get("predicted_class"):
            return (
                "You currently do not have a completed prediction (predictions begin from session 15 onward). "
                "Once you log session 15, the model will break down the top factors influencing your result."
            )
        p_class = latest["predicted_class"]
        prob = latest.get(f"prob_{p_class.lower()}", 0.0)
        prob_str = f"{prob * 100:.1f}%" if prob is not None else ""

        response = f"Your latest session (#{latest['session_number']}) was predicted as **{p_class}** ({prob_str} confidence) by {model_name}.\n\n"
        if expl and expl.get("top_towards"):
            top_factors = [f"- **{f['friendly_name']}** ({f['actual_value']}) pushed the prediction toward {p_class}" for f in expl["top_towards"][:3]]
            response += "Key factors contributing to this prediction:\n" + "\n".join(top_factors) + "\n\n"
            if expl.get("importance_sentence"):
                response += expl["importance_sentence"]
        return response

    # Question 2: Workload meaning (7-session and 14-session)
    if "workload" in p_lower or "7-session" in p_lower or "14-session" in p_lower:
        if not workload or not workload.get("has_window_7"):
            return (
                "Workload context compares your current session's training load against the average of your previous 7 and 14 sessions. "
                "This requires at least 7 earlier sessions of history to calculate. "
                "Note: Workload is displayed purely for your context and is not fed into the prediction model."
            )
        l_7 = workload.get("load_mean_7")
        l_14 = workload.get("load_mean_14")
        ratio_7 = workload.get("load_ratio_7")

        response = (
            f"**Your Workload Context:**\n"
            f"- Previous 7-session average load: **{l_7:.1f}**\n"
            f"- Previous 14-session average load: **{l_14:.1f}**\n"
        )
        if ratio_7 is not None:
            response += f"- Today's session load is **{ratio_7:.2f}x** your 7-session rolling average.\n\n"
        response += "This metric helps you track training volume fluctuations. Remember, the machine learning model evaluates only today's 7 values."
        return response

    # Question 3: Model accuracy
    if "accura" in p_lower or "model" in p_lower and ("good" in p_lower or "trust" in p_lower or "metrics" in p_lower):
        return (
            f"The active model is **{model_name}** with a held-out test accuracy of **{acc_pct}%** "
            f"(Macro F1: {context.get('macro_f1', 0.0)}).\n\n"
            f"This accuracy was established on an 80/20 test split using the 7 post-session features. "
            f"Predictions serve as an operational monitoring tool, not a clinical diagnosis."
        )

    # Question 4: Recovery trend
    if "recovery" in p_lower or "trend" in p_lower or "history" in p_lower:
        history = context.get("sessions_history", [])
        if not history:
            return "No previous session records found."
        rec_vals = [f"Session {s['session_number']}: {s['recovery_score']}" for s in history[-5:] if s.get("recovery_score") is not None]
        return (
            f"Recent recovery scores (last {len(rec_vals)} sessions):\n"
            + "\n".join(rec_vals) +
            "\n\nRecovery score reflects your baseline readiness prior to each training session."
        )

    # Default fallback
    return (
        "AI service not configured with an external API key. "
        "You can click on any of the starter questions above to explore your session analytics."
    )


def ask_assistant(prompt: str, context: Dict[str, Any]) -> str:
    """Routes user query to configured LLM or fallback template handler."""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if not api_key:
        return generate_template_response(prompt, context)

    # If API key is present, invoke external LLM with strict context
    context_str = json.dumps(context, indent=2, default=str)
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext Data:\n{context_str}"},
        {"role": "user", "content": prompt}
    ]

    try:
        if provider == "openai" or "gpt" in os.getenv("LLM_MODEL", "").lower():
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 400
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        # Fallback cleanly on network/API failure
        pass

    return generate_template_response(prompt, context)
