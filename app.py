import uuid
from analytics import init_db, log_turn
from flask import Flask, render_template, request, session, redirect
from flask_session import Session
import ollama
import os
import re
import numpy as np
from rag_embeddings import build_index, search, model as embedding_model
from rag import load_knowledge
import markdown as md

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-key")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./.flask_sessions"

Session(app) 
init_db()
client = ollama.Client()

MODEL = "llama3.2:3b"

chunks = load_knowledge()
faiss_index, embeddings, chunks = build_index(chunks)

CONCERN_EMBEDDINGS = {
    "pain": embedding_model.encode(
        "nipple pain soreness cracking bleeding burning during breastfeeding",
        normalize_embeddings=True
    ),
    "latch": embedding_model.encode(
        "baby not latching trouble latching poor latch difficulty attaching to breast",
        normalize_embeddings=True
    ),
    "supply": embedding_model.encode(
        "low milk supply not enough milk baby still hungry after feeding",
        normalize_embeddings=True
    ),
    "stress": embedding_model.encode(
        "overwhelmed exhausted crying giving up failing anxious stressed can't do this",
        normalize_embeddings=True
    ),
    "red_flag": embedding_model.encode(
        "fever mastitis abscess baby not feeding at all dehydration no wet diapers blood severe infection emergency dry mouth sunken fontanelle lethargic jaundice not waking to feed",
        normalize_embeddings=True
    ),
    "concern": embedding_model.encode(
        "worried unsure milk supply questions feeding often cluster feeding reassurance",
        normalize_embeddings=True
    ),
}

THRESHOLDS = {
    "pain": 0.15,
    "latch": 0.15,
    "supply": 0.15,
    "stress": 0.25,    
    "red_flag": 0.42,  
    "concern": 0.20,
}

# --- NEW: off-topic detection ---------------------------------------------
# Keywords that indicate the query is outside the breastfeeding/lactation
# support domain entirely (general healthcare logistics, unrelated life
# topics, etc). Kept intentionally broad since a false positive here just
# means falling back to a polite redirect rather than a mis-triaged clinical
# workflow.
OFF_TOPIC_KEYWORDS = [
    "hospital", "hospitals", "childcare", "child care", "daycare", "day care",
    "insurance", "pediatrician recommendation", "recommend a doctor",
    "school", "custody", "lawyer", "immigration", "housing", "job", "career",
]

# Minimum retrieval similarity required to consider a query "in-domain".
# If nothing in the knowledge base is even loosely relevant, that's a strong
# signal this isn't a breastfeeding/lactation question at all.
RETRIEVAL_OFF_TOPIC_THRESHOLD = 0.35


def is_off_topic(user_input, retrieval_scores_list):
    text = user_input.lower()
    if any(kw in text for kw in OFF_TOPIC_KEYWORDS):
        return True
    if retrieval_scores_list and max(retrieval_scores_list) < RETRIEVAL_OFF_TOPIC_THRESHOLD:
        return True
    return False
# ---------------------------------------------------------------------------


def clean_output(text):
    return md.markdown(text)


def build_conversation_text(chat_history):
    text = ""
    for msg in chat_history:
        text += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return text


def check_hard_medical_red_flags(user_input, chat_history):
    combined_text = user_input.lower()
    for msg in chat_history:
        combined_text += " " + msg["content"].lower()

    infant_danger_words = [
        "fever", "has a temp", "high temperature", "not drinking at all", 
        "won't eat", "refusing to feed", "refusing to eat", "hasn't fed at all"
    ]
    baby_words = ["baby", "infant", "newborn", "he", "she", "they"]
    if (
        any(d in combined_text for d in infant_danger_words)
        and any(b in combined_text for b in baby_words)
        and not (
            any(inf in combined_text for inf in ["mastitis", "streaks", "chills", "flu-like"])
            and any(br in combined_text for br in ["breast", "nipple", "boob"])
        )
    ):
        return "INFANT_CRISIS"

    diaper_pattern = r"\b(1|2|3|4|one|two|three|four)\b\s*(wet)?\s*diaper"
    feed_pattern = r"\b(1|2|3|4|one|two|three|four)\b\s*(feeds|feedings|times)\s*(a|per)?\s*day"
    
    if "more than" not in combined_text and "at least" not in combined_text:
        if re.search(diaper_pattern, combined_text) or re.search(feed_pattern, combined_text):
            return "LOW_COUNTS_URGENT"

    infection_words = ["fever", "mastitis", "101", "102", "103", "104", "chills", "streaks", "flu-like"]
    breast_words = ["breast", "nipple", "boob"]
    if any(inf in combined_text for inf in infection_words) and any(br in combined_text for br in breast_words):
        return "MATERNAL_URGENT"

    return None


def score_text_concern(user_input, chat_history):
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

    full_context_text = user_input
    for msg in recent_history:
        full_context_text += " " + msg["content"]

    context_vec = embedding_model.encode(
        full_context_text,
        normalize_embeddings=True
    )

    raw_scores = {}
    flags = {}

    for category, concern_vec in CONCERN_EMBEDDINGS.items():
        similarity = float(np.dot(context_vec, concern_vec))
        raw_scores[category] = similarity
        flags[category] = 1 if similarity >= THRESHOLDS[category] else 0

    return raw_scores, flags


def detect_detail_level(text):
    text = text.lower()
    detailed = ["months", "weeks", "feeds", "hours", "per day", "twice", "once", "schedule", "old", "days old"]
    vague = ["hungry", "worried", "concerned", "not sure", "help", "dont know", "don't know", "something's wrong"]

    if any(w in text for w in detailed):
        return "DETAILED"
    if any(w in text for w in vague):
        return "VAGUE"
    return "NEUTRAL"


def get_conversation_state(chat_history):
    if not chat_history:
        return "NEW"

    last_user = None
    for msg in reversed(chat_history):
        if msg["role"] == "user":
            last_user = msg["content"].lower()
            break

    if last_user and any(w in last_user for w in ["hungry", "feeding", "milk", "latch", "breast", "nipple"]):
        return "ACTIVE_BREASTFEEDING_THREAD"

    return "OTHER"


def baby_age_known(chat_history, user_input):
    all_text = user_input.lower()
    for msg in chat_history:
        all_text += " " + msg["content"].lower()
    age_pattern = r'\b\d+\s*(day|days|week|weeks|month|months)\s*(old)?\b'
    explicit_terms = ["newborn", "infant"]
    return bool(re.search(age_pattern, all_text)) or any(w in all_text for w in explicit_terms)


def needs_clarification(user_input, chat_history):
    text = user_input.lower()
    all_text = text + " " + " ".join([msg["content"].lower() for msg in chat_history])

    age_known = bool(re.search(r'\b\d+\s*(day|days|week|weeks|month|months)\s*(old)?\b', all_text)) \
        or any(w in all_text for w in ["newborn", "infant"])
        
    feeding_known = any(w in all_text for w in [
        "breastfeeding", "breastfeed", "breast fed", "formula", "pumping", "pump", "combination",
        "times a day", "feeds a day", "feeding every", "every 2 hours", "every 3 hours",
        "times per day", "feeds per day", "8 times", "10 times", "12 times",
    ])

    diaper_known = any(phrase in all_text for phrase in ["wet diaper", "wet diapers", "diapers wet"]) or \
                   bool(re.search(r'\b\d+\s*wet\s*diaper', all_text))

    context_score = sum([age_known, diaper_known, feeding_known])
    return context_score < 3


def is_closing_message(user_input):
    closing_words = [
        "thank you", "thanks", "that helped", "that was helpful", "that's helpful",
        "got it", "okay thanks", "great thanks", "ok thanks", "appreciate it",
        "makes sense", "that makes sense", "perfect", "awesome thanks", "helpful thanks"
    ]
    return any(phrase in user_input.lower() for phrase in closing_words)


def route_request(scores, flags, user_input, chat_history, retrieval_scores_list=None):
    if is_closing_message(user_input):
        return "CLOSING"

    flag_status = check_hard_medical_red_flags(user_input, chat_history)
    if flag_status == "INFANT_CRISIS":
        return "URGENT_INFANT"
    elif flag_status == "LOW_COUNTS_URGENT":
        return "URGENT_LOW_COUNTS"
    elif flag_status == "MATERNAL_URGENT":
        return "URGENT_MATERNAL"

    # NEW: off-topic gate. Runs after red-flag checks (safety always first)
    # but before any breastfeeding-specific routing logic, so unrelated
    # queries (e.g. hospital/childcare recommendations) never fall into the
    # clinical triage flow.
    if is_off_topic(user_input, retrieval_scores_list):
        return "OFF_TOPIC"

    if flags["pain"] or flags["latch"]:
        if not needs_clarification(user_input, chat_history):
            return "CLINICAL"

    if needs_clarification(user_input, chat_history):
        return "QUESTION_FIRST"

    clinical_score = flags["pain"] + flags["latch"] + flags["supply"]
    if flags["stress"] and clinical_score == 0 and not flags["concern"]:
        return "SUPPORT"

    if flags["concern"] and not (flags["pain"] or flags["latch"] or flags["supply"]):
        return "REASSURE"

    return "CLINICAL"


@app.route('/')
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html", chat_history=session["chat_history"])


@app.route("/submit", methods=["POST"])
def submit():
    user_input = request.form.get("user_input", "")
    
    if "chat_history" not in session:
        session["chat_history"] = []
    chat_history = session["chat_history"]

    scores, flags = score_text_concern(user_input, chat_history)

    # Retrieval is now run up-front (rather than only inside the CLINICAL/etc.
    # branch below) so its similarity scores can inform the off-topic gate
    # in route_request(). k=2 kept the same as before.
    search_results, retrieval_scores_list = search(user_input, faiss_index, chunks, k=2)
    retrieved_context = "\n".join(search_results) if search_results else ""

    route = route_request(scores, flags, user_input, chat_history, retrieval_scores_list)
    flag_status = check_hard_medical_red_flags(user_input, chat_history)
    
    if flag_status == "INFANT_CRISIS":
        infant_crisis_response = """
        <div class="triage-alert">
            <h3>⚠️ Seek Immediate Pediatric Care</h3>
            <p>Your baby's symptoms — including <strong>fever and/or direct feeding refusal</strong> — require prompt medical evaluation. Do not wait.</p>
            <ul>
                <li><strong>Call your pediatrician or visit the ER immediately.</strong></li>
                <li>Fever in young infants is always treated as a medical emergency.</li>
                <li>An absolute refusal to feed combined with high temperature requires rapid clinical assessment.</li>
            </ul>
        </div>
        """
        return render_template("result.html", user_input=user_input, response=infant_crisis_response)

    elif flag_status == "LOW_COUNTS_URGENT":
        low_count_response = """
        <div class="triage-alert">
            <h3>⚠️ Low Intake / Output Metrics Noted</h3>
            <p>You mentioned that your baby has had a low number of wet diapers or daily feedings. While this may not be a crisis if the baby is alert and responsive, low output requires careful management.</p>
            <ul>
                <li><strong>Monitor Closely:</strong> Infants typically need 6+ wet diapers and 8–12 regular feedings per 24 hours to stay adequately hydrated.</li>
                <li><strong>Coordinate Care:</strong> Contact your healthcare provider or a lactation consultant today to arrange a physical evaluation and weight check.</li>
                <li><strong>When to go to the ER:</strong> If your baby displays extreme lethargy, is difficult to wake up, develops a dry mouth/sunken soft spot, or experiences a fever, seek emergency medical care instantly.</li>
            </ul>
        </div>
        """
        return render_template("result.html", user_input=user_input, response=low_count_response)

    elif flag_status == "MATERNAL_URGENT":
        maternal_response = """
        <div class="triage-alert">
            <h3>⚠️ Immediate Medical Evaluation Recommended</h3>
            <p>Your symptoms strongly point toward a systemic infection like <strong>mastitis</strong>.</p>
            <ul>
                <li><strong>Seek Professional Care Immediately:</strong> Contact your provider or visit an urgent care today.</li>
                <li><strong>Breastfeeding Guidance:</strong> Standard guidelines advise continuing to breastfeed or pump frequently on the affected side to clear the blockage unless advised otherwise by a doctor.</li>
            </ul>
        </div>
        """
        return render_template("result.html", user_input=user_input, response=maternal_response)

    # NEW: off-topic branch. Handled with a static response and no LLM call
    # (saves a generate() round trip, and guarantees it never drifts into
    # asking feeding-triage questions for unrelated requests).
    elif route == "OFF_TOPIC":
        off_topic_response = """
        <div class="info-note">
            <p>I'm focused specifically on breastfeeding and postpartum feeding support,
            so I'm not the right resource for that request. For hospital or childcare
            recommendations, your insurance provider's directory or a local parent
            resource line would be a better starting point.</p>
        </div>
        """

        try:
            session_id = session.get("session_id", str(uuid.uuid4()))
            session["session_id"] = session_id
            log_turn(
                session_id=session_id,
                turn_number=len(chat_history) // 2 + 1,
                route=route,
                scores={
                    "pain":    scores["pain"],
                    "latch":   scores["latch"],
                    "supply":  scores["supply"],
                    "stress":  scores["stress"],
                    "urgency": scores["red_flag"],
                },
                flags={
                    "pain":    flags["pain"],
                    "latch":   flags["latch"],
                    "supply":  flags["supply"],
                    "stress":  flags["stress"],
                    "urgency": flags["red_flag"],
                },
                detail_level=detect_detail_level(user_input),
                conv_state=get_conversation_state(chat_history),
                retrieval_chunks=[retrieved_context] if retrieved_context else [],
                retrieval_scores=retrieval_scores_list,
                user_msg_len=len(user_input),
                baby_age_known=baby_age_known(chat_history, user_input),
                is_closing=is_closing_message(user_input),
            )
        except Exception as log_err:
            print(f"⚠️ Analytics Database Sync Skipped: {log_err}", flush=True)

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": off_topic_response})
        session["chat_history"] = chat_history

        return render_template("result.html", user_input=user_input, response=off_topic_response)

    intent_handling_directive = (
        "USER INTENT HANDLING DIRECTIVE:\n"
        "1. Distinguish between an objective, general educational question and a personal symptom disclosure.\n"
        "2. If the user's input is a general/educational question, provide a direct, objective medical answer.\n"
        "3. If the user explicitly mentions they are personally experiencing a symptom, adopt an empathetic tone and follow workflows.\n\n"
    )

    if route == "QUESTION_FIRST":
        all_text = user_input.lower() + " " + " ".join([m["content"].lower() for m in chat_history])
        age_known = bool(re.search(r'\b\d+\s*(day|days|week|weeks|month|months)\s*(old)?\b', all_text)) \
            or any(w in all_text for w in ["newborn", "infant"])
        feeding_known = any(w in all_text for w in [
            "breastfeeding", "breastfeed", "breast fed", "formula", "pumping", "pump",
            "combination", "times a day", "feeds a day", "feeding every",
        ])
        diaper_known = any(phrase in all_text for phrase in ["wet diaper", "wet diapers", "diapers wet"]) or \
                       bool(re.search(r'\b\d+\s*wet\s*diaper', all_text))

        unknown_facts = []
        if not age_known:
            unknown_facts.append("how old your baby is (in days or weeks)")
        if not feeding_known:
            unknown_facts.append("how you are currently feeding (breast, pump, or formula)")
        if not diaper_known:
            unknown_facts.append("the number of wet diapers in the last 24 hours")

        if unknown_facts:
            # We create a very explicit string for the model to include
            missing_str = " and ".join(unknown_facts)
            clarification_directive = (
                "You are a helpful maternal health assistant. The user's query is too vague to provide clinical guidance.\n\n"
                "STRICT INSTRUCTION:\n"
                f"1. You MUST ask the user for: {missing_str}.\n"
                "2. Do not provide any medical advice, tips, or 'normal' ranges yet.\n"
                "3. Start with a brief empathetic opening, then ask the missing questions clearly.\n"
                "4. Keep your total response under 40 words.\n"
            )
        else:
            clarification_directive = (
                "Ask one focused follow-up question to better understand the concern.\n"
            )

        system_prompt = (
            f"{clarification_directive}\n"
            f"Grounding Context (Use for tone only, do not cite facts yet):\n{retrieved_context}\n"
        )

        system_prompt = (
            f"{clarification_directive}"
            f"Grounding Context:\n{retrieved_context}\n\n"
        )
    elif route == "SUPPORT":
        system_prompt = (
            "You are a warm, compassionate maternal health assistant. This mother is struggling emotionally. "
            "Start by validating her feelings genuinely and specifically. Offer 1–2 gentle, practical suggestions.\n\n"
            f"Grounding Context:\n{retrieved_context}\n\n"
        )
    elif route == "REASSURE":
        system_prompt = (
            "You are a calm, reassuring maternal health assistant. The user has expressed a worry or concern "
            "but there are no red flags or clinical symptoms indicated.\n\n"
            "REASSURE DIRECTIVE:\n"
            "- Lead with normalisation: explain that what they're noticing is common and usually not a cause for concern.\n"
            "- Provide 2–3 concrete, observable signs that things ARE going well (e.g. wet diapers, weight gain, feeding cues).\n"
            "- Close with a clear, low-stress next step or a simple 'when to seek help' note.\n"
            "- Tone: warm and confident. Not dismissive — acknowledge their concern genuinely.\n\n"
            f"{intent_handling_directive}"
            f"Grounding Context:\n{retrieved_context}\n\n"
        )
    else:
        system_prompt = (
            "You are a helpful and empathetic maternal health assistant and lactation consultant.\n\n"
            "CRITICAL INTEGRITY INSTRUCTION:\n"
            "Review the 'Current User Input' and 'Conversation History' fields carefully. "
            "NEVER state, imply, or assume the user or their baby is experiencing specific symptoms "
            "(e.g., slow weight gain, specific low diaper counts, bleeding) UNLESS the user explicitly stated "
            "those symptoms in their messages. Treat the 'Grounding Context' strictly as an objective reference textbook, "
            "not a description of the current patient.\n\n"
            "RESPONSE STRATEGY:\n"
            "1. Acknowledge what the user shared. If their message is brief, address it generally without assuming pathologies.\n"
            "2. Read the Conversation History carefully. Do not repeat questions already answered.\n"
            "3. Offer 2-3 practical, supportive educational insights based on the Grounding Context. Frame them objectively "
            "(e.g., 'In general, newborns typically need...' or 'Lactation guidelines suggest looking for...') rather than "
            "diagnosing the user.\n"
            "4. Conclude with clear, low-stress parameters on when they should reach out to a healthcare provider or IBCLC.\n\n"
            f"{intent_handling_directive}"
            "GROUNDING CONTEXT (Reference Materials Only - Do not assume the patient matches this):\n"
            f"{retrieved_context}\n\n"
            "IMPORTANT: Do not mention mastitis unless the user brings it up or systemic symptoms like fever are present. "
            "Formatting: Use clear, simple Markdown with bold section headers."
        )

    history_context = build_conversation_text(chat_history)
    full_prompt = (
        f"### SYSTEM INSTRUCTION:\n{system_prompt}\n\n"
        f"### CONVERSATION HISTORY:\n{history_context if history_context else 'No prior history.'}\n\n"
        f"### ACTUAL PATIENT INPUT:\n\"{user_input}\"\n\n"
        f"### ASSISTANT RESPONSE:\n"
    )

    try:
        response = client.generate(
            model=MODEL, 
            prompt=full_prompt,
            options={
                "num_predict": 600,   
                "temperature": 0.2,   
                "top_k": 20           
            }
        )
        ai_text = response['response']

        try:
            session_id = session.get("session_id", str(uuid.uuid4()))
            session["session_id"] = session_id
            log_turn(
                session_id=session_id,
                turn_number=len(chat_history) // 2 + 1,
                route=route,
                scores={
                    "pain":    scores["pain"],
                    "latch":   scores["latch"],
                    "supply":  scores["supply"],
                    "stress":  scores["stress"],
                    "urgency": scores["red_flag"],
                },
                flags={
                    "pain":    flags["pain"],
                    "latch":   flags["latch"],
                    "supply":  flags["supply"],
                    "stress":  flags["stress"],
                    "urgency": flags["red_flag"],
                },
                detail_level=detect_detail_level(user_input),
                conv_state=get_conversation_state(chat_history),
                retrieval_chunks=[retrieved_context] if retrieved_context else [],
                retrieval_scores=retrieval_scores_list,
                user_msg_len=len(user_input),
                baby_age_known=baby_age_known(chat_history, user_input),
                is_closing=is_closing_message(user_input),
            )
        except Exception as log_err:
            print(f"⚠️ Analytics Database Sync Skipped: {log_err}", flush=True)

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": ai_text})
        session["chat_history"] = chat_history

        html_response = md.markdown(ai_text)
        return render_template("result.html", user_input=user_input, response=html_response)

    except Exception as e:
        print(f"💥 Backend Prompt Generation Error: {e}", flush=True)
        return render_template("result.html", user_input=user_input, response=f"<p>Execution Error: {e}</p>")


@app.route('/reset', methods=['GET', 'POST'])
def reset():
    session.pop("chat_history", None)
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True, port=5002)