from flask import Flask, render_template, request, session
import ollama
from rag_embeddings import build_index, search
from rag import load_knowledge

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

client = ollama.Client()

chunks = load_knowledge()
faiss_index, embeddings, chunks = build_index(chunks)


def score_text_concern(user_input, chat_history):
    # Combine the latest input with previous messages to retain memory of critical issues
    full_context_text = user_input.lower()
    for msg in chat_history:
        full_context_text += " " + msg["content"].lower()

    scores = {"pain": 0, "latch": 0, "supply": 0, "stress": 0, "urgency": 0}

    if any(w in full_context_text for w in ["pain", "sore", "cracked", "bleeding"]):
        scores["pain"] = 1
    if any(w in full_context_text for w in ["latch", "not latching", "trouble latching"]):
        scores["latch"] = 1
    if any(w in full_context_text for w in ["not enough milk", "still hungry", "low milk"]):
        scores["supply"] = 1
    if any(w in full_context_text for w in ["stressed", "crying", "overwhelmed"]):
        scores["stress"] = 1
    
    # CRITICAL TRIGGER: Catches standard emergency keywords
    if any(w in full_context_text for w in ["fever", "severe pain", "not feeding"]):
        scores["urgency"] = 1

    # ROBUST DANGEROUS SCHEDULE DETECTOR: 
    # Catches variations like "twice a day", "three times a day", "2 times a day", etc.
    low_frequencies = ["once", "twice", "three", "1 time", "2 times", "3 times", "1time", "2time", "3time"]
    time_frames = ["a day", "per day", "every day", "daily", "24 hours"]
    
    # If ANY low frequency word AND ANY time frame word show up in the history, check for urgency
    if any(freq in full_context_text for freq in low_frequencies) and any(time in full_context_text for time in time_frames):
        # FIX: Explicitly ignore emergency routing if they also mentioned a completely healthy range
        if not any(healthy in full_context_text for healthy in ["8", "9", "10", "11", "12", "eight", "ten", "twelve"]):
            scores["urgency"] = 1

    return scores


def detect_detail_level(text):
    text = text.lower()

    detailed = ["months", "weeks", "feeds", "hours", "per day", "twice", "once", "schedule"]
    vague = ["hungry", "worried", "concerned", "not sure", "help", "dont know", "don't know"]

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

    if last_user and any(w in last_user for w in ["hungry", "feeding", "milk", "latch"]):
        return "ACTIVE_BREASTFEEDING_THREAD"

    return "OTHER"


def route_request(scores, user_input, chat_history):
    # Urgency always overrides everything else, keeping the user in the safe triage loop
    if scores["urgency"]:
        return "URGENT"

    state = get_conversation_state(chat_history)
    detail = detect_detail_level(user_input)

    follow_up_words = ["yes", "yeah", "ok", "okay", "correct", "right"]

    if user_input.strip().lower() in follow_up_words and state == "ACTIVE_BREASTFEEDING_THREAD":
        return "CLINICAL"

    if state == "ACTIVE_BREASTFEEDING_THREAD" and detail in ["DETAILED", "NEUTRAL"]:
        return "CLINICAL"

    if detail == "VAGUE":
        return "QUESTION_FIRST"

    if scores["pain"] and scores["latch"]:
        return "CLINICAL"

    if scores["stress"]:
        return "SUPPORT"

    return "GENERAL"


@app.route('/')
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html", chat_history=session["chat_history"])


@app.route('/submit', methods=['POST'])
def submit():
    if "chat_history" not in session:
        session["chat_history"] = []

    user_input = request.form['message']
    chat_history = session["chat_history"]

    # Evaluates context across the whole chat thread to prevent routing drop-offs
    scores = score_text_concern(user_input, chat_history)
    mode = route_request(scores, user_input, chat_history)

    conversation_text = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n"

    # BRANCH 1: Vague input from user, collect vital context safely
    if mode == "QUESTION_FIRST":
        prompt = """
You are an expert, empathetic breastfeeding triage assistant. 

The user is concerned that their baby is still hungry after a feed, but you don't have enough details yet to know if the baby is safe.

Your goal is to gently ask exactly 2 short, warm questions to find out how often they are feeding and if the baby is getting enough milk.

RULES:
- ONLY ask 2 short questions.
- Question 1 must ask how many times a day the baby is being fed.
- Question 2 must ask about the baby's wet/dirty diapers or if they seem content after some feeds.
- Do not provide any medical advice, explanations, or formatting. Keep it conversational and supportive.
"""
        response = client.generate(model='qwen2.5:1.5b', prompt=prompt)
        output = response["response"].strip()

    # BRANCH 2: High priority feeding frequency alert or medical hazard detected
    elif mode == "URGENT":
        prompt = f"""
You are an empathetic, clinically safe breastfeeding triage assistant.

CRITICAL MEDICAL ALERT: 
The user's baby may be dangerously underfed (e.g., only eating 1-3 times a day) or experiencing another medical risk. Newborn babies must feed 8–12 times every 24 hours to prevent severe dehydration and lethargy.

RULES:
- Completely drop the standard markdown format template (do NOT use sections like Clarify, Key Insight, Steps).
- Provide immediate, clear, and deeply supportive guidance explaining that newborns need to be fed 8-12 times a day (every 2-3 hours).
- Keep your tone calm, direct, and completely focused on this feeding frequency rule.
- Explicitly advise the parent to contact a pediatrician or lactation consultant immediately.

CHAT HISTORY:
{conversation_text}

USER:
{user_input}
"""
        response = client.generate(model='qwen2.5:1.5b', prompt=prompt)
        output = response["response"].strip()

    # BRANCH 3: Standard routine clinical advice using RAG vector context
    else:
        context = search(user_input, faiss_index, chunks)[:3]
        context_text = "\n\n".join(context) if context else "No relevant context."

        prompt = f"""
You are a safe breastfeeding assistant.

RULES:
- Use CONTEXT when helpful to directly answer the user's situation.
- You may use general safe breastfeeding knowledge if context is missing.
- Do NOT hallucinate extreme medical conditions not supported by the context.

FORMAT:

Clarify:
- 0–2 questions if needed

Key Insight:
- 1–3 bullets

Steps:
- 2–4 practical steps

When to Seek Help:
- 1 sentence

CONTEXT:
{context_text}

CHAT HISTORY:
{conversation_text}

USER:
{user_input}
"""
        response = client.generate(model='qwen2.5:1.5b', prompt=prompt)
        output = response["response"].strip()

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": output})
    session["chat_history"] = chat_history

    return render_template("index.html", chat_history=chat_history)


@app.route('/reset', methods=['POST'])
def reset():
    session.pop("chat_history", None)
    return render_template("index.html", chat_history=[])


if __name__ == '__main__':
    app.run(debug=True, port=5001)