# Maternal Breastfeeding Support — AI Triage Assistant

An AI-powered breastfeeding support tool that routes user concerns through a semantic triage pipeline and responds with clinically grounded, empathy-aware advice. Built with Flask, FAISS, sentence embeddings, and a local LLM (Llama 3.2).

---

## What It Does

Most AI chatbots treat every message the same. This tool doesn't. Every user message is semantically scored, classified into a concern category, and routed to a purpose-built prompt — before a single token is generated.

**Routing pipeline:**

| Route | Triggered When | Response Style |
|---|---|---|
| `URGENT` | Fever, abscess, baby not feeding | Calm, direct, refer to clinician today |
| `SUPPORT` | Emotional distress, giving up | Empathy first, 1–2 practical suggestions |
| `CLINICAL` | Specific symptoms with enough context | Structured: Key Insight → Steps → When to Seek Help |
| `QUESTION_FIRST` | Vague concern without enough detail | Asks 2 open-ended clarifying questions |
| `CLOSING` | Thank-you messages | Warm close, no new advice |

---

## Architecture

```
User Input
    │
    ▼
Semantic Scoring (sentence-transformers + cosine similarity)
    │   Pre-computed embeddings for: pain, latch, supply, stress, urgency
    ▼
Rule-Based Router (route_request)
    │   Also checks: detail level, conversation state, baby age known
    ▼
RAG Retrieval (FAISS index over knowledge base .txt files)
    │   Top-k chunks retrieved, filtered by cosine > 0.3
    ▼
Prompt Construction (mode-specific templates)
    │
    ▼
LLM (Llama 3.2 3B via Ollama)
    │
    ▼
Response → Markdown → HTML → Rendered in chat
```

**Key design decisions:**
- Concern embeddings are **pre-computed at startup**, not per-request — avoids redundant encoding on every message
- Chat history is **capped at 20 messages** to prevent context overflow
- Semantic scoring uses a **sliding window of the last 6 messages** to avoid old context bleeding into current concern detection
- The LLM is explicitly instructed it is an AI, not a clinician — safety guardrails baked into every prompt

---

## Knowledge Base

Seven curated `.txt` files in `/knowledge_base/`, split into paragraph-level chunks for RAG retrieval:

- `breastfeeding.txt` — general foundations, newborn feeding norms
- `latch_techniques.txt` — positioning, tongue tie, nipple shields
- `nipple_pain.txt` — causes, thrush, vasospasm, blebs, treatment
- `engorgement.txt` — reverse pressure softening, cabbage leaves, timing of warmth/cold
- `milk_supply.txt` — supply-and-demand mechanics, power pumping, galactagogues
- `clogged_ducts.txt` — massage, lecithin, dangle feed, vibration technique
- `mastitis.txt` — symptoms, antibiotics, abscess risk, subclinical mastitis

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally
- Llama 3.2 3B pulled: `ollama pull llama3.2:3b`

### Install

```bash
git clone https://github.com/yourusername/ai-breastfeeding-support
cd ai-breastfeeding-support
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open `http://localhost:5001` in your browser.

---

## Evaluation

Run the routing and retrieval evaluation suite:

```bash
python eval.py
```

See [`eval.py`](eval.py) for test cases covering all 5 routing modes. Reports routing accuracy, retrieval hit rate, and flags potential hallucination risk.

---

## Project Timeline

This project was built across a 9-week structured development plan:

| Week | Focus |
|---|---|
| 1 | Input schema, rule-based prototype, early feedback |
| 2 | LLM integration, Flask web interface |
| 3 | Semantic scoring and routing layer |
| 4 | Knowledge base curation, RAG pipeline |
| 5 | Session persistence, conversation state tracking |
| 6 | User testing, UX and response framing refinement |
| 7 | Safety filters, urgency detection, clinician referral guardrails |
| 8 | Performance optimization, architecture cleanup |
| 9 | Final prototype, demo preparation |

---

## Limitations & Future Work

- **Memory:** Currently stores raw chat history. A stronger system would extract structured facts (baby age, feeding frequency, previous concerns) rather than re-reading full conversation text each turn.
- **Retrieval:** Uses dense vector search only. Hybrid BM25 + FAISS retrieval would improve recall on keyword-specific queries.
- **Evaluation:** Routing accuracy measured on hand-labeled test cases. LLM response quality (hallucination rate, citation accuracy) not yet formally evaluated.
- **Citations:** Responses do not currently surface which knowledge base file a claim came from.

---

## Disclaimer

This tool is an AI assistant for informational support only. It is not a substitute for professional medical advice. Users experiencing medical emergencies should contact a healthcare provider directly.
