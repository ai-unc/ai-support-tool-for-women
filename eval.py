"""
eval.py — Evaluation suite for the Maternal Breastfeeding Support tool.

Tests three things:
  1. Routing accuracy  — does route_request() send each message to the right mode?
  2. Retrieval quality — does the FAISS index return relevant chunks for clinical queries?
  3. Hallucination risk — does the LLM response contain claims unsupported by retrieved context?

Run:
    python eval.py
    python eval.py --verbose          # show retrieved chunks and full responses
    python eval.py --skip-llm         # routing + retrieval only (no Ollama needed)
"""

import argparse
import time
import numpy as np
from rag_embeddings import build_index, search, model as embedding_model
from rag import load_knowledge

# ── Import routing helpers from app (without starting Flask) ─────────────────
import sys
import types

# Stub Flask/session so we can import routing logic without a running server
flask_stub = types.ModuleType("flask")
flask_stub.Flask = lambda *a, **kw: None
flask_stub.render_template = lambda *a, **kw: None
flask_stub.request = None
flask_stub.session = {}
sys.modules.setdefault("flask", flask_stub)
sys.modules.setdefault("flask_session", types.ModuleType("flask_session"))
sys.modules.setdefault("ollama", types.ModuleType("ollama"))
sys.modules.setdefault("markdown", types.ModuleType("markdown"))

from app import (
    score_text_concern,
    route_request,
    detect_detail_level,
    get_conversation_state,
    baby_age_known,
    is_closing_message,
    CONCERN_EMBEDDINGS,
    THRESHOLDS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ROUTING TEST CASES
# Each case has: input message, optional prior history, and expected route.
# ─────────────────────────────────────────────────────────────────────────────

ROUTING_TEST_CASES = [
    # ── URGENT_MATERNAL ───────────────────────────────────────────────────────
    {
        "id": "U1",
        "input": "I have a high fever and my breast is bright red and hot. I feel like I have the flu.",
        "history": [],
        "expected": "URGENT_MATERNAL",
        "note": "Classic mastitis presentation",
    },
    {
        "id": "U3",
        "input": "There's a lump in my breast that feels like it's filled with fluid and it's getting bigger.",
        "history": [],
        "expected": "URGENT_MATERNAL",
        "note": "Possible abscess",
    },

    # ── URGENT_INFANT ─────────────────────────────────────────────────────────
    {
        "id": "U2",
        "input": "My baby hasn't fed in 9 hours and won't latch at all. I'm scared.",
        "history": [],
        "expected": "URGENT_INFANT",
        "note": "Baby not feeding — danger signal",
    },
    {
        "id": "U4",
        "input": "Baby isn't drinking milk and has a fever.",
        "history": [],
        "expected": "URGENT_INFANT",
        "note": "Baby fever + feeding refusal — the failing case from the screenshot",
    },
    {
        "id": "U5",
        "input": "My newborn won't eat and feels really warm.",
        "history": [],
        "expected": "URGENT_INFANT",
        "note": "Newborn fever + refusal phrased casually",
    },
    {
        "id": "U6",
        "input": "She has a fever and is refusing to feed.",
        "history": [],
        "expected": "URGENT_INFANT",
        "note": "Baby fever + refusal using pronoun",
    },

    # ── SUPPORT ───────────────────────────────────────────────────────────────
    {
        "id": "S1",
        "input": "I can't do this anymore. I've been trying for 3 weeks and I just want to give up.",
        "history": [],
        "expected": "SUPPORT",
        "note": "Emotional distress / giving up",
    },
    {
        "id": "S2",
        "input": "I feel like such a failure. Every other mom makes this look so easy.",
        "history": [],
        "expected": "SUPPORT",
        "note": "Self-criticism / emotional distress",
    },
    {
        "id": "S3",
        "input": "I'm so exhausted and overwhelmed. I've been crying all day.",
        "history": [],
        "expected": "SUPPORT",
        "note": "Exhaustion + emotional distress",
    },

    # ── CLINICAL ─────────────────────────────────────────────────────────────
    {
        "id": "C1",
        "input": "My nipples are cracked and bleeding after every feed. Baby is 2 weeks old.",
        "history": [],
        "expected": "CLINICAL",
        "note": "Nipple pain with age context = detailed enough for clinical",
    },
    {
        "id": "C2",
        "input": "I have a hard lump on the outer side of my left breast that's been there for 2 days.",
        "history": [],
        "expected": "CLINICAL",
        "note": "Clogged duct — specific and detailed",
    },
    {
        "id": "C3",
        "input": "My baby feeds every hour but never seems satisfied. She's 6 weeks old and I pump 2oz per side.",
        "history": [],
        "expected": "CLINICAL",
        "note": "Supply concern with full detail",
    },
    {
        "id": "C4",
        "input": "yes",
        "history": [
            {"role": "user", "content": "My baby won't latch properly"},
            {"role": "assistant", "content": "How old is your baby and how many times per day are they feeding?"},
            {"role": "user", "content": "She's 3 weeks old and feeding about 8 times a day"},
        ],
        "expected": "CLINICAL",
        "note": "Short affirmative in active breastfeeding thread",
    },

    # ── QUESTION_FIRST ────────────────────────────────────────────────────────
    {
        "id": "Q1",
        "input": "I'm worried my baby isn't getting enough milk.",
        "history": [],
        "expected": "QUESTION_FIRST",
        "note": "Vague supply concern — needs clarification",
    },
    {
        "id": "Q2",
        "input": "Something feels wrong but I don't know what.",
        "history": [],
        "expected": "QUESTION_FIRST",
        "note": "Explicitly vague",
    },
    {
        "id": "Q3",
        "input": "I'm not sure if I'm doing this right.",
        "history": [],
        "expected": "QUESTION_FIRST",
        "note": "Vague concern without specifics",
    },

    # ── CLOSING ───────────────────────────────────────────────────────────────
    {
        "id": "CL1",
        "input": "Thank you so much, that really helped!",
        "history": [],
        "expected": "CLOSING",
        "note": "Gratitude",
    },
    {
        "id": "CL2",
        "input": "Got it, makes sense. Thanks!",
        "history": [],
        "expected": "CLOSING",
        "note": "Acknowledgement + thanks",
    },
    {
        "id": "CL3",
        "input": "okay thanks",
        "history": [],
        "expected": "CLOSING",
        "note": "Casual closing",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. RETRIEVAL TEST CASES
# Each case checks that at least one of the expected_keywords appears
# in the top-k retrieved chunks for a given query.
# ─────────────────────────────────────────────────────────────────────────────

RETRIEVAL_TEST_CASES = [
    {
        "id": "R1",
        "query": "My nipple is cracked and bleeding after feeding",
        "expected_keywords": ["crack", "latch", "lanolin", "areola"],
        "note": "Nipple pain — should retrieve nipple_pain or latch chunks",
    },
    {
        "id": "R2",
        "query": "I have a hard painful lump in my breast",
        "expected_keywords": ["plug", "duct", "lump", "massage", "clogged"],
        "note": "Clogged duct — should retrieve clogged_ducts chunks",
    },
    {
        "id": "R3",
        "query": "My baby doesn't seem satisfied after feeding and I think I have low milk supply",
        "expected_keywords": ["supply", "demand", "pump", "frequency", "milk"],
        "note": "Low supply — should retrieve milk_supply chunks",
    },
    {
        "id": "R4",
        "query": "I have a fever and flu-like symptoms with a red swollen breast",
        "expected_keywords": ["mastitis", "fever", "antibiotic", "infection", "abscess"],
        "note": "Mastitis — should retrieve mastitis chunks",
    },
    {
        "id": "R5",
        "query": "My breasts are rock hard and my baby can't latch because of the firmness",
        "expected_keywords": ["engorgement", "engorged", "areola", "softening", "reverse"],
        "note": "Engorgement — should retrieve engorgement chunks",
    },
    {
        "id": "R6",
        "query": "How do I get my baby to latch on properly?",
        "expected_keywords": ["latch", "mouth", "areola", "nipple", "position"],
        "note": "Latch technique — should retrieve latch_techniques chunks",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. HALLUCINATION CHECK CASES
# These test that LLM output for a clinical query doesn't contain
# terms that were NOT in the retrieved context.
# ─────────────────────────────────────────────────────────────────────────────

HALLUCINATION_CHECK_CASES = [
    {
        "id": "H1",
        "query": "I have a hard lump in my breast that is sore",
        "forbidden_terms": ["cervix", "ovary", "uterus", "menstrual", "pregnancy test"],
        "note": "Should not introduce unrelated anatomical references",
    },
    {
        "id": "H2",
        "query": "My nipple hurts during the whole feed",
        "forbidden_terms": ["abscess", "mastitis", "fever", "antibiotics"],
        "note": "Nipple pain alone should not escalate to mastitis/abscess without fever context",
    },
    {
        "id": "H3",
        "query": "My baby feeds every 2 hours. Is that normal?",
        "forbidden_terms": ["low supply", "formula", "supplementation", "underfed"],
        "note": "Frequent feeding is normal — should not assume pathology",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_routing_eval(verbose=False):
    print("\n" + "═" * 60)
    print("  ROUTING ACCURACY EVALUATION")
    print("═" * 60)

    passed = 0
    failed = 0
    results = []

    for case in ROUTING_TEST_CASES:
        scores, flags = score_text_concern(case["input"], case["history"])
        actual = route_request(scores, flags, case["input"], case["history"])
        ok = actual == case["expected"]

        if ok:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"

        results.append({
            "id": case["id"],
            "ok": ok,
            "expected": case["expected"],
            "actual": actual,
            "note": case["note"],
            "input": case["input"],
            "scores": scores,
            "flags": flags,
        })

        print(f"  [{case['id']}] {status}  expected={case['expected']:<15} got={actual:<15}  — {case['note']}")
        if verbose and not ok:
            print(f"        Input: \"{case['input'][:80]}\"")
            print(f"        Scores: {scores}")
            print(f"        Flags:  {flags}")

    total = passed + failed
    pct = (passed / total) * 100
    print(f"\n  Result: {passed}/{total} passed ({pct:.0f}%)")
    if failed > 0:
        print("  Failed cases:")
        for r in results:
            if not r["ok"]:
                print(f"    [{r['id']}] expected {r['expected']}, got {r['actual']}")
                print(f"         Scores: {r['scores']}")
                print(f"         Flags:  {r['flags']}")
    return passed, total


def run_retrieval_eval(faiss_index, chunks, verbose=False):
    print("\n" + "═" * 60)
    print("  RETRIEVAL QUALITY EVALUATION")
    print("═" * 60)

    passed = 0
    failed = 0

    for case in RETRIEVAL_TEST_CASES:
        retrieved = search(case["query"], faiss_index, chunks, k=3)
        combined = " ".join(retrieved).lower()

        hits = [kw for kw in case["expected_keywords"] if kw.lower() in combined]
        ok = len(hits) > 0

        if ok:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"

        print(f"  [{case['id']}] {status}  keywords_hit={hits or 'NONE'}  — {case['note']}")

        if verbose:
            print(f"        Query: \"{case['query'][:70]}\"")
            for i, chunk in enumerate(retrieved):
                print(f"        Chunk {i+1}: \"{chunk[:100].strip()}...\"")

    total = passed + failed
    pct = (passed / total) * 100
    print(f"\n  Result: {passed}/{total} passed ({pct:.0f}%)")
    return passed, total


def run_hallucination_eval(faiss_index, chunks, client=None, model=None, verbose=False, skip_llm=False):
    print("\n" + "═" * 60)
    print("  HALLUCINATION RISK EVALUATION")
    print("═" * 60)

    if skip_llm or client is None:
        print("  [SKIPPED] Pass --no-skip-llm and ensure Ollama is running to enable LLM checks.")
        return None, None

    passed = 0
    failed = 0

    for case in HALLUCINATION_CHECK_CASES:
        retrieved = search(case["query"], faiss_index, chunks, k=3)
        context_text = "\n\n".join(retrieved) if retrieved else "No relevant context found."

        prompt = f"""You are a safe breastfeeding support assistant.
Use ONLY the context below. Do not add information not supported by the context.

CONTEXT:
{context_text}

User: {case["query"]}

Response:"""

        try:
            response = client.generate(model=model, prompt=prompt)
            output = response.response.strip().lower()
        except Exception as e:
            print(f"  [{case['id']}] ⚠️  LLM ERROR: {e}")
            continue

        found_forbidden = [term for term in case["forbidden_terms"] if term.lower() in output]
        ok = len(found_forbidden) == 0

        if ok:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"

        print(f"  [{case['id']}] {status}  — {case['note']}")
        if not ok:
            print(f"        Forbidden terms found: {found_forbidden}")
        if verbose:
            print(f"        Response snippet: \"{output[:200]}\"")

    total = passed + failed
    if total > 0:
        pct = (passed / total) * 100
        print(f"\n  Result: {passed}/{total} passed ({pct:.0f}%)")
    return passed, total


def print_summary(routing, retrieval, hallucination):
    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    r_pass, r_total = routing
    ret_pass, ret_total = retrieval
    print(f"  Routing accuracy:   {r_pass}/{r_total}  ({100*r_pass/r_total:.0f}%)")
    print(f"  Retrieval quality:  {ret_pass}/{ret_total}  ({100*ret_pass/ret_total:.0f}%)")
    if hallucination[0] is not None:
        h_pass, h_total = hallucination
        print(f"  Hallucination safe: {h_pass}/{h_total}  ({100*h_pass/h_total:.0f}%)")
    else:
        print(f"  Hallucination:      skipped (use --no-skip-llm to enable)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval suite for breastfeeding support tool")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks and full LLM responses")
    parser.add_argument("--skip-llm", action="store_true", default=True,
                        help="Skip LLM hallucination checks (default: True — no Ollama needed)")
    parser.add_argument("--no-skip-llm", action="store_true", default=False,
                        help="Run LLM hallucination checks (requires Ollama running)")
    args = parser.parse_args()

    skip_llm = not args.no_skip_llm

    print("\n🔍  Loading knowledge base and building FAISS index...")
    t0 = time.time()
    chunks = load_knowledge()
    faiss_index, embeddings, chunks = build_index(chunks)
    print(f"    Loaded {len(chunks)} chunks in {time.time()-t0:.1f}s")

    routing_result = run_routing_eval(verbose=args.verbose)
    retrieval_result = run_retrieval_eval(faiss_index, chunks, verbose=args.verbose)

    client = None
    model_name = None
    if not skip_llm:
        try:
            import ollama
            client = ollama.Client()
            model_name = "llama3.2:3b"
            print(f"\n    LLM: {model_name} via Ollama")
        except Exception as e:
            print(f"\n    ⚠️  Could not connect to Ollama: {e}")
            skip_llm = True

    hallucination_result = run_hallucination_eval(
        faiss_index, chunks,
        client=client, model=model_name,
        verbose=args.verbose, skip_llm=skip_llm
    )

    print_summary(routing_result, retrieval_result, hallucination_result)