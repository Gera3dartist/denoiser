#!/usr/bin/env python3
"""
Eval harness for the email signal/noise filter (Phase 3 of the build plan).

What it does
------------
- Runs your agent over a labeled set of emails.
- The agent's DECISION is which tool it calls:
      preserve_signal -> "signal"     log_reason -> "noise"
- Reports overall accuracy, a 2x2 confusion matrix, a per-category breakdown
  (so you can see the purchase-vs-bill bug directly), and latency.

Usage
-----
    pip install ollama
    # warm the model once so timings exclude cold start:
    #   python -c "import ollama; ollama.chat(model='qwen3:4b', messages=[{'role':'user','content':''}], keep_alive=-1)"

    python eval_harness.py                         # runs the built-in seed set
    python eval_harness.py --data dataset.jsonl    # runs your labeled set
    python eval_harness.py --model qwen3:1.7b      # try a smaller model
    python eval_harness.py --think                 # enable thinking (default off)

dataset.jsonl format (one JSON object per line)
-----------------------------------------------
    {"id": "e1", "text": "<raw email>", "gold": "signal", "category": "bill"}

gold      : "signal" | "noise"
category  : bill | technical | purchase | notification | other
"""

import argparse
import json
import sys
import time
from collections import defaultdict

import ollama

# --------------------------------------------------------------------------
# Prompt under test. Edit this in ONE place; the harness measures the result.
# --------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an email filtering agent. Decide whether an email is SIGNAL or NOISE.

email_message: {email_message}

CORE TEST - apply this first:
  Does the email require the recipient to take a NEW action (pay a bill, respond, decide)?
  - Pending bill / obligation the recipient must still settle  -> SIGNAL
  - Record or confirmation of something already done           -> NOISE

SIGNAL (preserve):
- Utility/commodity bills still owed: electricity, water, heating, gas,
  rent, housing-office fees, municipal taxes, internet/phone service
- Technical info: mailing-list digests, dev/infra updates, release notes, security advisories
- Anything else demanding the recipient act, respond, or decide

NOISE (skip):
- Purchase activity of any kind - order confirmations, receipts, shipping/
  delivery updates, "payment received" - the action is already complete,
  even though money is mentioned
- Generic notifications: social, app, promo, "someone did X", newsletters

KEY DISTINCTION: a utility BILL is money you still owe (signal); a purchase
RECEIPT is money already spent (noise).

If SIGNAL - call preserve_signal with a short summary plus `from`, `topic`, `category`.
If NOISE - call log_reason with a one-sentence reason (<=10 words) plus `category`."""

# --------------------------------------------------------------------------
# Tool schemas. The `category` enum forces the model to commit to a type.
# --------------------------------------------------------------------------
CATEGORIES = ["bill", "technical", "purchase", "notification", "other"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "preserve_signal",
            "description": "Record an important email that requires the recipient's attention or action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "from": {"type": "string"},
                    "topic": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                },
                "required": ["summary", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_reason",
            "description": "Record why an email was skipped as noise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                },
                "required": ["reason", "category"],
            },
        },
    },
]

TOOL_TO_LABEL = {"preserve_signal": "signal", "log_reason": "noise"}

# --------------------------------------------------------------------------
# Built-in seed set: the hard pairs, so the harness runs out of the box.
# Replace with your own real emails via --data as soon as you can.
# --------------------------------------------------------------------------
SEED = [
    {"id": "s1", "category": "bill", "gold": "signal",
     "text": "ЖЕК: рахунок за опалення 1 240 грн, сплатіть до 20 червня."},
    {"id": "s2", "category": "bill", "gold": "signal",
     "text": "Oblenergo: electricity invoice for May, 612 UAH, due June 18."},
    {"id": "s3", "category": "bill", "gold": "signal",
     "text": "Водоканал: передайте показники лічильника до 15 числа."},
    {"id": "s4", "category": "purchase", "gold": "noise",
     "text": "Ваше замовлення Rozetka #5512 відправлено, очікуйте кур'єра."},
    {"id": "s5", "category": "purchase", "gold": "noise",
     "text": "Payment received - thank you for your order at Nova Shop!"},
    {"id": "s6", "category": "purchase", "gold": "noise",
     "text": "Your Amazon package was delivered. Rate your experience."},
    {"id": "s7", "category": "technical", "gold": "signal",
     "text": "[python-dev] Security advisory: CVE-2026-1234 in urllib3, upgrade now."},
    {"id": "s8", "category": "technical", "gold": "signal",
     "text": "Postgres 18.2 released - changelog and migration notes attached."},
    {"id": "s9", "category": "notification", "gold": "noise",
     "text": "John liked your photo. 3 new people viewed your profile."},
    {"id": "s10", "category": "notification", "gold": "noise",
     "text": "Weekly newsletter: 10 productivity hacks you must try!"},
    {"id": "s11", "category": "bill", "gold": "signal",
     "text": "Rent reminder: 9,000 UAH due to landlord by the 5th."},
    {"id": "s12", "category": "purchase", "gold": "noise",
     "text": "Order confirmed. We'll email you when it ships."},
]


def load_dataset(path):
    if not path:
        return SEED
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify(email_text, model, num_ctx, think):
    """Return (predicted_label, predicted_category, latency_s, gen_tokens)."""
    messages = [{"role": "user",
                 "content": PROMPT_TEMPLATE.format(email_message=email_text)}]
    t0 = time.perf_counter()
    resp = ollama.chat(
        model=model,
        messages=messages,
        tools=TOOLS,
        think=think,
        keep_alive=-1,
        options={"num_ctx": num_ctx, "temperature": 0},
    )
    latency = time.perf_counter() - t0

    pred_label, pred_cat = "error", "none"
    calls = resp.message.tool_calls or []
    if calls:
        name = calls[0].function.name
        args = calls[0].function.arguments or {}
        pred_label = TOOL_TO_LABEL.get(name, "error")
        pred_cat = args.get("category", "none")

    gen_tokens = getattr(resp, "eval_count", 0) or 0
    return pred_label, pred_cat, latency, gen_tokens


def report(results):
    n = len(results)
    correct = sum(1 for r in results if r["pred"] == r["gold"])
    errors = sum(1 for r in results if r["pred"] == "error")

    # 2x2 confusion matrix: rows = gold, cols = pred
    cm = defaultdict(int)
    for r in results:
        cm[(r["gold"], r["pred"])] += 1

    print("\n" + "=" * 56)
    print(f"  SAMPLES: {n}    ACCURACY: {correct}/{n} = {correct/n:.1%}"
          + (f"    UNPARSEABLE: {errors}" if errors else ""))
    print("=" * 56)

    print("\nConfusion matrix (rows = truth, cols = prediction)")
    labels = ["signal", "noise", "error"]
    header = "          " + "".join(f"{l:>9}" for l in labels)
    print(header)
    for g in ["signal", "noise"]:
        row = "".join(f"{cm[(g, p)]:>9}" for p in labels)
        print(f"{g:>9} {row}")

    # The two error types, named by cost.
    fn = cm[("signal", "noise")]            # missed a real signal (usually worst)
    fp = cm[("noise", "signal")]            # kept noise as signal (your current bug)
    print(f"\nMissed signals (signal->noise): {fn}   <- usually the costly one")
    print(f"Noise kept     (noise->signal): {fp}   <- your purchase-as-signal bug")

    # Per-category breakdown: where exactly is it failing?
    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        c = r.get("category", "unknown")
        by_cat[c][0] += 1
        by_cat[c][1] += int(r["pred"] == r["gold"])
    print("\nPer-category accuracy")
    for c in sorted(by_cat):
        tot, ok = by_cat[c]
        print(f"  {c:<13} {ok}/{tot} = {ok/tot:.0%}")

    # Latency
    lats = [r["latency"] for r in results]
    toks = [r["tokens"] for r in results]
    avg_lat = sum(lats) / n
    tot_tok = sum(toks)
    tot_lat = sum(lats)
    tps = (tot_tok / tot_lat) if tot_lat else 0
    print("\nLatency")
    print(f"  avg per email : {avg_lat:.2f} s")
    print(f"  p95           : {sorted(lats)[max(0, int(0.95*n)-1)]:.2f} s")
    print(f"  throughput    : {tps:.1f} generated tok/s")
    print("=" * 56 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to dataset.jsonl (default: built-in seed set)")
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--think", action="store_true", help="enable thinking mode (default: off, for latency)")
    args = ap.parse_args()

    data = load_dataset(args.data)
    print(f"Model: {args.model} | think={args.think} | num_ctx={args.num_ctx} "
          f"| {len(data)} emails ({'seed set' if not args.data else args.data})")

    results = []
    for row in data:
        try:
            pred, cat, lat, tok = classify(row["text"], args.model, args.num_ctx, args.think)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [{row['id']}] ERROR: {exc}", file=sys.stderr)
            pred, cat, lat, tok = "error", "none", 0.0, 0
        ok = "OK " if pred == row["gold"] else "XX "
        print(f"  {ok}{row['id']:>4}  gold={row['gold']:<6} pred={pred:<6} "
              f"cat={cat:<12} {lat:.2f}s")
        results.append({**row, "pred": pred, "pred_cat": cat,
                        "latency": lat, "tokens": tok})

    report(results)


if __name__ == "__main__":
    main()
