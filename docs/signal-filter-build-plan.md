# Email Signal Filter — Build Plan

**Guiding principle:** nail the *decision* and how you *measure* it first.
Model and latency are the **last** knobs, not the first. A fast wrong answer
is still wrong.

Work top to bottom. Each phase produces the input the next one needs.

---

## Phase 1 — Define the decision  ▢
*Goal: a one-sentence discriminating principle + a list of the look-alikes.*

- [ ] Write the signal/noise definition in prose (≤5 lines).
- [ ] State the **discriminating principle** in one sentence.
      _Current: "obligation still owed → signal vs action already completed → noise."_
- [ ] List the **hard pairs** — things that look alike but classify differently:
      - utility bill (signal) vs purchase receipt (noise)
      - meter-reading request (signal) vs shipping update (noise)
      - security advisory (signal) vs product newsletter (noise)
- [ ] Decide the **cost asymmetry**: is a missed signal worse than kept noise?
      Pick the tie-breaker now (default: *when uncertain → signal*).

**Done when:** a teammate could classify 10 emails using only your written rule.

---

## Phase 2 — Collect & label real data  ▢
*Goal: a small, honest, label-set that over-represents the confusing cases.*

- [ ] Pull 30–100 **real** emails (yours), Ukrainian + any English.
- [ ] Deliberately over-sample the hard pairs from Phase 1.
- [ ] Hand-label each: `gold` (signal/noise) + `category`
      (bill / technical / purchase / notification / other).
- [ ] Split: **dev set** (you read it constantly) + **held-out test set**
      (never tune against it).
- [ ] Save as JSONL — one object per line (see `eval_harness.py` format).

**Done when:** `dataset.jsonl` exists and you trust the labels.

---

## Phase 3 — Build the eval harness  ▢  ← most important step
*Goal: turn guessing into measurement. Do this BEFORE optimizing anything.*

- [ ] Run the system over the labeled set; capture which tool it called.
- [ ] Report: overall accuracy, **confusion matrix**, per-category breakdown.
- [ ] Surface your actual bug directly: *how many purchases were called signal?*
- [ ] Capture **latency** per email (Ollama `eval_count` / `eval_duration`).
- [ ] `eval_harness.py` is the scaffold — point it at your `dataset.jsonl`.

**Done when:** one command prints accuracy + confusion matrix + latency.

---

## Phase 4 — Establish a dumb baseline  ▢
*Goal: a reference number before any cleverness.*

- [ ] Run capable model + clear prompt, no tricks. Record the number.
- [ ] This is what every later change must beat on the **held-out** set.

**Done when:** you have a baseline accuracy you can quote.

---

## Phase 5 — Iterate on decision logic  ▢  ← where the accuracy lives
*Goal: fix the boundary in the prompt + structure, not the model.*

- [ ] Make SIGNAL/NOISE categories mutually exclusive (no shared trait).
- [ ] Add **contrastive examples** for each hard pair.
- [ ] Force a `category` enum in the tool call (commit to a type).
- [ ] Move obvious cases out of the LLM: utility-sender allowlist,
      retailer blocklist — decided in plain Python.
- [ ] Measure EVERY change on the held-out set. Keep what helps there,
      discard what only helps on dev (that's overfitting your own examples).

**Done when:** held-out accuracy clears your target and the purchase-vs-bill
confusion is gone.

---

## Phase 6 — Optimize model & latency  ▢  (only now)
*Goal: the smallest/fastest setup that holds Phase-5 accuracy.*

- [ ] Find the smallest model that keeps accuracy (`qwen3:4b` → try `1.7b`).
- [ ] Run **non-thinking** mode (`think=False` / `/no_think`).
- [ ] `keep_alive=-1` to kill cold starts; warm-up call at process start.
- [ ] `num_ctx=4096` — emails are small; don't pay for unused context.
- [ ] Re-run the eval to confirm the smaller/faster setup didn't regress.

**Done when:** latency target met AND accuracy unchanged vs Phase 5.

---

## Phase 7 — Failure modes & long tail  ▢
- [ ] Implement the uncertainty default (Phase 1 tie-breaker).
- [ ] Handle malformed / missing tool calls and model timeouts.
- [ ] Route genuinely ambiguous emails to a human instead of dropping them.

---

## Phase 8 — Production & feedback loop  ▢
- [ ] Log every decision + reason + category (you already emit `log_reason`).
- [ ] Periodically sample real decisions, re-label, fold back into the eval set.
- [ ] Re-run the eval on every prompt/model change as a regression guard.
- [ ] Watch for drift: new senders, new email formats.

---

## Phase 9 — Fine-tune (only if Phase 5 plateaus)  ▢
- [ ] If prompt + examples can't reach target, fine-tune a small specialist
      (e.g. FunctionGemma) on your labeled data.
- [ ] Last resort: most effort, and it depends on Phases 2–3 already existing.

---

### The through-line
labeled data → honest eval → real accuracy ceiling → smallest affordable model
→ production logs → refill the dataset. Skip the early phases and you tune blind.
