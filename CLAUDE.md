# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project-Specific: Customer LTV (Olist)

### Architecture
- `src/data.py`     — load CSVs, build ABT, compute targets, split
- `src/features.py` — RFM, behavioral, temporal features (all leak-free)
- `src/model.py`    — two-stage model: Stage 1 classifier + Stage 2 regressor/segmenter
- `src/predict.py`  — inference pipeline (must mirror training features exactly)
- `src/api.py`      — FastAPI serving layer

### Key constraints
- **Leak-free features**: every feature must use only orders strictly before `cutoff_date`.
  Violation here silently inflates metrics and breaks production.
- **Cutoff date**: `2018-07-01` (set in config.yaml). Prediction window: 90 days after.
- **ABT is cached**: load from `data/processed/abt.parquet` — don't re-merge from CSVs unless schema changes.
- **Two-stage targets**: `will_return` (Stage 1, binary) and `future_spend` (Stage 2, continuous).
  Both computed in `compute_ltv_targets()` in data.py.
- **Chronological split**: train/test split by customer first order date — no random shuffling.

### Data facts (know before touching the model)
- 96K delivered orders, 93K unique customers
- Cutoff-eligible customers: ~81K
- Returners (will_return=1): ~280 (0.3%) — extremely imbalanced
- Stage 2 training set: ~197 returners only
- Average future spend (returners): ~$150

### Don't do
- Don't use `customer_id` as the customer key — use `customer_unique_id` (stable across orders).
- Don't include post-cutoff orders in feature computation.
- Don't random-shuffle before splitting.
- Don't save raw CSVs or trained models to git (both are gitignored).
