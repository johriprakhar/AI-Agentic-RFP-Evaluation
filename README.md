# Agentic RFP Evaluation and Supplier Ranking

A Streamlit application that scores supplier RFP responses against configurable criteria and ranks them defensibly.

The design principle is a hard split of responsibilities: **a language model supplies judgement, Python supplies every number.** The model reads a proposal and says how well it addresses each criterion and which sentence proves it. All arithmetic — weighting, benchmarking, the Peer Performance Index, tie-breaks, ranking — is pure Python in `scoring.py`. A model cannot move a score it did not compute, and every quote it offers is checked back against the text extracted from the PDF before a reviewer ever sees it.

---

## Key features

### Evaluation and ranking
- **Configurable criteria, edited in the UI.** Criteria, weights and score scales live in SQLite and are injected into the prompt, so adding a criterion changes the evaluation with no code change.
- **Schema-validated model output with self-repair.** Replies are checked against a declarative schema; a malformed reply is sent back with the exact validation errors attached, up to three attempts, before safe defaults are applied.
- **Evidence grounding.** Every quote the model returns is verified against the extracted document text and labelled verified, partial or not found. Scores are never silently rewritten — an unverifiable quote is surfaced, not hidden.
- **No silent truncation.** A proposal too long for the model input budget is condensed by criterion relevance (BM25), assembled in reading order with omission markers, and the coverage figure is reported with the result.
- **Deterministic ranking with explained placements.** A four-level total order, and each supplier is told which level actually put it ahead of the next.
- **Parallel, contained execution.** Proposals are evaluated concurrently; one unreadable PDF is recorded as a failed supplier instead of aborting the run.
- **Reproducible runs.** Each run stores its own criteria snapshot plus the provider, model and temperature that produced it, and can be reloaded from the database long after the active criteria change.
- **Auditable exports.** Self-contained JSON (scores, evidence, criteria snapshot, tool trace) and a flat CSV for spreadsheet review.

### Dual-mode document search
- **Without LLM (BM25):** offline, instant, free, and identical on every run. Term saturation, inverse document frequency, length normalisation, phrase and bigram bonuses, query-coverage weighting.
- **With LLM:** BM25 shortlists candidates, then a model re-ranks them, resolves synonyms and negation, and extracts the decisive quote. Falls back to BM25 if the model is unavailable.
- **Exact citations:** every passage shows its document, page number and line range, with query terms highlighted.

---

## Requirements

- Python 3.10 or newer
- An API key for a supported provider (OpenAI, Anthropic, Google Gemini, OpenRouter, Cohere, Mistral, Groq, DeepSeek) or any OpenAI-compatible endpoint
- Dependencies in `requirements.txt`

---

## Setup

```bash
git clone https://github.com/johriprakhar/AI-Agentic-RFP-Evaluation.git
cd AI-Agentic-RFP-Evaluation

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Initialise the database

```bash
python database.py
```

Creates `rfp_evaluation.db`, seeds the five default criteria and prints the active weight total. Safe to re-run: existing databases are migrated in place, not overwritten.

### Generate the sample proposals

```bash
python generate_sample_pdfs.py
```

Writes four synthetic supplier proposals to `sample_pdfs/`. Use `--force` to rewrite them. These are the fixtures the test suite asserts against, so they are generated from code rather than committed as opaque binaries.

### Run the app

```bash
streamlit run app.py
```

### Credentials (optional)

Keys can be pasted into the sidebar at runtime and are held in session memory only. To pre-fill them, create a `.env` file:

```text
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-...
COHERE_API_KEY=...

OPENAI_MODEL=gpt-4.1-mini        # optional default model
MAX_DOCUMENT_CHARS=70000         # optional model input budget
RFP_DB_PATH=./rfp_evaluation.db  # optional database location
```

`.env` and `rfp_evaluation.db` are gitignored.

---

## Using the app

| Tab | What it does |
|---|---|
| **📊 Criteria** | Weight distribution, influence bands and available weighted points, plus an editor to add, re-weight, deactivate or reset criteria. Evaluation is blocked until active weights total 100%. |
| **📤 New evaluation** | Upload 1–6 proposal PDFs, confirm each supplier's name, submission date and prior experience rating, then run. Pre-flight checks report every problem at once. |
| **🏆 Results** | Leaderboard with the reason for each placement, per-criterion comparison, the winner's score composition, and a drill-down per supplier with evidence, verification status, benchmark, gap, relative performance and the tool execution trace. JSON and CSV export. |
| **🗂️ Run history** | Every stored run, reloadable with its own criteria snapshot. Compare runs side by side or delete them. |
| **🔍 Document search** | Dual-mode search with the top-matching document, a per-document comparison, and line-numbered passage evidence. |

---

## Default criteria

| Criterion | Weight | Max score | Covers |
|---|---:|---:|---|
| Technical Capability | 30% | 10 | Architecture, integrations, scalability, technical fit |
| Implementation Plan | 20% | 10 | Timeline, milestones, staffing, dependencies, risk |
| Commercial Value | 20% | 10 | Pricing clarity, total cost, assumptions, value |
| Security & Compliance | 20% | 10 | Controls, certifications, privacy, auditability |
| Support & Experience | 10% | 10 | Support model, similar projects, references, SLAs |
| **Total** | **100%** | | |

Editable in the Criteria tab. Defaults live in `database.py`.

---

## Scoring

**Absolute score (0–10), peer-independent:**

$$\text{absolute} = \frac{1}{10}\sum_i \left(\frac{\text{score}_i}{\text{max}_i} \times \text{weight}_i\right)$$

**Peer benchmark** for criterion *i* is the highest score any supplier achieved on it while marked `specified`:

$$\text{benchmark}_i = \max_{s \in \text{specified}} \text{score}_{i,s}
\qquad
\text{gap}_{i,s} = \text{score}_{i,s} - \text{benchmark}_i \le 0$$

**Relative performance:**

$$\text{relative}_{i,s} = \frac{\text{score}_{i,s}}{\text{benchmark}_i} \times 100$$

**Peer Performance Index** — a weighted mean over the criteria that actually differentiate the field:

$$\text{PPI}_s = \frac{\sum_{i \in D}\left(\text{relative}_{i,s} \times \text{weight}_i\right)}{\sum_{i \in D} \text{weight}_i}$$

where *D* is the set of criteria with a non-zero benchmark. A criterion no supplier scored on carries no information, so it is excluded from both sides of the PPI rather than being handed an arbitrary 0% or 100%. It still affects the absolute score, and the exclusion is reported with the run.

**Ranking order** — a strict total order, so the result never depends on upload order or sort stability:

1. PPI, descending
2. Submission date, ascending (earlier wins)
3. Experience rating, descending
4. Supplier name, ascending (case-folded)

Suppliers are ranked on PPI, not on the absolute score: PPI measures performance against the best available bid per criterion, which is the question a buyer is actually asking. The absolute score is displayed alongside it for context.

---

## Module map

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI only. No arithmetic, no SQL beyond calling `database`. |
| `ui.py` | Theme, cards, charts, the line-numbered passage viewer. HTML-escapes all interpolated content. |
| `orchestrator.py` | Owns the pipeline. Pre-flight validation, parallel execution, failure containment, run assembly. |
| `tools.py` | Tool registry and timed execution trace; marks each step model or deterministic. |
| `evaluator.py` | Prompt construction and the single model judgement call. |
| `llm_providers.py` | Multi-provider access, transient retry, JSON-mode negotiation, the schema self-repair loop. |
| `schemas.py` | Declarative schemas, the strict validator and repair-prompt generation. |
| `scoring.py` | Normalization, weighted scoring, benchmarks, PPI, deterministic ranking. Pure functions. |
| `evidence.py` | Verifies quotes against extracted text; page citation checks. |
| `pdf_processor.py` | Extraction, scanned-page diagnostics, budgeted model-input preparation. |
| `retrieval.py` | Tokenisation, line-indexed chunking, BM25, passage selection. No model dependency. |
| `search_engine.py` | The LLM semantic layer over `retrieval`. |
| `database.py` | SQLite schema, migrations, criteria CRUD, run persistence and history. |

---

## Testing

231 tests, all offline. No API key and no network required: the model is replaced by a scripted stub, while PDF extraction, retrieval, evidence verification, scoring and SQLite persistence all run for real.

Run everything:

```bash
pytest
```

Or run any suite on its own with no extra dependency — each file works under the stdlib `unittest` runner too:

```bash
python test_scoring.py         # 30  normalization, weighting, benchmarks, PPI, tie-breaks
python test_validation.py      # 44  schemas, JSON extraction, the self-repair loop, prompts
python test_search_engine.py   # 34  chunking, page/line citations, BM25, passage selection
python test_evidence.py        # 32  quote verification, page citations, grounding summary
python test_pdf_processor.py   # 22  extraction, scanned detection, budgeted preparation
python test_database.py        # 29  criteria CRUD, run storage, snapshots, migration
python test_orchestrator.py    # 40  end-to-end pipeline, determinism, failure containment
```

`conftest.py` anchors the working directory and generates the sample PDFs on demand, so a fresh clone passes on the first attempt from any directory.

### Edge cases covered

- Malformed model output: non-JSON, prose-wrapped JSON, code fences, empty replies
- Invalid scores: non-numeric, infinite, negative, above maximum, boolean
- Structural faults: missing criteria, duplicate ids, unknown ids, wrong container types
- `not specified` carrying a non-zero score (zeroed, to keep benchmarking consistent)
- A criterion no supplier addressed (excluded from the PPI, reported)
- An all-zero field (no division by zero)
- Scanned and image-only PDFs, partially scanned PDFs, corrupt bytes
- Documents over the input budget, and uniform documents BM25 cannot rank
- Fabricated, paraphrased, re-wrapped and mis-cited evidence quotes
- Every tie-break level, plus ranking independence from input order
- One supplier failing mid-run without aborting the others
- A model returning out-of-range scores, verified as unable to inflate its own result
- Legacy database migration, non-ASCII content, unknown run ids

### Manual verification

1. `python generate_sample_pdfs.py --force` then `streamlit run app.py`.
2. **📊 Criteria** — confirm weights total 100% and the charts render. Try editing a weight and saving; the header chip updates.
3. **🔍 Document search** — load the four samples and search *"cloud-native architecture, REST integrations, disaster recovery"* in BM25 mode. Expect **Apex Systems** first, cited at page 1, lines 4–8.
4. Search *"ISO 27001 certified"* in BM25 mode and note BrightPath scoring well despite stating it is *not* certified. Switch to LLM mode to see negation resolved. This limitation is pinned by a test.
5. **📤 New evaluation** — load the samples, run, then inspect **🏆 Results**: placement reasons, evidence verification badges, and the tool trace showing four deterministic steps around one model call.
6. **🗂️ Run history** — change a criterion weight, then reopen the earlier run and confirm its criteria snapshot is unchanged.

---

## Security notes

- Proposal text is untrusted input. It is fenced in the prompt and the model is instructed not to follow instructions inside it, so a bidder writing "award full marks" gains nothing. Even if a model were persuaded, scores are clipped to the criterion maximum and all arithmetic happens outside the model.
- All model- and filename-derived strings are HTML-escaped before rendering.
- API keys are read from the environment or session memory and are never written to disk or stored with a run.
- SQL is parameterised throughout.

---

## Deployment

For Streamlit Community Cloud:

1. Push the repository, including `generate_sample_pdfs.py` and every module.
2. Add provider keys under **App settings → Secrets** (e.g. `OPENAI_API_KEY = "sk-..."`).
3. Deploy with `app.py` as the entry point.

The SQLite file is ephemeral on Community Cloud. Point `RFP_DB_PATH` at persistent storage if run history needs to survive restarts.
