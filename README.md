# Agentic RFP Evaluation and Supplier Ranking

A beginner-friendly Streamlit application for evaluating supplier RFP response PDFs against configurable criteria. It uses an LLM for evidence-based criterion scoring, Python for validation and deterministic calculations, and SQLite for persistence.

## Features
- Upload up to six supplier proposal PDFs
- Enter supplier name, submission date, and historical experience rating
- Extract PDF text with page numbers
- Load active criteria and weights from SQLite
- Ask an LLM for criterion-wise scores, justifications, evidence, risks, and summary
- Validate missing or malformed criterion results and clip scores to allowed ranges
- Calculate absolute weighted score, criterion benchmarks, gaps, relative performance, and Peer Performance Index (PPI)
- Rank deterministically using PPI, submission date, experience rating, and supplier name
- Store completed results in SQLite and download the result as JSON
- Four synthetic supplier proposal PDFs are included in `sample_pdfs/`

## Requirements
- Python 3.10+
- An API key for a compatible OpenAI model
- Internet access for the model API

## Setup

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

Keys can be entered at runtime in the sidebar under **LLM Settings**, so a `.env` file is
optional. To pre-fill a key, copy `.env.example` to `.env` and set the variable matching your
provider:

```text
# any one of these is picked up automatically for its provider
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
OPENROUTER_API_KEY=
COHERE_API_KEY=
MISTRAL_API_KEY=
GROQ_API_KEY=
DEEPSEEK_API_KEY=

# optional default model per provider
OPENAI_MODEL=gpt-4.1-mini
```

### Choosing a provider
The sidebar **LLM Settings** panel selects the configuration provider (OpenAI, Anthropic,
Google Gemini, OpenRouter, Cohere, Mistral AI, Groq, DeepSeek, or any custom
OpenAI-compatible endpoint). The model dropdown is populated from that provider's defaults and
can be refreshed with **↻ Models** to list exactly what your key can access. **✓ Test** sends a
one-token request to confirm the key, model and endpoint work. Keys typed in the sidebar live in
the browser session only and are never written to disk.

Initialize the SQLite database and seed default criteria:

```bash
python -c "from database import init_db; init_db(); print('Database ready')"
```

Start the app:

```bash
streamlit run app.py
```

## Default criteria
| Criterion | Weight |
|---|---:|
| Technical Capability | 30% |
| Implementation Plan | 20% |
| Commercial Value | 20% |
| Security & Compliance | 20% |
| Support & Experience | 10% |
| **Total** | **100%** |

To modify criteria, edit `DEFAULT_CRITERIA` in `database.py`. For a fresh seed, remove `rfp_evaluation.db` before reinitializing. In a production version, add an authenticated criteria editor.

## Scoring formulas
- Absolute weighted score: sum of `(criterion score / max score) × criterion weight`; displayed on a 0–10 scale.
- Criterion benchmark: highest specified score observed among suppliers for that criterion.
- Criterion gap: supplier score minus benchmark.
- Relative performance: `(supplier score / benchmark) × 100`. When benchmark is zero, the implementation uses a safe defined fallback.
- PPI: weighted average of criterion relative-performance percentages.
- Ranking: PPI descending, submission date ascending, historical experience descending, supplier name ascending.

## Notes and limitations
- This MVP uses direct document text in the LLM request, with a character cap. RAG/vector search is not required by the attached assignment and is a recommended extension for long documents.
- Scanned PDFs may need OCR; this MVP reports an error when no text can be extracted.
- The app sends proposal text to the configured LLM API. Do not use confidential supplier documents unless permitted by your organization's policies.
- LLM scores are advisory. Review the evidence and justifications before making procurement decisions.
- The PPI formula can behave differently from absolute weighted score. It is implemented as specified in the project brief.
- The starter UI displays the current/latest run in the session. Results are persisted in SQLite; a historical run browser can be added as an extension.
- Never commit `.env` or API keys to source control.

## Suggested testing
1. Run with the four synthetic PDFs in `sample_pdfs/`.
2. Try a malformed or scanned PDF and confirm the error is surfaced.
3. Inspect the JSON result and verify each active criterion is present.
4. Test weight validation by changing a weight so the active weights no longer total 100%.
5. Check tie-breaks using identical PPI values and different dates/experience/name values.
6. Compare the output against a manually reviewed scoring sheet.

## Deployment
For Streamlit Community Cloud, push the project to a private/public Git repository as appropriate, configure the API key variable for your provider (for example `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) in the app's Secrets settings, and deploy `app.py`. SQLite files on ephemeral cloud storage may not provide durable persistence across restarts; use a managed database for production.
