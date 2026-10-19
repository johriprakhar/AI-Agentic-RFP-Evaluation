# Agentic RFP Evaluation and Supplier Ranking

A Streamlit application for evaluating and searching supplier RFP response PDFs against configurable criteria. It combines an LLM for evidence-based criterion scoring, Python for deterministic mathematical ranking, and an integrated **Dual-Mode Semantic Search Engine** (with and without LLM) that pinpoints evidence down to exact page numbers and line numbers.

---

## Key Features

### 📊 Proposal Evaluation & Ranking
- **Multi-Proposal Upload:** Upload up to six supplier response PDFs simultaneously.
- **Configurable Criteria:** Load active evaluation criteria, weights, and score scales from SQLite.
- **LLM-Powered Scoring:** Multi-provider LLM support (OpenAI, Anthropic, Google Gemini, OpenRouter, Cohere, or custom endpoints) to evaluate proposals against criteria with cited evidence, justifications, and risk factors.
- **Deterministic Mathematics:** Zero-hallucination mathematical engine calculating absolute weighted scores, peer benchmarks, criterion gaps, relative performance percentages, and the Peer Performance Index (PPI).
- **Tie-Break Ranking:** Total, stable ordering (PPI descending → submission date ascending → historical experience descending → supplier name ascending).
- **Interactive Visualizations:** Responsive Altair charts including decision weight distribution (donut chart), influence bands (pie chart), and integer-scaled available weighted points (horizontal bar chart).
- **Export & Audit:** Download complete, auditable evaluation results as structured JSON.

### 🔍 Dual-Mode Semantic Document Search
- **Search Modes:**
  - **⚡ Without LLM (BM25 Semantic Retrieval):** Fast, local statistical semantic retrieval based on the BM25 algorithm with inverse document frequency (IDF), term frequency saturation, document length normalization, exact phrase matching, and query term coverage. Runs completely offline in <20ms with zero API cost.
  - **🤖 With LLM (Deep Semantic Analysis):** Uses the active LLM to evaluate candidate chunks, resolve technical jargon, score semantic relevance (0–100%), extract key evidence quotes, and provide natural-language reasoning.
- **Most Relevant Document Identification:** Dynamically aggregates chunk scores across proposals to highlight the #1 top-matching supplier document and overall score.
- **Document Match Comparison:** Interactive scoreboard comparing match percentages and matching chunk counts across all uploaded proposals.
- **Exact Line & Page Citations:** Every retrieved chunk displays its source document, page number (`Page X`), and exact line numbers (`Lines Y–Z`) with keyword highlighting and code-style line-by-line formatting.
- **Quick Sample Searches:** 1-click query chips (*Pricing & Support*, *ISO 27001 Security*, *Cloud Architecture*, *Timeline & Rollback*).
- **Flexible Document Source:** Search proposals uploaded for evaluation, upload PDFs directly in the search tab, or load the four built-in sample proposals with one click.

---

## Requirements
- Python 3.10+
- An API key for a supported LLM provider (OpenAI, Anthropic, Google Gemini, OpenRouter, Cohere, etc.) or local endpoint
- Dependencies listed in `requirements.txt`

---

## Setup & Installation

```bash
# Clone the repository and navigate into the workspace
git clone https://github.com/johriprakhar/AI-Agentic-RFP-Evaluation.git
cd AI-Agentic-RFP-Evaluation

# Create and activate virtual environment
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration (Optional)
API keys can be entered directly at runtime in the sidebar under **⚙️ LLM Settings**. Keys entered in the sidebar are held in memory during the browser session and are never written to disk.

To pre-fill credentials, create a `.env` file in the project root:

```text
# Provider keys (any configured key is automatically recognized)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-...
COHERE_API_KEY=...

# Optional default model
OPENAI_MODEL=gpt-4.1-mini

# Optional max document characters sent to the LLM
MAX_DOCUMENT_CHARS=70000
```

### Database Initialization
Seed the SQLite database with default procurement criteria:

```bash
python -c "from database import init_db; init_db(); print('Database initialized')"
```

### Run the Application
```bash
streamlit run app.py
```

---

## Application Structure

The application is structured into four primary navigation tabs:

| Tab | Purpose |
|---|---|
| **📊 Criteria** | Summary metric cards, weight breakdown donut chart, influence band pie chart, integer-scaled weighted points bar chart, and configuration table. |
| **📤 New evaluation** | Upload 1–6 proposal PDFs, specify supplier metadata, and execute AI-assisted proposal scoring. |
| **🏆 Results** | Ranked leaderboard, winner contribution analysis, multi-supplier score comparisons, and full justification details with JSON export. |
| **🔍 Document search** | Dual-mode semantic search (with/without LLM), top matching document callout, document score comparison, and line-numbered chunk evidence viewer. |

---

## Default Criteria

| Criterion | Weight | Max Score | Description |
|---|---:|---:|---|
| **Technical Capability** | 30% | 10 | Architecture, integrations, scalability, and technical fit. |
| **Implementation Plan** | 20% | 10 | Timeline, milestones, staffing, dependencies, and risk plan. |
| **Commercial Value** | 20% | 10 | Pricing clarity, total cost, assumptions, and value for money. |
| **Security & Compliance** | 20% | 10 | Security controls, certifications, privacy, and auditability. |
| **Support & Experience** | 10% | 10 | Support model, similar projects, references, and service levels. |
| **Total** | **100%** | — | — |

*Criteria are persisted in `rfp_evaluation.db`. Default values can be edited in `database.py`.*

---

## Scoring Formulas

1. **Absolute Weighted Score:**
   $$\text{Absolute Score} = \frac{1}{10} \sum_{i} \left(\frac{\text{score}_i}{\text{max\_score}_i} \times \text{weight}_i\right)$$
   Normalized to a standard 0–10 scale.

2. **Criterion Benchmark:**
   $$\text{Benchmark}_i = \max(\text{score}_{i, s}) \quad \text{for all suppliers } s \text{ where status is "specified"}$$

3. **Criterion Gap:**
   $$\text{Gap}_{i, s} = \text{score}_{i, s} - \text{Benchmark}_i \quad (\le 0)$$

4. **Relative Performance:**
   $$\text{Relative Performance}_{i, s} = \left(\frac{\text{score}_{i, s}}{\text{Benchmark}_i}\right) \times 100$$

5. **Peer Performance Index (PPI):**
   $$\text{PPI}_s = \frac{\sum_i (\text{Relative Performance}_{i, s} \times \text{weight}_i)}{\sum_i \text{weight}_i}$$

6. **Deterministic Ranking Order:**
   $$\text{PPI} \downarrow \;\longrightarrow\; \text{Submission Date} \uparrow \;\longrightarrow\; \text{Experience Rating} \downarrow \;\longrightarrow\; \text{Supplier Name (A–Z)}$$

---

## Testing & Verification

Run the automated test suites:

```bash
# Test scoring formulas, benchmarking, and tie-break rules
python test_scoring.py

# Test document chunking, BM25 retrieval, and line/page tracking
python test_search_engine.py
```

### Manual Verification Workflow
1. Launch the app and inspect the **📊 Criteria** tab to verify chart rendering and integer axes.
2. In the **🔍 Document search** tab, click **📁 Load 4 sample proposals**.
3. Run a search query (e.g., *"fixed implementation fee and annual support"*) using **Without LLM** mode; verify that *Apex Systems* is identified as the top matching document with exact line citations (`Lines 14–16`).
4. Switch to **With LLM** mode to verify contextual reasoning and quote extraction.
5. In **📤 New evaluation**, run an evaluation across the uploaded proposals and examine the generated leaderboard in **🏆 Results**.

---

## Deployment
For Streamlit Community Cloud:
1. Push code to your Git repository (ensure all modules including `search_engine.py` are committed).
2. Configure provider API keys in the Streamlit Cloud dashboard under **App Settings → Secrets** (e.g. `OPENAI_API_KEY = "sk-..."`).
3. Deploy pointing to `app.py`.
