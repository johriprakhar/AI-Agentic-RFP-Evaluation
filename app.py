import glob
import json
import os
from datetime import date

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import ui
from database import init_db, get_active_criteria, save_run
from evaluator import evaluate_supplier
from llm_providers import (
    CUSTOM_MODEL_OPTION,
    DEFAULT_PROVIDER,
    env_api_key,
    env_model,
    fallback_models,
    fetch_models,
    provider_config,
    provider_names,
    resolve_base_url,
    test_connection,
)
from pdf_processor import extract_pdf_pages
from scoring import validate_and_normalize, calculate_supplier_metrics, rank_suppliers
from search_engine import extract_document_chunks, search_without_llm, search_with_llm, tokenize

load_dotenv()
init_db()

st.set_page_config(
    page_title="Agentic RFP Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_theme()


# ----------------------------------------------------------------------------
# Sidebar: LLM Settings
# ----------------------------------------------------------------------------
def render_llm_settings():
    """Provider / model / API-key picker. Returns the active llm_config dict."""
    st.markdown(
        """
        <div class="rfp-sidebar-brand">
            <div class="rfp-brand-title">📊 RFP Evaluator</div>
            <div class="rfp-brand-sub">Evidence-backed supplier scoring</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### ⚙️ LLM Settings")

    provider = st.selectbox(
        "Configuration provider",
        provider_names(),
        index=provider_names().index(DEFAULT_PROVIDER),
        key="llm_provider",
        help="Pick the API vendor that will score the proposals.",
    )
    config = provider_config(provider)

    base_url_input = ""
    if config.get("requires_base_url"):
        base_url_state = f"base_url::{provider}"
        base_url_input = st.text_input(
            "Base URL",
            key=base_url_state,
            placeholder="https://your-endpoint/v1",
            help="Any endpoint that implements POST /chat/completions.",
        )

    # --- API key -------------------------------------------------------------
    key_state = f"api_key::{provider}"
    if key_state not in st.session_state:
        st.session_state[key_state] = env_api_key(provider)
    api_key = st.text_input(
        f"{provider} API key",
        type="password",
        key=key_state,
        placeholder=config.get("key_prefix_hint", ""),
        help="Stored only in this browser session; never written to disk.",
    ).strip()
    if config.get("keys_url"):
        st.caption(f"Get a key: {config['keys_url']}")

    # --- Model dropdown ------------------------------------------------------
    models_state = f"models::{provider}"
    fetched = st.session_state.get(models_state)
    available = list(fetched) if fetched else fallback_models(provider)
    options = available + [CUSTOM_MODEL_OPTION]

    model_state = f"model::{provider}"
    if model_state not in st.session_state:
        preferred = env_model(provider)
        st.session_state[model_state] = preferred if preferred in options else options[0]
    if st.session_state[model_state] not in options:
        st.session_state[model_state] = options[0]

    selected = st.selectbox("Model", options, key=model_state)
    model = selected
    if selected == CUSTOM_MODEL_OPTION:
        model = st.text_input(
            "Model id",
            key=f"custom_model::{provider}",
            placeholder="exact model identifier",
        ).strip()

    load_col, test_col = st.columns(2)
    if load_col.button("↻ Models", width="stretch", help="Load the model list this key can access."):
        if not api_key:
            st.session_state[f"models_msg::{provider}"] = ("warn", "Enter an API key first.")
        else:
            try:
                live = fetch_models(provider, api_key, base_url_input)
                if live:
                    st.session_state[models_state] = live
                    st.session_state[f"models_msg::{provider}"] = ("ok", f"Loaded {len(live)} models.")
                else:
                    st.session_state[f"models_msg::{provider}"] = ("warn", "Provider returned no chat models.")
            except Exception as error:
                st.session_state[f"models_msg::{provider}"] = ("warn", f"Could not load models: {error}")
            st.rerun()

    if test_col.button("✓ Test", width="stretch", help="Send a one-token request to verify the setup."):
        try:
            reply = test_connection({
                "provider": provider,
                "model": model,
                "api_key": api_key,
                "base_url": resolve_base_url(provider, base_url_input),
            })
            st.session_state[f"models_msg::{provider}"] = ("ok", f"Connection OK — model replied: {reply[:40]}")
        except Exception as error:
            st.session_state[f"models_msg::{provider}"] = ("warn", f"Connection failed: {error}")

    message = st.session_state.get(f"models_msg::{provider}")
    if message:
        kind, text = message
        (st.success if kind == "ok" else st.warning)(text)

    if fetched:
        st.caption(f"Model list loaded live from {provider}.")
    elif available:
        st.caption("Showing built-in defaults. Use ↻ Models to load the live list.")

    with st.expander("Advanced"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1,
                                help="Keep at 0 for repeatable scoring.")
        max_chars = st.number_input(
            "Max characters per document", min_value=5000, max_value=400000,
            value=int(os.getenv("MAX_DOCUMENT_CHARS", "70000")), step=5000,
            help="Longer proposals are truncated before being sent to the model.",
        )

    base_url = resolve_base_url(provider, base_url_input)
    ready = bool(api_key) and bool(model) and (not config.get("requires_base_url") or bool(base_url))
    st.divider()
    if ready:
        ui.pill(f"● Ready — {provider}", "ok")
    elif not api_key:
        ui.pill("● API key required", "warn")
    elif not model:
        ui.pill("● Select a model", "warn")
    else:
        ui.pill("● Base URL required", "warn")

    llm_config = {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_document_chars": int(max_chars),
        "ready": ready,
    }
    st.session_state["llm_config"] = llm_config
    return llm_config


with st.sidebar:
    llm_config = render_llm_settings()


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
criteria = get_active_criteria()
total_weight = sum(float(c["weight"]) for c in criteria) if criteria else 0.0

ui.hero(
    "Agentic RFP Evaluation &amp; Supplier Ranking",
    "Score supplier proposals against configurable criteria with evidence-backed, auditable results.",
    chips=[
        f"🤖 {llm_config['provider']}",
        f"🧠 {llm_config['model'] or 'no model selected'}",
        f"🎯 {len(criteria)} active criteria",
        f"⚖️ {total_weight:.0f}% total weight",
    ],
)

tab_criteria, tab_eval, tab_results, tab_search = st.tabs(
    ["📊 Criteria", "📤 New evaluation", "🏆 Results", "🔍 Document search"]
)


# ----------------------------------------------------------------------------
# Tab 1 — Criteria (with donut, pie and bar views)
# ----------------------------------------------------------------------------
def weight_band(weight):
    if weight >= 25:
        return "High influence (≥25%)"
    if weight >= 15:
        return "Medium influence (15–25%)"
    return "Supporting (<15%)"


with tab_criteria:
    if not criteria:
        st.warning("No active criteria found. Run `python database.py` to seed the defaults.")
    else:
        criteria_df = pd.DataFrame([{
            "ID": c["criterion_id"],
            "Criterion": c["name"],
            "Description": c["description"],
            "Weight": float(c["weight"]),
            "Max score": float(c["max_score"]),
            "Band": weight_band(float(c["weight"])),
            "Weighted points": round(float(c["weight"]) * float(c["max_score"]) / 100.0, 2),
        } for c in criteria])

        heaviest = criteria_df.sort_values("Weight", ascending=False).iloc[0]
        cards = st.columns(4)
        with cards[0]:
            ui.metric_card("Active criteria", len(criteria_df), "Applied to every supplier", ui.ACCENTS[0])
        with cards[1]:
            ui.metric_card(
                "Total weight", f"{total_weight:.1f}%",
                "Balanced" if abs(total_weight - 100) <= 0.001 else "Must total 100%",
                ui.ACCENTS[4] if abs(total_weight - 100) <= 0.001 else "#EF4444",
            )
        with cards[2]:
            ui.metric_card("Score scale", f"0 – {criteria_df['Max score'].max():.0f}",
                           "Per-criterion raw score", ui.ACCENTS[1])
        with cards[3]:
            ui.metric_card("Heaviest criterion", heaviest["Criterion"],
                           f"{heaviest['Weight']:.1f}% of the decision", ui.ACCENTS[3])

        if abs(total_weight - 100) > 0.001:
            st.error(f"Active criteria weights total {total_weight:.1f}%, not 100%. "
                     "Evaluation is blocked until the weights are corrected in the database.")

        st.write("")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            ui.section("Weight distribution", ui.ACCENTS[0])
            st.altair_chart(
                ui.donut_chart(criteria_df, "Criterion", "Weight",
                               "Share of the final decision (%)", "Weight %"),
                width="stretch",
            )
        with chart_right:
            ui.section("Influence bands", ui.ACCENTS[2])
            band_df = (
                criteria_df.groupby("Band", as_index=False)
                .agg(Weight=("Weight", "sum"), Criteria=("Criterion", "count"))
            )
            st.altair_chart(
                ui.pie_chart(band_df, "Band", "Weight",
                             "Weight grouped by influence band (%)", "Weight %"),
                width="stretch",
            )

        ui.section("Weighted points available per criterion", ui.ACCENTS[1])
        st.altair_chart(
            ui.hbar_chart(criteria_df, "Criterion", "Weighted points",
                          "Maximum weighted points a supplier can earn", "Weighted points", ".0f"),
            width="stretch",
        )

        ui.section("Criteria configuration", ui.ACCENTS[5])
        st.dataframe(
            criteria_df[["ID", "Criterion", "Description", "Weight", "Max score", "Band"]],
            hide_index=True,
            width="stretch",
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Weight": st.column_config.ProgressColumn(
                    "Weight (%)", min_value=0, max_value=100, format="%.1f%%",
                    color=ui.ACCENTS[0],
                ),
                "Max score": st.column_config.NumberColumn("Max score", format="%.0f"),
            },
        )
        st.caption("Criteria live in SQLite. Edit the seed values in `database.py` or the database "
                   "directly to change active criteria and weights.")


# ----------------------------------------------------------------------------
# Tab 2 — New evaluation
# ----------------------------------------------------------------------------
with tab_eval:
    ui.section("1. Upload supplier proposals", ui.ACCENTS[0])
    uploaded = st.file_uploader(
        "Upload up to 6 supplier RFP response PDFs",
        type=["pdf"], accept_multiple_files=True,
    )
    if uploaded and len(uploaded) > 6:
        st.error("Please upload no more than 6 PDFs.")

    supplier_meta = {}
    if uploaded and len(uploaded) <= 6:
        ui.section("2. Confirm supplier details", ui.ACCENTS[1])
        for i, f in enumerate(uploaded):
            with st.expander(f"🏢 Supplier {i+1}: {f.name}", expanded=(i == 0)):
                cols = st.columns(3)
                name = cols[0].text_input(
                    "Supplier name",
                    value=os.path.splitext(f.name)[0].replace("_", " ").title(),
                    key=f"name_{i}",
                )
                submitted = cols[1].date_input("Submission date", value=date.today(), key=f"date_{i}")
                experience = cols[2].number_input(
                    "Historical experience rating (0–10)",
                    min_value=0.0, max_value=10.0, value=5.0, step=0.5, key=f"exp_{i}",
                )
                supplier_meta[f.name] = {
                    "supplier_name": name.strip(),
                    "submission_date": submitted.isoformat(),
                    "experience_rating": float(experience),
                    "bytes": f.getvalue(),
                }
        if supplier_meta:
            st.session_state["cached_proposals"] = supplier_meta

    ui.section("3. Run the evaluation", ui.ACCENTS[4])
    if not llm_config["ready"]:
        st.info("Configure a provider, model and API key in the sidebar under **LLM Settings** to enable evaluation.")
    else:
        st.caption(f"Using {llm_config['provider']} · {llm_config['model']}")

    run_disabled = not uploaded or len(uploaded) > 6 or not llm_config["ready"]
    if st.button("🚀 Evaluate suppliers", type="primary", disabled=run_disabled):
        if not criteria:
            st.error("No active criteria available.")
        elif abs(total_weight - 100) > 0.001:
            st.error("Active criterion weights must total 100%.")
        elif len({m["supplier_name"].casefold() for m in supplier_meta.values()}) != len(supplier_meta):
            st.error("Supplier names must be unique within a run.")
        elif any(not m["supplier_name"] for m in supplier_meta.values()):
            st.error("Every supplier needs a name.")
        else:
            all_results = []
            progress = st.progress(0)
            status = st.empty()
            for i, f in enumerate(uploaded):
                meta = supplier_meta[f.name]
                status.info(f"Processing {meta['supplier_name']} ({i+1}/{len(uploaded)})…")
                try:
                    pages = extract_pdf_pages(meta["bytes"])
                    full_text = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages)
                    if not full_text.strip():
                        raise ValueError(
                            "No text could be extracted. This may be a scanned PDF; "
                            "OCR is not included in this MVP."
                        )
                    ai_result = evaluate_supplier(
                        supplier_name=meta["supplier_name"],
                        document_text=full_text,
                        criteria=criteria,
                        llm_config=llm_config,
                    )
                    normalized, warnings = validate_and_normalize(ai_result, criteria)
                    risks = ai_result.get("risks", [])
                    all_results.append({
                        **meta,
                        "source_filename": f.name,
                        "criteria": normalized,
                        "risks": risks if isinstance(risks, list) else [],
                        "overall_summary": str(ai_result.get("overall_summary", "")),
                        "warnings": warnings,
                        "pages_count": len(pages),
                    })
                except Exception as e:
                    st.error(f"{meta['supplier_name']}: {e}")
                    all_results.append({
                        **meta,
                        "source_filename": f.name,
                        "criteria": validate_and_normalize({}, criteria)[0],
                        "risks": [f"Evaluation failed: {e}"],
                        "overall_summary": "Evaluation failed; scores are not reliable.",
                        "warnings": [str(e)],
                        "pages_count": 0,
                        "evaluation_failed": True,
                    })
                progress.progress((i + 1) / len(uploaded))

            if all_results:
                for result in all_results:
                    result.pop("bytes", None)
                results = rank_suppliers(calculate_supplier_metrics(all_results, criteria))
                run_id = save_run(results, criteria)
                status.success(f"Evaluation completed. Run ID: {run_id}")
                st.session_state["latest_run_id"] = run_id
                st.session_state["latest_results"] = results
                st.session_state["latest_provider"] = f"{llm_config['provider']} · {llm_config['model']}"
                st.rerun()


# ----------------------------------------------------------------------------
# Tab 3 — Results
# ----------------------------------------------------------------------------
with tab_results:
    results = st.session_state.get("latest_results")
    run_id = st.session_state.get("latest_run_id")

    if not results:
        st.info("Complete an evaluation to see the leaderboard, score charts and evidence here. "
                "Saved runs are stored in SQLite (`rfp_evaluation.db`).")
    else:
        winner = results[0]
        cards = st.columns(4)
        with cards[0]:
            ui.metric_card("Suppliers evaluated", len(results), run_id or "", ui.ACCENTS[0])
        with cards[1]:
            ui.metric_card("Leading supplier", winner["supplier_name"],
                           f"Rank 1 of {len(results)}", ui.ACCENTS[2])
        with cards[2]:
            ui.metric_card("Top absolute score", f"{winner['absolute_score']:.2f}/10",
                           "Weighted across all criteria", ui.ACCENTS[4])
        with cards[3]:
            ui.metric_card("Top PPI", f"{winner['ppi']:.1f}%",
                           "Peer performance index", ui.ACCENTS[3])

        if st.session_state.get("latest_provider"):
            st.caption(f"Scored with {st.session_state['latest_provider']}")

        ui.section("Leaderboard", ui.ACCENTS[0])
        leaderboard = pd.DataFrame([{
            "Rank": r["final_rank"],
            "Supplier": r["supplier_name"],
            "Absolute score": round(r["absolute_score"], 2),
            "PPI": round(r["ppi"], 2),
            "Submission date": r["submission_date"],
            "Experience": r["experience_rating"],
        } for r in results])
        st.dataframe(
            leaderboard, hide_index=True, width="stretch",
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Absolute score": st.column_config.ProgressColumn(
                    "Absolute score", min_value=0, max_value=10, format="%.2f",
                    color=ui.ACCENTS[0],
                ),
                "PPI": st.column_config.ProgressColumn(
                    "PPI (%)", min_value=0, max_value=100, format="%.1f%%",
                    color=ui.ACCENTS[4],
                ),
            },
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            ui.section("Score comparison by criterion", ui.ACCENTS[1])
            compare_df = pd.DataFrame([{
                "Criterion": c["name"], "Supplier": r["supplier_name"], "Score": c["score"],
            } for r in results for c in r["criteria"]])
            st.altair_chart(
                ui.grouped_bar_chart(compare_df, "Criterion", "Score", "Supplier",
                                     "Raw criterion scores per supplier"),
                width="stretch",
            )
        with chart_right:
            ui.section(f"Where {winner['supplier_name']} earns its score", ui.ACCENTS[3])
            contribution_df = pd.DataFrame([{
                "Criterion": c["name"],
                "Weighted points": round(
                    (c["score"] / c["max_score"] if c["max_score"] else 0) * c["weight"] / 10.0, 3
                ),
            } for c in winner["criteria"]])
            st.altair_chart(
                ui.donut_chart(contribution_df, "Criterion", "Weighted points",
                               "Weighted points earned by the leading supplier",
                               "Points", ".2f"),
                width="stretch",
            )

        ui.section("Evidence and justification", ui.ACCENTS[5])
        for r in results:
            header = (f"#{r['final_rank']} — {r['supplier_name']} | "
                      f"Score {r['absolute_score']:.2f}/10 | PPI {r['ppi']:.2f}%")
            with st.expander(header, expanded=(r["final_rank"] == 1)):
                if r.get("overall_summary"):
                    st.write(r["overall_summary"])
                if r.get("risks"):
                    st.markdown("**Risks**")
                    for risk in r["risks"]:
                        st.markdown(f"- {risk}")
                detail = pd.DataFrame([{
                    "Criterion": c["name"], "Score": c["score"], "Max": c["max_score"],
                    "Weight (%)": c["weight"], "Benchmark": c.get("benchmark"),
                    "Gap": c.get("gap"), "Relative (%)": c.get("relative_performance_pct"),
                    "Status": c["status"], "Evidence": c.get("evidence", ""),
                    "Justification": c.get("justification", ""),
                } for c in r["criteria"]])
                st.dataframe(
                    detail, hide_index=True, width="stretch",
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score", min_value=0,
                            max_value=float(detail["Max"].max()) if not detail.empty else 10,
                            format="%.1f", color=ui.ACCENTS[1],
                        ),
                        "Relative (%)": st.column_config.ProgressColumn(
                            "Relative (%)", min_value=0, max_value=100, format="%.1f%%",
                            color=ui.ACCENTS[2],
                        ),
                    },
                )
                if r.get("warnings"):
                    st.warning("Warnings: " + "; ".join(r["warnings"]))
                st.caption("Tie-break order: PPI descending → submission date ascending → "
                           "experience descending → supplier name ascending.")

        export_obj = {"rfp_run_id": run_id, "results": results}
        st.download_button(
            "⬇️ Download complete result as JSON",
            json.dumps(export_obj, indent=2, ensure_ascii=False),
            file_name=f"{run_id}.json",
            mime="application/json",
        )


# ----------------------------------------------------------------------------
# Tab 4 — Document Search (Semantic search with and without LLM)
# ----------------------------------------------------------------------------
with tab_search:
    ui.section("Semantic document search", ui.ACCENTS[1])
    st.caption(
        "Search across uploaded supplier proposals using fast non-LLM statistical "
        "semantic retrieval (BM25) or deep LLM semantic analysis. View ranked documents "
        "and line-by-line chunk evidence with exact page and line numbers."
    )

    proposals = st.session_state.get("cached_proposals", {})

    # Document management section
    if not proposals:
        st.info("No supplier proposals loaded yet. You can upload PDFs below or load the built-in sample proposals.")
        doc_col1, doc_col2 = st.columns([2, 1])
        with doc_col1:
            uploaded_search_files = st.file_uploader(
                "Upload supplier RFP PDF documents to search",
                type=["pdf"],
                accept_multiple_files=True,
                key="search_tab_uploader",
            )
            if uploaded_search_files:
                new_meta = {}
                for f in uploaded_search_files:
                    new_meta[f.name] = {
                        "supplier_name": os.path.splitext(f.name)[0].replace("_", " ").title(),
                        "submission_date": date.today().isoformat(),
                        "experience_rating": 5.0,
                        "bytes": f.getvalue(),
                    }
                st.session_state["cached_proposals"] = new_meta
                st.rerun()
        with doc_col2:
            st.write("")
            st.write("")
            if st.button("📁 Load 4 sample proposals", help="Load sample PDFs (Apex, BrightPath, NexaWorks, Orbit)"):
                sample_meta = {}
                for p in sorted(glob.glob("sample_pdfs/*.pdf")):
                    fname = os.path.basename(p)
                    sname = os.path.splitext(fname)[0].replace("_", " ").title()
                    with open(p, "rb") as sf:
                        sample_meta[fname] = {
                            "supplier_name": sname,
                            "submission_date": date.today().isoformat(),
                            "experience_rating": 5.0,
                            "bytes": sf.read(),
                        }
                st.session_state["cached_proposals"] = sample_meta
                st.rerun()
    else:
        doc_expander_col, clear_btn_col = st.columns([4, 1])
        with doc_expander_col:
            st.write(f"📁 **Active documents indexed ({len(proposals)}):** " +
                     ", ".join(f"`{m['supplier_name']}`" for m in proposals.values()))
        with clear_btn_col:
            if st.button("🔄 Change documents", key="clear_proposals"):
                st.session_state.pop("cached_proposals", None)
                st.session_state.pop("search_results", None)
                st.rerun()

    if proposals:
        # Index chunks
        all_chunks = []
        for fname, meta in proposals.items():
            chunks = extract_document_chunks(
                meta["bytes"],
                filename=fname,
                supplier_name=meta.get("supplier_name"),
            )
            all_chunks.extend(chunks)

        st.divider()

        # Search Controls
        search_col1, search_col2 = st.columns([3, 2])
        with search_col1:
            query = st.text_input(
                "Search query",
                value=st.session_state.get("active_search_query", ""),
                placeholder="e.g. disaster recovery, fixed pricing, annual support cost, ISO 27001, implementation timeline...",
                key="doc_search_query_input",
            )
        with search_col2:
            search_mode = st.radio(
                "Search mode",
                ["⚡ Without LLM (BM25 Semantic)", "🤖 With LLM (Deep Semantic)"],
                horizontal=True,
                help="Without LLM: instant statistical BM25 retrieval. With LLM: uses the configured LLM for deep semantic reasoning and re-ranking.",
            )

        # Quick query buttons
        st.write("**Quick sample searches:**")
        q_cols = st.columns(4)
        sample_queries = [
            ("💰 Pricing & Support", "fixed implementation fee and annual support cost"),
            ("🛡️ ISO 27001 Security", "ISO 27001 compliance, encryption, and audit logs"),
            ("⚙️ Cloud Architecture", "cloud-native architecture, REST integrations, and disaster recovery"),
            ("⏱️ Timeline & Rollback", "implementation timeline, milestones, and rollback plan"),
        ]
        for col, (label, sample_q) in zip(q_cols, sample_queries):
            if col.button(label, key=f"quick_{label}"):
                st.session_state["active_search_query"] = sample_q
                query = sample_q
                st.rerun()

        filter_col, k_col = st.columns([3, 1])
        all_doc_names = sorted(list({c["document_name"] for c in all_chunks}))
        with filter_col:
            selected_docs = st.multiselect(
                "Filter documents",
                options=all_doc_names,
                default=all_doc_names,
                help="Limit search to specific supplier proposals",
            )
        with k_col:
            top_k = st.slider("Max chunks to show", min_value=3, max_value=15, value=6)

        filtered_chunks = [c for c in all_chunks if c["document_name"] in selected_docs]

        effective_query = query.strip() or st.session_state.get("active_search_query", "").strip()

        if st.button("🔍 Run Semantic Search", type="primary", disabled=not effective_query or not filtered_chunks):
            with st.spinner("Searching document chunks across proposals..."):
                if "With LLM" in search_mode:
                    if not llm_config.get("ready"):
                        st.warning("⚠️ LLM is not configured. Falling back to Without LLM mode. (Configure an API key in the sidebar for LLM mode).")
                        results = search_without_llm(effective_query, filtered_chunks, top_k=top_k)
                    else:
                        results = search_with_llm(effective_query, filtered_chunks, llm_config, top_k=top_k)
                else:
                    results = search_without_llm(effective_query, filtered_chunks, top_k=top_k)

            st.session_state["search_results"] = results

        # Display Search Results if available
        search_results = st.session_state.get("search_results")
        if search_results and search_results.get("results"):
            st.divider()
            ui.section("Search results & document matching", ui.ACCENTS[4])

            # Top matching document highlight card
            top_doc = search_results.get("top_document")
            top_score = search_results.get("top_document_score", 0.0)

            if top_doc:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #EEF2FF 0%, #E0F2FE 100%);
                                border: 2px solid #818CF8; border-radius: 14px; padding: 18px 22px;
                                margin-bottom: 20px; box-shadow: 0 4px 16px rgba(79,70,229,.12);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <div style="font-size:0.85rem; font-weight:700; color:#4338CA; text-transform:uppercase; letter-spacing:0.05em;">
                                🏆 Most Relevant Document Overall
                            </div>
                            <span style="background:#4F46E5; color:#FFFFFF; padding:4px 14px; border-radius:999px; font-weight:800; font-size:0.9rem;">
                                Match Score: {top_score:.1f}%
                            </span>
                        </div>
                        <div style="font-size:1.65rem; font-weight:800; color:#1E1B4B; margin-bottom:6px;">
                            🏢 {top_doc}
                        </div>
                        <div style="font-size:0.96rem; color:#374151; line-height:1.5;">
                            {search_results.get('overall_answer') or search_results.get('top_document_summary', 'Highest relevance across matched clauses and criteria.')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Document Match Score Comparison Table
            rankings = search_results.get("document_rankings", [])
            if len(rankings) > 1:
                ui.section("Document match comparison ranking", ui.ACCENTS[0])
                rank_df = pd.DataFrame([
                    {
                        "Document": r["document_name"],
                        "Overall Match (%)": r["score"],
                        "Peak Chunk Match (%)": r["max_score"],
                        "Matching Chunks": r["match_count"],
                    }
                    for r in rankings
                ])
                st.dataframe(
                    rank_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Overall Match (%)": st.column_config.ProgressColumn(
                            "Overall Match", min_value=0, max_value=100, format="%.1f%%",
                            color=ui.ACCENTS[0],
                        ),
                        "Peak Chunk Match (%)": st.column_config.ProgressColumn(
                            "Peak Chunk Match", min_value=0, max_value=100, format="%.1f%%",
                            color=ui.ACCENTS[1],
                        ),
                    },
                )

            # Matching Chunks with Page & Line Numbers
            ui.section("Matching chunk evidence (with page & line numbers)", ui.ACCENTS[2])
            query_terms = tokenize(search_results.get("query", ""))
            for chunk in search_results["results"]:
                ui.chunk_card(
                    chunk,
                    query_terms=query_terms,
                    reasoning=chunk.get("reasoning"),
                    matched_quote=chunk.get("matched_quote"),
                )
        elif search_results and not search_results.get("results"):
            st.info("No matching chunks found for your search query. Try broader keywords.")
