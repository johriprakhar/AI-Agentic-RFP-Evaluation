"""
Streamlit front end.

This module is presentation only. It gathers inputs, calls `orchestrator`, and
renders what comes back. It contains no scoring arithmetic, no ranking logic and
no SQL beyond calling the functions in `database`, which is what keeps the
evaluation pipeline testable without a browser.
"""

import glob
import io
import json
import os
from datetime import date

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import database
import orchestrator
import ui
from database import (
    get_active_criteria,
    get_all_criteria,
    get_run,
    init_db,
    list_runs,
    replace_criteria,
    reset_criteria_to_defaults,
)
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
from scoring import TIE_BREAK_ORDER

try:
    from search_engine import extract_document_chunks, search_with_llm, search_without_llm, tokenize
    SEARCH_AVAILABLE = True
    SEARCH_IMPORT_ERROR = ""
except Exception as error:  # pragma: no cover - import guard
    SEARCH_AVAILABLE = False
    SEARCH_IMPORT_ERROR = str(error)

load_dotenv()
init_db()

st.set_page_config(
    page_title="Agentic RFP Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_theme()

MAX_SUPPLIERS = 6


# ----------------------------------------------------------------------------
# Sidebar: LLM settings
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
        base_url_input = st.text_input(
            "Base URL",
            key=f"base_url::{provider}",
            placeholder="https://your-endpoint/v1",
            help="Any endpoint that implements POST /chat/completions.",
        )

    key_state = f"api_key::{provider}"
    if key_state not in st.session_state:
        st.session_state[key_state] = env_api_key(provider)
    api_key = st.text_input(
        f"{provider} API key",
        type="password",
        key=key_state,
        placeholder=config.get("key_prefix_hint", ""),
        help="Held in this browser session only; never written to disk.",
    ).strip()
    if config.get("keys_url"):
        st.caption(f"Get a key: {config['keys_url']}")

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
        temperature = st.slider(
            "Temperature", 0.0, 1.0, 0.0, 0.1,
            help="Keep at 0 for repeatable scoring.",
        )
        max_chars = st.number_input(
            "Max characters per document", min_value=5000, max_value=400000,
            value=int(os.getenv("MAX_DOCUMENT_CHARS", "70000")), step=5000,
            help="Longer proposals are condensed by criterion relevance, not cut off.",
        )
        max_attempts = st.number_input(
            "Schema repair attempts", min_value=1, max_value=5, value=3, step=1,
            help="How many times to send validation errors back to the model before "
                 "falling back to safe defaults.",
        )
        workers = st.number_input(
            "Parallel suppliers", min_value=1, max_value=6, value=orchestrator.MAX_WORKERS, step=1,
            help="How many proposals to evaluate concurrently.",
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
        "temperature": float(temperature),
        "max_document_chars": int(max_chars),
        "max_attempts": int(max_attempts),
        "max_workers": int(workers),
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
weights_valid = bool(criteria) and abs(total_weight - 100.0) <= 1e-3

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

tab_criteria, tab_eval, tab_results, tab_history, tab_search = st.tabs(
    ["📊 Criteria", "📤 New evaluation", "🏆 Results", "🗂️ Run history", "🔍 Document search"]
)


def weight_band(weight):
    if weight >= 25:
        return "High influence (≥25%)"
    if weight >= 15:
        return "Medium influence (15–25%)"
    return "Supporting (<15%)"


# ----------------------------------------------------------------------------
# Tab 1 — Criteria (view + editor)
# ----------------------------------------------------------------------------
with tab_criteria:
    view_tab, edit_tab = st.tabs(["Overview", "✏️ Edit criteria"])

    with view_tab:
        if not criteria:
            st.warning("No active criteria. Add some under **Edit criteria**, or run "
                       "`python database.py` to seed the defaults.")
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
                    "Balanced" if weights_valid else "Must total 100%",
                    ui.ACCENTS[4] if weights_valid else "#EF4444",
                )
            with cards[2]:
                ui.metric_card("Score scale", f"0 – {criteria_df['Max score'].max():.0f}",
                               "Per-criterion raw score", ui.ACCENTS[1])
            with cards[3]:
                ui.metric_card("Heaviest criterion", heaviest["Criterion"],
                               f"{heaviest['Weight']:.1f}% of the decision", ui.ACCENTS[3])

            if not weights_valid:
                st.error(f"Active weights total {total_weight:.1f}%, not 100%. Evaluation is blocked "
                         "until this is corrected under **Edit criteria**.")

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
                hide_index=True, width="stretch",
                column_config={
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "Weight": st.column_config.ProgressColumn(
                        "Weight (%)", min_value=0, max_value=100, format="%.1f%%", color=ui.ACCENTS[0],
                    ),
                    "Max score": st.column_config.NumberColumn("Max score", format="%.0f"),
                },
            )

    with edit_tab:
        ui.section("Edit evaluation criteria", ui.ACCENTS[3])
        st.caption(
            "Criteria are stored in SQLite and injected into the scoring prompt, so changes here "
            "take effect on the next run with no code change. Historical runs are unaffected: each "
            "run snapshots the criteria it used."
        )

        editor_df = pd.DataFrame([{
            "ID": int(c["criterion_id"]),
            "Criterion": c["name"],
            "Description": c["description"],
            "Weight": float(c["weight"]),
            "Max score": float(c["max_score"]),
            "Active": bool(c["is_active"]),
        } for c in get_all_criteria()])
        if editor_df.empty:
            editor_df = pd.DataFrame(
                columns=["ID", "Criterion", "Description", "Weight", "Max score", "Active"]
            )

        edited = st.data_editor(
            editor_df, hide_index=True, width="stretch", num_rows="dynamic",
            key="criteria_editor",
            column_config={
                "ID": st.column_config.NumberColumn(
                    "ID", help="Leave blank on a new row and an id is assigned.", width="small",
                ),
                "Criterion": st.column_config.TextColumn("Criterion", required=True),
                "Description": st.column_config.TextColumn(
                    "Description", help="Also used as the retrieval query when a long proposal "
                                        "must be condensed.", width="large",
                ),
                "Weight": st.column_config.NumberColumn(
                    "Weight (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f",
                ),
                "Max score": st.column_config.NumberColumn(
                    "Max score", min_value=1.0, max_value=100.0, step=1.0, format="%.0f",
                ),
                "Active": st.column_config.CheckboxColumn("Active"),
            },
        )

        pending_active = edited[edited["Active"].fillna(False)] if not edited.empty else edited
        pending_total = float(pending_active["Weight"].fillna(0).sum()) if not pending_active.empty else 0.0
        if abs(pending_total - 100.0) <= 1e-3:
            ui.pill(f"● Active weights total {pending_total:.1f}%", "ok")
        else:
            ui.pill(f"● Active weights total {pending_total:.1f}% — must be 100% to evaluate", "warn")

        save_col, reset_col = st.columns([1, 1])
        if save_col.button("💾 Save criteria", type="primary"):
            rows, problems, next_id = [], [], 0
            existing_ids = [int(v) for v in editor_df["ID"].dropna().tolist()] if not editor_df.empty else []
            next_id = max(existing_ids) if existing_ids else 0

            for position, row in edited.iterrows():
                name = str(row.get("Criterion") or "").strip()
                if not name:
                    problems.append(f"Row {position + 1}: a criterion name is required.")
                    continue
                try:
                    weight = float(row.get("Weight") or 0.0)
                    max_score = float(row.get("Max score") or 10.0)
                except (TypeError, ValueError):
                    problems.append(f"Row {position + 1}: weight and max score must be numbers.")
                    continue
                if max_score <= 0:
                    problems.append(f"Row {position + 1}: max score must be greater than zero.")
                    continue

                raw_id = row.get("ID")
                if pd.isna(raw_id):
                    next_id += 1
                    criterion_id = next_id
                else:
                    criterion_id = int(raw_id)
                    next_id = max(next_id, criterion_id)

                rows.append({
                    "criterion_id": criterion_id,
                    "name": name,
                    "description": str(row.get("Description") or ""),
                    "weight": weight,
                    "max_score": max_score,
                    "is_active": bool(row.get("Active", True)),
                })

            ids = [r["criterion_id"] for r in rows]
            if len(set(ids)) != len(ids):
                problems.append("Criterion ids must be unique.")

            if problems:
                for problem in problems:
                    st.error(problem)
            else:
                replace_criteria(rows)
                active_total = sum(r["weight"] for r in rows if r["is_active"])
                st.success(f"Saved {len(rows)} criteria. Active weights total {active_total:.1f}%.")
                st.rerun()

        if reset_col.button("↩️ Restore defaults"):
            count = reset_criteria_to_defaults()
            st.success(f"Restored the {count} default criteria.")
            st.rerun()


# ----------------------------------------------------------------------------
# Tab 2 — New evaluation
# ----------------------------------------------------------------------------
with tab_eval:
    ui.section("1. Upload supplier proposals", ui.ACCENTS[0])
    uploaded = st.file_uploader(
        f"Upload up to {MAX_SUPPLIERS} supplier RFP response PDFs",
        type=["pdf"], accept_multiple_files=True,
    )
    if uploaded and len(uploaded) > MAX_SUPPLIERS:
        st.error(f"Please upload no more than {MAX_SUPPLIERS} PDFs.")

    proposals = []
    if uploaded and len(uploaded) <= MAX_SUPPLIERS:
        ui.section("2. Confirm supplier details", ui.ACCENTS[1])
        cached = {}
        for index, file in enumerate(uploaded):
            with st.expander(f"🏢 Supplier {index + 1}: {file.name}", expanded=(index == 0)):
                cols = st.columns(3)
                name = cols[0].text_input(
                    "Supplier name",
                    value=os.path.splitext(file.name)[0].replace("_", " ").title(),
                    key=f"name_{index}",
                )
                submitted = cols[1].date_input(
                    "Submission date", value=date.today(), key=f"date_{index}",
                    help="Used as the first tie-break: an earlier submission wins a tied PPI.",
                )
                experience = cols[2].number_input(
                    "Historical experience rating (0–10)", min_value=0.0, max_value=10.0,
                    value=5.0, step=0.5, key=f"exp_{index}",
                    help="Buyer-supplied prior rating. Used as the second tie-break.",
                )
            payload = {
                "supplier_name": name.strip(),
                "submission_date": submitted.isoformat(),
                "experience_rating": float(experience),
                "source_filename": file.name,
                "bytes": file.getvalue(),
            }
            proposals.append(payload)
            cached[file.name] = payload
        st.session_state["cached_proposals"] = cached

    ui.section("3. Run the evaluation", ui.ACCENTS[4])
    problems = orchestrator.validate_run_inputs(proposals, criteria, MAX_SUPPLIERS) if proposals else []
    if not llm_config["ready"]:
        st.info("Configure a provider, model and API key in the sidebar under **LLM Settings** "
                "to enable evaluation.")
    else:
        st.caption(f"Using {llm_config['provider']} · {llm_config['model']} · "
                   f"temperature {llm_config['temperature']:.1f}")
    for problem in problems:
        st.warning(problem)

    run_disabled = bool(problems) or not proposals or not llm_config["ready"]
    if st.button("🚀 Evaluate suppliers", type="primary", disabled=run_disabled):
        progress_bar = st.progress(0.0)
        status = st.empty()

        def on_progress(completed, total, label):
            progress_bar.progress(completed / total)
            status.info(f"Completed {completed}/{total} — {label}")

        try:
            with st.spinner("Evaluating proposals…"):
                outcome = orchestrator.run_evaluation(
                    proposals, criteria, llm_config,
                    progress=on_progress,
                    max_workers=llm_config.get("max_workers", orchestrator.MAX_WORKERS),
                    max_attempts=llm_config.get("max_attempts", 3),
                )
        except Exception as error:
            progress_bar.empty()
            status.empty()
            st.error(f"The run could not be completed: {error}")
        else:
            progress_bar.progress(1.0)
            st.session_state["active_run"] = outcome
            status.success(
                f"Run {outcome['run_id']} completed in {outcome['meta']['duration_seconds']:.1f}s. "
                "Open the **🏆 Results** tab."
            )


# ----------------------------------------------------------------------------
# Results rendering (shared by the Results and Run history tabs)
# ----------------------------------------------------------------------------
def render_results(run):
    results = run.get("results") or []
    run_id = run.get("run_id")
    meta = run.get("meta") or {}
    run_criteria = run.get("criteria") or []

    if not results:
        st.info("This run holds no supplier results.")
        return

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
        grounding = (meta.get("evidence_grounding") or {}).get("grounding_rate")
        ui.metric_card(
            "Evidence grounded", f"{grounding:.0%}" if grounding is not None else "n/a",
            "Quotes found in the source PDFs", ui.ACCENTS[3],
        )

    details = []
    if meta.get("provider"):
        details.append(f"Scored with {meta['provider']} · {meta.get('model', '')}")
    if meta.get("completed_at"):
        details.append(f"completed {meta['completed_at'][:19].replace('T', ' ')} UTC")
    if details:
        st.caption(" · ".join(details))

    for warning in meta.get("warnings", []) or run.get("warnings", []) or []:
        st.warning(warning)

    # --- Leaderboard --------------------------------------------------------
    ui.section("Leaderboard", ui.ACCENTS[0])
    leaderboard = pd.DataFrame([{
        "Rank": r["final_rank"],
        "Supplier": r["supplier_name"],
        "Absolute score": round(float(r["absolute_score"]), 2),
        "PPI": round(float(r["ppi"]), 2),
        "Submission date": r["submission_date"],
        "Experience": r["experience_rating"],
        "Evidence": _grounding_text(r),
        "Why this rank": r.get("rank_explanation", ""),
    } for r in results])
    st.dataframe(
        leaderboard, hide_index=True, width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Absolute score": st.column_config.ProgressColumn(
                "Absolute score", min_value=0, max_value=10, format="%.2f", color=ui.ACCENTS[0],
            ),
            "PPI": st.column_config.ProgressColumn(
                "PPI (%)", min_value=0, max_value=100, format="%.1f%%", color=ui.ACCENTS[4],
            ),
            "Why this rank": st.column_config.TextColumn("Why this rank", width="large"),
        },
    )
    st.caption(
        "Ranking order: " + " → ".join(rule["label"] for rule in TIE_BREAK_ORDER)
        + ". Suppliers are ranked on PPI (performance against the best peer per criterion); "
        "the absolute score is shown for context and is peer-independent."
    )

    # --- Charts -------------------------------------------------------------
    chart_left, chart_right = st.columns(2)
    with chart_left:
        ui.section("Score comparison by criterion", ui.ACCENTS[1])
        compare_df = pd.DataFrame([{
            "Criterion": row["name"], "Supplier": r["supplier_name"], "Score": row["score"],
        } for r in results for row in r["criteria"]])
        st.altair_chart(
            ui.grouped_bar_chart(compare_df, "Criterion", "Score", "Supplier",
                                 "Raw criterion scores per supplier"),
            width="stretch",
        )
    with chart_right:
        ui.section(f"Where {winner['supplier_name']} earns its score", ui.ACCENTS[3])
        contribution_df = pd.DataFrame([{
            "Criterion": row["name"],
            "Weighted points": row.get(
                "weighted_points",
                round((row["score"] / row["max_score"] if row["max_score"] else 0) * row["weight"] / 10.0, 3),
            ),
        } for row in winner["criteria"]])
        st.altair_chart(
            ui.donut_chart(contribution_df, "Criterion", "Weighted points",
                           "Weighted points earned by the leading supplier", "Points", ".2f"),
            width="stretch",
        )

    # --- Drill-down ---------------------------------------------------------
    ui.section("Evidence and justification", ui.ACCENTS[5])
    for result in results:
        flag = " ⚠️" if result.get("evaluation_failed") else ""
        header = (f"#{result['final_rank']} — {result['supplier_name']}{flag} | "
                  f"Score {float(result['absolute_score']):.2f}/10 | "
                  f"PPI {float(result['ppi']):.2f}%")
        with st.expander(header, expanded=(result["final_rank"] == 1)):
            if result.get("rank_explanation"):
                st.markdown(f"**Placement.** {result['rank_explanation']}")
            if result.get("overall_summary"):
                st.write(result["overall_summary"])
            if result.get("risks"):
                st.markdown("**Risks**")
                for risk in result["risks"]:
                    st.markdown(f"- {risk}")

            detail = pd.DataFrame([{
                "Criterion": row["name"],
                "Score": row["score"],
                "Max": row["max_score"],
                "Weight (%)": row["weight"],
                "Points": row.get("weighted_points"),
                "Benchmark": row.get("benchmark"),
                "Gap": row.get("gap"),
                "Relative (%)": row.get("relative_performance_pct"),
                "Status": row["status"],
                "Evidence check": ui.evidence_label(row.get("evidence_status")),
                "Evidence": row.get("evidence", ""),
                "Justification": row.get("justification", ""),
            } for row in result["criteria"]])
            st.dataframe(
                detail, hide_index=True, width="stretch",
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0,
                        max_value=float(detail["Max"].max()) if not detail.empty else 10,
                        format="%.1f", color=ui.ACCENTS[1],
                    ),
                    "Relative (%)": st.column_config.NumberColumn("Relative (%)", format="%.1f"),
                    "Evidence": st.column_config.TextColumn("Evidence", width="large"),
                    "Justification": st.column_config.TextColumn("Justification", width="large"),
                },
            )

            notes = [
                f"**{row['name']}** — {row['evidence_check']['note']}"
                for row in result["criteria"]
                if (row.get("evidence_check") or {}).get("note")
                and row.get("evidence_status") in {"partial", "unverified", "empty"}
            ]
            if notes:
                with st.container():
                    st.markdown("**Evidence verification notes**")
                    for note in notes:
                        st.markdown(f"- {note}")

            preparation = result.get("preparation") or {}
            stats = result.get("document_stats") or {}
            info_bits = []
            if stats.get("page_count"):
                info_bits.append(f"{stats['page_count']} page(s), {stats.get('char_count', 0):,} characters")
            if preparation.get("strategy"):
                info_bits.append(
                    f"input strategy: {preparation['strategy']} "
                    f"({preparation.get('coverage', 1):.0%} of the document sent)"
                )
            if info_bits:
                st.caption(" · ".join(info_bits))

            if result.get("warnings"):
                st.warning("Warnings: " + "; ".join(str(w) for w in result["warnings"]))

            trace = result.get("tool_trace") or {}
            if trace.get("invocations"):
                with st.expander("🔧 Tool execution trace", expanded=False):
                    st.caption(
                        f"{trace.get('deterministic_calls', 0)} deterministic step(s) and "
                        f"{trace.get('model_calls', 0)} model call(s) in "
                        f"{trace.get('total_ms', 0):.0f} ms. Every number in the table above comes "
                        "from a deterministic step."
                    )
                    st.dataframe(
                        pd.DataFrame([{
                            "Step": item["tool"],
                            "Kind": item["kind"],
                            "OK": item["ok"],
                            "ms": round(item["duration_ms"], 1),
                            "Detail": item.get("summary") or item.get("error", ""),
                        } for item in trace["invocations"]]),
                        hide_index=True, width="stretch",
                    )

    # --- Run provenance -----------------------------------------------------
    with st.expander("🧾 Run provenance and tool manifest", expanded=False):
        provenance = {
            "Run id": run_id,
            "Provider": meta.get("provider"),
            "Model": meta.get("model"),
            "Temperature": meta.get("temperature"),
            "Started": meta.get("created_at"),
            "Completed": meta.get("completed_at"),
            "Duration (s)": meta.get("duration_seconds"),
            "Criteria": meta.get("criteria_count", len(run_criteria)),
            "Weight total": meta.get("criteria_weight_total"),
            "App version": meta.get("app_version"),
        }
        st.dataframe(
            pd.DataFrame([{"Field": k, "Value": "" if v is None else str(v)}
                          for k, v in provenance.items()]),
            hide_index=True, width="stretch",
        )
        if run_criteria:
            st.markdown("**Criteria snapshot used by this run**")
            st.dataframe(
                pd.DataFrame([{
                    "ID": c["criterion_id"], "Criterion": c["name"],
                    "Weight (%)": c["weight"], "Max score": c["max_score"],
                } for c in run_criteria]),
                hide_index=True, width="stretch",
            )
        manifest = meta.get("tool_manifest") or []
        if manifest:
            st.markdown("**Tool manifest**")
            st.dataframe(
                pd.DataFrame([{
                    "Tool": t["tool"], "Kind": t["kind"], "Responsibility": t["description"],
                } for t in manifest]),
                hide_index=True, width="stretch",
                column_config={"Responsibility": st.column_config.TextColumn(width="large")},
            )

    # --- Exports ------------------------------------------------------------
    ui.section("Export", ui.ACCENTS[2])
    export = orchestrator.export_payload(run_id, results, run_criteria, meta)
    json_col, csv_col = st.columns(2)
    json_col.download_button(
        "⬇️ Complete result (JSON)",
        json.dumps(export, indent=2, ensure_ascii=False, default=str),
        file_name=f"{run_id or 'rfp-run'}.json",
        mime="application/json",
        width="stretch",
        help="Self-contained audit record: scores, evidence, criteria snapshot and run metadata.",
    )
    csv_col.download_button(
        "⬇️ Criterion scores (CSV)",
        _criteria_csv(run_id, results),
        file_name=f"{run_id or 'rfp-run'}-criteria.csv",
        mime="text/csv",
        width="stretch",
        help="One row per supplier and criterion, for spreadsheet review.",
    )


def _grounding_text(result):
    summary = result.get("evidence_summary") or {}
    checked = summary.get("checked") or 0
    if not checked:
        return "n/a"
    return f"{summary.get('verified', 0)}/{checked} verified"


def _criteria_csv(run_id, results):
    rows = []
    for result in results:
        for row in result["criteria"]:
            rows.append({
                "rfp_run_id": run_id,
                "final_rank": result["final_rank"],
                "supplier_name": result["supplier_name"],
                "absolute_score": result["absolute_score"],
                "ppi": result["ppi"],
                "criterion_id": row["criterion_id"],
                "criterion": row["name"],
                "weight_pct": row["weight"],
                "score": row["score"],
                "max_score": row["max_score"],
                "weighted_points": row.get("weighted_points"),
                "benchmark": row.get("benchmark"),
                "gap": row.get("gap"),
                "relative_performance_pct": row.get("relative_performance_pct"),
                "status": row["status"],
                "evidence_status": row.get("evidence_status"),
                "evidence": row.get("evidence", ""),
                "justification": row.get("justification", ""),
            })
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue()


# ----------------------------------------------------------------------------
# Tab 3 — Results
# ----------------------------------------------------------------------------
with tab_results:
    active_run = st.session_state.get("active_run")
    if not active_run:
        st.info("Run an evaluation, or open a stored run from the **🗂️ Run history** tab. "
                "Every run is persisted to SQLite (`rfp_evaluation.db`).")
    else:
        if active_run.get("loaded_from_history"):
            st.caption(f"Viewing stored run {active_run.get('run_id')} loaded from the database.")
        render_results(active_run)


# ----------------------------------------------------------------------------
# Tab 4 — Run history
# ----------------------------------------------------------------------------
with tab_history:
    ui.section("Stored evaluation runs", ui.ACCENTS[0])
    st.caption(
        "Every run is reloadable from SQLite together with the criteria snapshot it used, so a "
        "historical ranking stays explainable even after the active criteria change."
    )

    runs = list_runs(limit=100)
    if not runs:
        st.info("No runs stored yet. Complete an evaluation to populate the history.")
    else:
        history_df = pd.DataFrame([{
            "Run id": r["rfp_run_id"],
            "Created (UTC)": str(r["created_at"])[:19].replace("T", " "),
            "Status": r["status"],
            "Winner": r["winner"] or "—",
            "Suppliers": r["supplier_count"],
            "Criteria": r["criteria_count"],
            "Weight total": r["criteria_weight_total"],
            "Provider": r["provider"] or "—",
            "Model": r["model"] or "—",
        } for r in runs])
        st.dataframe(history_df, hide_index=True, width="stretch")

        pick_col, load_col, delete_col = st.columns([3, 1, 1])
        selected = pick_col.selectbox(
            "Select a run",
            [r["rfp_run_id"] for r in runs],
            format_func=lambda rid: next(
                (f"{rid} — {r['winner'] or 'no winner'} ({str(r['created_at'])[:10]})"
                 for r in runs if r["rfp_run_id"] == rid), rid,
            ),
            key="history_pick",
        )
        load_col.write("")
        delete_col.write("")
        if load_col.button("📂 Open run", width="stretch"):
            stored = get_run(selected)
            if not stored:
                st.error("That run could not be found.")
            else:
                st.session_state["active_run"] = {
                    "run_id": stored["run_id"],
                    "results": stored["results"],
                    "criteria": stored["criteria"],
                    "meta": stored["run"].get("meta", {}),
                    "warnings": stored["run"].get("meta", {}).get("warnings", []),
                    "loaded_from_history": True,
                }
                st.success(f"Loaded {selected}. Open the **🏆 Results** tab.")

        if delete_col.button("🗑️ Delete", width="stretch"):
            database.delete_run(selected)
            if (st.session_state.get("active_run") or {}).get("run_id") == selected:
                st.session_state.pop("active_run", None)
            st.success(f"Deleted {selected}.")
            st.rerun()

        with st.expander("Compare stored runs", expanded=False):
            comparable = [r["rfp_run_id"] for r in runs]
            chosen = st.multiselect("Runs to compare", comparable, default=comparable[:3])
            rows = []
            for run_id in chosen:
                for result in get_run(run_id)["results"]:
                    rows.append({
                        "Run": run_id,
                        "Supplier": result["supplier_name"],
                        "Rank": result["final_rank"],
                        "Absolute score": round(float(result["absolute_score"]), 2),
                        "PPI": round(float(result["ppi"]), 2),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------
# Tab 5 — Document search
# ----------------------------------------------------------------------------
with tab_search:
    ui.section("Semantic document search", ui.ACCENTS[1])

    if not SEARCH_AVAILABLE:
        st.error(
            "The search engine could not be loaded, so this tab is unavailable. "
            "The rest of the application is unaffected. "
            f"Import error: {SEARCH_IMPORT_ERROR}"
        )
    else:
        st.caption(
            "Search across supplier proposals using offline BM25 retrieval or deep LLM semantic "
            "analysis. Results cite the exact document, page and line numbers."
        )

        search_proposals = st.session_state.get("cached_proposals", {})

        if not search_proposals:
            st.info("No proposals loaded yet. Upload PDFs below, or load the built-in samples.")
            upload_col, sample_col = st.columns([2, 1])
            with upload_col:
                search_files = st.file_uploader(
                    "Upload supplier RFP PDFs to search",
                    type=["pdf"], accept_multiple_files=True, key="search_tab_uploader",
                )
                if search_files:
                    st.session_state["cached_proposals"] = {
                        file.name: {
                            "supplier_name": os.path.splitext(file.name)[0].replace("_", " ").title(),
                            "submission_date": date.today().isoformat(),
                            "experience_rating": 5.0,
                            "source_filename": file.name,
                            "bytes": file.getvalue(),
                        } for file in search_files
                    }
                    st.rerun()
            with sample_col:
                st.write("")
                st.write("")
                sample_paths = sorted(glob.glob(os.path.join("sample_pdfs", "*.pdf")))
                if st.button(f"📁 Load {len(sample_paths)} sample proposals",
                             disabled=not sample_paths,
                             help="Synthetic fixtures. Regenerate with python generate_sample_pdfs.py"):
                    loaded = {}
                    for path in sample_paths:
                        filename = os.path.basename(path)
                        with open(path, "rb") as handle:
                            loaded[filename] = {
                                "supplier_name": os.path.splitext(filename)[0].replace("_", " ").title(),
                                "submission_date": date.today().isoformat(),
                                "experience_rating": 5.0,
                                "source_filename": filename,
                                "bytes": handle.read(),
                            }
                    st.session_state["cached_proposals"] = loaded
                    st.rerun()
                if not sample_paths:
                    st.caption("No samples found. Run `python generate_sample_pdfs.py`.")
        else:
            info_col, clear_col = st.columns([4, 1])
            with info_col:
                names = ", ".join(f"`{m['supplier_name']}`" for m in search_proposals.values())
                st.write(f"📁 **Indexed documents ({len(search_proposals)}):** {names}")
            with clear_col:
                if st.button("🔄 Change documents", key="clear_proposals"):
                    st.session_state.pop("cached_proposals", None)
                    st.session_state.pop("search_results", None)
                    st.rerun()

        if search_proposals:
            all_chunks = []
            for filename, meta in search_proposals.items():
                all_chunks.extend(extract_document_chunks(
                    meta["bytes"], filename=filename, supplier_name=meta.get("supplier_name"),
                ))

            st.divider()
            query_col, mode_col = st.columns([3, 2])
            with query_col:
                query = st.text_input(
                    "Search query",
                    value=st.session_state.get("active_search_query", ""),
                    placeholder="e.g. disaster recovery, fixed pricing, ISO 27001, rollback plan…",
                    key="doc_search_query_input",
                )
            with mode_col:
                search_mode = st.radio(
                    "Search mode",
                    ["⚡ Without LLM (BM25)", "🤖 With LLM (deep semantic)"],
                    horizontal=True,
                    help="BM25 is instant, offline and free. LLM mode re-ranks the BM25 shortlist "
                         "and resolves synonyms and negation.",
                )

            st.write("**Quick sample searches:**")
            quick_cols = st.columns(4)
            quick_queries = [
                ("💰 Pricing & support", "fixed implementation fee and annual support cost"),
                ("🛡️ ISO 27001 security", "ISO 27001 compliance, encryption, and audit logs"),
                ("⚙️ Cloud architecture", "cloud-native architecture, REST integrations, disaster recovery"),
                ("⏱️ Timeline & rollback", "implementation timeline, milestones, and rollback plan"),
            ]
            for column, (label, sample) in zip(quick_cols, quick_queries):
                if column.button(label, key=f"quick_{label}"):
                    st.session_state["active_search_query"] = sample
                    st.rerun()

            filter_col, k_col = st.columns([3, 1])
            document_names = sorted({chunk["document_name"] for chunk in all_chunks})
            with filter_col:
                selected_docs = st.multiselect(
                    "Filter documents", options=document_names, default=document_names,
                )
            with k_col:
                top_k = st.slider("Max chunks to show", min_value=3, max_value=15, value=6)

            filtered = [c for c in all_chunks if c["document_name"] in selected_docs]
            effective_query = (query or "").strip() or st.session_state.get("active_search_query", "").strip()

            if st.button("🔍 Run semantic search", type="primary",
                         disabled=not effective_query or not filtered):
                with st.spinner("Searching proposal passages…"):
                    if "With LLM" in search_mode:
                        if not llm_config.get("ready"):
                            st.warning("LLM mode needs an API key. Falling back to BM25.")
                            found = search_without_llm(effective_query, filtered, top_k=top_k)
                        else:
                            found = search_with_llm(effective_query, filtered, llm_config, top_k=top_k)
                    else:
                        found = search_without_llm(effective_query, filtered, top_k=top_k)
                st.session_state["search_results"] = found

            found = st.session_state.get("search_results")
            if found and found.get("results"):
                st.divider()
                ui.section("Search results and document matching", ui.ACCENTS[4])

                if found.get("llm_error"):
                    st.warning(f"The model call failed, so BM25 results are shown: {found['llm_error']}")
                st.caption(f"Mode: {found.get('mode', 'unknown')}")

                top_document = found.get("top_document")
                if top_document:
                    ui.callout(
                        f"🏢 {top_document}",
                        found.get("overall_answer")
                        or found.get("top_document_summary")
                        or "Highest aggregate relevance across matched passages.",
                        badge=f"Match score: {float(found.get('top_document_score', 0)):.1f}%",
                    )

                rankings = found.get("document_rankings", [])
                if len(rankings) > 1:
                    ui.section("Document match comparison", ui.ACCENTS[0])
                    st.dataframe(
                        pd.DataFrame([{
                            "Document": r["document_name"],
                            "Overall match (%)": r["score"],
                            "Peak chunk match (%)": r["max_score"],
                            "Matching chunks": r["match_count"],
                        } for r in rankings]),
                        hide_index=True, width="stretch",
                        column_config={
                            "Overall match (%)": st.column_config.ProgressColumn(
                                "Overall match", min_value=0, max_value=100,
                                format="%.1f%%", color=ui.ACCENTS[0],
                            ),
                            "Peak chunk match (%)": st.column_config.ProgressColumn(
                                "Peak chunk match", min_value=0, max_value=100,
                                format="%.1f%%", color=ui.ACCENTS[1],
                            ),
                        },
                    )

                ui.section("Matching passages with page and line citations", ui.ACCENTS[2])
                query_terms = tokenize(found.get("query", ""))
                for chunk in found["results"]:
                    ui.chunk_card(
                        chunk,
                        query_terms=query_terms,
                        reasoning=chunk.get("reasoning"),
                        matched_quote=chunk.get("matched_quote"),
                    )
            elif found:
                st.info("No passages matched that query. Try broader wording, or switch to LLM mode "
                        "for synonym and negation handling.")
