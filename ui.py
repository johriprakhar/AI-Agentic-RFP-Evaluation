"""Presentation helpers: colour palette, CSS theme, cards and Altair charts."""

import html
import re

import altair as alt
import streamlit as st

# Accessible, high-contrast categorical palette used across every chart.
PALETTE = [
    "#4F46E5",  # indigo
    "#06B6D4",  # cyan
    "#F59E0B",  # amber
    "#EC4899",  # pink
    "#10B981",  # emerald
    "#8B5CF6",  # violet
    "#EF4444",  # red
    "#0EA5E9",  # sky
]

ACCENTS = ["#4F46E5", "#06B6D4", "#F59E0B", "#EC4899", "#10B981", "#8B5CF6"]

_CSS = """
<style>
.stApp {
    background:
        radial-gradient(1200px 500px at 8% -8%, #E8ECFF 0%, rgba(232,236,255,0) 60%),
        radial-gradient(1000px 500px at 95% 0%, #FFE9F3 0%, rgba(255,233,243,0) 55%),
        linear-gradient(180deg, #F7F9FF 0%, #FFFFFF 55%, #F4FBFF 100%);
}
.block-container { padding-top: 1.6rem; }

/* ---------- Hero banner ---------- */
.rfp-hero {
    background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 42%, #06B6D4 100%);
    border-radius: 20px;
    padding: 22px 26px;
    color: #FFFFFF;
    box-shadow: 0 14px 36px rgba(79, 70, 229, 0.26);
    margin-bottom: 1.1rem;
}
.rfp-hero h1 { color:#FFFFFF; font-size: 1.85rem; font-weight: 800; margin: 0 0 .35rem 0; letter-spacing:-.4px; }
.rfp-hero p { color: rgba(255,255,255,.92); font-size: .97rem; margin: 0 0 .75rem 0; }
.rfp-chips { display:flex; flex-wrap:wrap; gap:.45rem; }
.rfp-chip {
    background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.34);
    color:#FFFFFF; font-size:.78rem; font-weight:600;
    padding:.26rem .66rem; border-radius:999px; backdrop-filter: blur(4px);
}

/* ---------- Metric cards ---------- */
.rfp-card {
    background:#FFFFFF; border:1px solid #E7EAF6; border-radius:16px;
    padding:14px 16px; box-shadow:0 4px 16px rgba(30,34,53,.06);
    border-top:4px solid var(--accent, #4F46E5); height:100%;
}
.rfp-card .rfp-card-label {
    font-size:.74rem; font-weight:700; letter-spacing:.07em;
    text-transform:uppercase; color:#5B6478; margin-bottom:.3rem;
}
.rfp-card .rfp-card-value { font-size:1.62rem; font-weight:800; color:#1E2235; line-height:1.15; }
.rfp-card .rfp-card-sub { font-size:.8rem; color:#6B7488; margin-top:.2rem; }

/* ---------- Section headings ---------- */
.rfp-section { display:flex; align-items:center; gap:.55rem; margin:.4rem 0 .65rem 0; }
.rfp-section .rfp-dot { width:10px; height:10px; border-radius:50%; background:var(--accent,#4F46E5); }
.rfp-section h3 { margin:0; font-size:1.08rem; font-weight:750; color:#1E2235; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"],
.stTabs [role="tablist"],
[data-testid="stTabs"] [role="tablist"] {
    display: inline-flex;
    gap: .45rem;
    background: #FFFFFF;
    padding: .42rem;
    border-radius: 14px;
    border: 1px solid #E7EAF6;
    box-shadow: 0 3px 14px rgba(30,34,53,.05);
    align-items: center;
}
.stTabs [role="tablist"]::after,
[data-testid="stTabs"] [role="tablist"]::after {
    display: none !important;
}

.stTabs [data-baseweb="tab"],
.stTabs [data-testid="stTab"],
.stTabs [role="tab"],
[data-testid="stTabs"] [data-testid="stTab"] {
    height: 42px;
    border-radius: 11px;
    padding: 0 20px;
    font-weight: 650;
    color: #49506A;
    background: transparent;
    border: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
    cursor: pointer;
}
.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-testid="stTab"]:hover,
.stTabs [role="tab"]:hover {
    background: #F1F3FF;
    color: #4F46E5;
}
.stTabs [aria-selected="true"],
.stTabs [data-selected="true"],
[data-testid="stTabs"] [aria-selected="true"],
[data-testid="stTabs"] [data-selected="true"] {
    background: linear-gradient(135deg, #4F46E5, #06B6D4) !important;
    border-radius: 11px !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] div,
.stTabs [aria-selected="true"] span,
.stTabs [data-selected="true"] p,
.stTabs [data-selected="true"] div,
.stTabs [data-selected="true"] span,
[data-testid="stTabs"] [aria-selected="true"] p,
[data-testid="stTabs"] [aria-selected="true"] div,
[data-testid="stTabs"] [aria-selected="true"] span {
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"],
.stTabs .react-aria-SelectionIndicator,
[data-testid="stTabs"] .react-aria-SelectionIndicator {
    display: none !important;
    background: transparent !important;
    height: 0 !important;
}

/* ---------- Buttons ---------- */
.stButton > button { border-radius:11px; font-weight:650; }
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#4F46E5 0%,#06B6D4 100%);
    border:0; color:#FFFFFF; box-shadow:0 6px 18px rgba(79,70,229,.3);
}
.stButton > button[kind="primary"]:hover { filter:brightness(1.07); }
.stDownloadButton > button {
    border-radius:11px; font-weight:650;
    background:linear-gradient(135deg,#10B981 0%,#06B6D4 100%); color:#FFFFFF; border:0;
}

/* ---------- Containers ---------- */
[data-testid="stExpander"] {
    border:1px solid #E7EAF6; border-radius:14px; background:#FFFFFF;
    box-shadow:0 3px 12px rgba(30,34,53,.05); overflow:hidden;
}
[data-testid="stFileUploaderDropzone"] {
    border:2px dashed #A6B0FF; border-radius:16px;
    background:linear-gradient(135deg,#F6F7FF 0%,#F3FDFF 100%);
}
[data-testid="stMetric"] {
    background:#FFFFFF; border:1px solid #E7EAF6; border-radius:14px; padding:12px 14px;
}

/* ---------- Chunk Viewer ---------- */
.rfp-chunk-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(15,23,42,.04);
}
.rfp-chunk-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid #F1F5F9;
}
.rfp-chunk-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    font-size: 0.84rem;
    font-weight: 600;
    color: #475569;
}
.rfp-chunk-badge {
    background: #EEF2FF;
    color: #4338CA;
    padding: 3px 10px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 700;
    border: 1px solid #C7D2FE;
}
.rfp-chunk-box {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 12px 14px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #E2E8F0;
    overflow-x: auto;
}
.rfp-chunk-line {
    display: flex;
    align-items: baseline;
    gap: 12px;
}
.rfp-line-num {
    color: #64748B;
    user-select: none;
    font-weight: 600;
    min-width: 42px;
    text-align: right;
    font-size: 0.8rem;
}
.rfp-line-text {
    flex: 1;
    color: #F8FAFC;
    word-break: break-word;
}
.rfp-match-mark {
    background-color: #FEF08A;
    color: #854D0E;
    padding: 1px 4px;
    border-radius: 4px;
    font-weight: 700;
}
.rfp-reasoning-box {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #10B981;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.84rem;
    color: #065F46;
    margin-bottom: 10px;
}

/* ---------- Status pills ---------- */
.rfp-pill {
    display:inline-flex; align-items:center; gap:.35rem;
    font-size:.78rem; font-weight:700; padding:.28rem .66rem; border-radius:999px;
}
.rfp-pill-ok,
section[data-testid="stSidebar"] .rfp-pill-ok {
    background:#E7F8F0 !important; color:#0A7A52 !important; border:1px solid #A9E5CC !important;
}
.rfp-pill-warn,
section[data-testid="stSidebar"] .rfp-pill-warn {
    background:#FFF3E0 !important; color:#9A5B00 !important; border:1px solid #FFD79A !important;
}
.rfp-pill-info,
section[data-testid="stSidebar"] .rfp-pill-info {
    background:#EEF1FF !important; color:#3A33B8 !important; border:1px solid #C3CAFF !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#221F57 0%,#312E81 48%,#0E7490 100%);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li { color:#EEF0FF !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#C7CCF5 !important; }
section[data-testid="stSidebar"] a {
    color: #38BDF8 !important;
    text-decoration: underline !important;
    text-underline-offset: 2px;
    font-weight: 600;
}
section[data-testid="stSidebar"] a:hover {
    color: #7DD3FC !important;
}
section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
section[data-testid="stSidebar"] [data-testid="stHelpIcon"] svg {
    fill: #A5B4FC !important;
    color: #A5B4FC !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #FFFFFF !important;
    border: 1px solid #D5DAE7 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
    transition: all 0.15s ease;
}
section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button span,
section[data-testid="stSidebar"] .stButton > button div {
    color: #1E2235 !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #EEF2FF !important;
    border-color: #A5B4FC !important;
}
section[data-testid="stSidebar"] .stButton > button:hover p,
section[data-testid="stSidebar"] .stButton > button:hover span {
    color: #4F46E5 !important;
}
section[data-testid="stSidebar"] [data-testid="stAlert"] p,
section[data-testid="stSidebar"] [data-testid="stAlert"] span {
    color: #1E2235 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.22) !important;
    margin: 1rem 0 !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    color: #15182B !important;
    -webkit-text-fill-color: #15182B !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:#FFFFFF; border-radius:10px; border-color:#FFFFFF;
}
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] span { color:#15182B !important; }
section[data-testid="stSidebar"] [data-baseweb="input"] {
    border-radius: 10px;
    background: #FFFFFF !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"] input::placeholder {
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8 !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"] button svg {
    fill: #64748B !important;
    color: #64748B !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #15182B !important;
    border-radius: 10px;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.22);
}
section[data-testid="stSidebar"] [data-testid="stExpander"] p,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary { color:#EEF0FF !important; }
.rfp-sidebar-brand {
    background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.26);
    border-radius:14px; padding:12px 14px; margin-bottom:.8rem;
}
.rfp-sidebar-brand .rfp-brand-title { font-size:1rem; font-weight:800; color:#FFFFFF; }
.rfp-sidebar-brand .rfp-brand-sub { font-size:.78rem; color:#CFD4FA; }
</style>
"""


def inject_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title, subtitle, chips=()):
    chip_html = "".join(f"<span class='rfp-chip'>{c}</span>" for c in chips)
    st.markdown(
        f"""
        <div class="rfp-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="rfp-chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, sub="", accent="#4F46E5"):
    st.markdown(
        f"""
        <div class="rfp-card" style="--accent:{accent}">
            <div class="rfp-card-label">{label}</div>
            <div class="rfp-card-value">{value}</div>
            <div class="rfp-card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, accent="#4F46E5"):
    st.markdown(
        f"""
        <div class="rfp-section" style="--accent:{accent}">
            <span class="rfp-dot"></span><h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text, kind="info"):
    st.markdown(f"<span class='rfp-pill rfp-pill-{kind}'>{text}</span>", unsafe_allow_html=True)


def chunk_card(chunk, query_terms=None, reasoning=None, matched_quote=None):
    """Render a document chunk with line numbers, page number, match badge, and optional highlight."""
    score = float(chunk.get("score", 0))
    score_color = "#10B981" if score >= 75 else ("#F59E0B" if score >= 45 else "#64748B")

    # Format lines with line numbers and optional term highlighting
    terms = [re.escape(t) for t in (query_terms or []) if len(t) > 1]
    pattern = re.compile(rf"\b({'|'.join(terms)})\b", re.IGNORECASE) if terms else None

    line_items_html = []
    for item in chunk.get("lines", []):
        num = item["line_num"]
        raw_text = html.escape(item["text"])
        if pattern:
            raw_text = pattern.sub(r"<mark class='rfp-match-mark'>\1</mark>", raw_text)
        line_items_html.append(
            f'<div class="rfp-chunk-line">'
            f'<span class="rfp-line-num">L{num} |</span>'
            f'<span class="rfp-line-text">{raw_text}</span>'
            f'</div>'
        )
    lines_body = "".join(line_items_html)

    reasoning_html = ""
    if reasoning:
        escaped_reasoning = html.escape(reasoning)
        reasoning_html = f"<div class='rfp-reasoning-box'>💡 <b>Semantic Match:</b> {escaped_reasoning}</div>"

    quote_html = ""
    if matched_quote:
        escaped_quote = html.escape(matched_quote)
        quote_html = f"<div style='font-size:0.83rem; color:#475569; margin-bottom:8px;'>📌 <b>Key Evidence:</b> <i>\"{escaped_quote}\"</i></div>"

    st.markdown(
        f"""
        <div class="rfp-chunk-card">
            <div class="rfp-chunk-header">
                <div class="rfp-chunk-meta">
                    <span class="rfp-chunk-badge">🏢 {html.escape(str(chunk.get('document_name', '')))}</span>
                    <span>📄 Page {chunk.get('page', 1)}</span>
                    <span>📍 Lines {chunk.get('start_line', 1)}–{chunk.get('end_line', 1)}</span>
                </div>
                <div style="font-weight:800; font-size:0.92rem; color:{score_color};">
                    🎯 {score:.1f}% Match
                </div>
            </div>
            {reasoning_html}
            {quote_html}
            <div class="rfp-chunk-box">
                {lines_body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _arc_chart(df, category, value, inner_radius, title, value_title, value_format=".1f"):
    base = alt.Chart(df).encode(
        theta=alt.Theta(f"{value}:Q", stack=True),
        color=alt.Color(
            f"{category}:N",
            scale=alt.Scale(range=PALETTE),
            legend=alt.Legend(
                title=None,
                orient="bottom",
                columns=2,
                labelLimit=240,
                rowPadding=6,
                columnPadding=16,
                symbolType="circle",
            ),
        ),
        order=alt.Order(f"{value}:Q", sort="descending"),
        tooltip=[
            alt.Tooltip(f"{category}:N", title=category),
            alt.Tooltip(f"{value}:Q", title=value_title, format=value_format),
        ],
    )
    arcs = base.mark_arc(
        innerRadius=inner_radius,
        outerRadius=86,
        stroke="#FFFFFF",
        strokeWidth=2,
        cornerRadius=3,
    )
    return arcs.properties(
        height=320,
        title=alt.TitleParams(
            text=title,
            anchor="start",
            fontSize=13,
            fontWeight=600,
            color="#475569",
            offset=12,
        ),
        padding={"top": 12, "bottom": 12, "left": 10, "right": 10},
    )


def donut_chart(df, category, value, title, value_title="Value", value_format=".1f"):
    return _arc_chart(df, category, value, 50, title, value_title, value_format)


def pie_chart(df, category, value, title, value_title="Value", value_format=".1f"):
    return _arc_chart(df, category, value, 0, title, value_title, value_format)


def hbar_chart(df, category, value, title, value_title="Value", value_format=".0f"):
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=6, height=20)
        .encode(
            x=alt.X(
                f"{value}:Q",
                title=value_title,
                axis=alt.Axis(grid=True, gridDash=[2, 3], format=value_format, tickMinStep=1),
            ),
            y=alt.Y(
                f"{category}:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=300, labelFontSize=12),
                scale=alt.Scale(paddingInner=0.35),
            ),
            color=alt.Color(f"{category}:N", scale=alt.Scale(range=PALETTE), legend=None),
            tooltip=[
                alt.Tooltip(f"{category}:N", title=category),
                alt.Tooltip(f"{value}:Q", title=value_title, format=value_format),
            ],
        )
        .properties(
            height=max(180, 42 * max(len(df), 1)),
            title=alt.TitleParams(
                text=title,
                anchor="start",
                fontSize=13,
                fontWeight=600,
                color="#475569",
                offset=12,
            ),
            padding={"top": 12, "bottom": 12, "left": 10, "right": 20},
        )
    )


def grouped_bar_chart(df, x, y, color, title, value_title="Score"):
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(f"{x}:N", title=None, axis=alt.Axis(labelAngle=-25, labelLimit=200)),
            y=alt.Y(f"{y}:Q", title=value_title),
            color=alt.Color(
                f"{color}:N",
                scale=alt.Scale(range=PALETTE),
                legend=alt.Legend(title=None, orient="bottom", rowPadding=6, columnPadding=16),
            ),
            xOffset=alt.XOffset(f"{color}:N"),
            tooltip=[
                alt.Tooltip(f"{x}:N"),
                alt.Tooltip(f"{color}:N"),
                alt.Tooltip(f"{y}:Q", format=".2f", title=value_title),
            ],
        )
        .properties(
            height=360,
            title=alt.TitleParams(
                text=title,
                anchor="start",
                fontSize=13,
                fontWeight=600,
                color="#475569",
                offset=12,
            ),
            padding={"top": 12, "bottom": 12, "left": 10, "right": 10},
        )
    )
