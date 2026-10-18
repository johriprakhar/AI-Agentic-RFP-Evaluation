"""Presentation helpers: colour palette, CSS theme, cards and Altair charts."""

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
.stTabs [data-baseweb="tab-list"] {
    gap:.45rem; background:#FFFFFF; padding:.42rem; border-radius:14px;
    border:1px solid #E7EAF6; box-shadow:0 3px 14px rgba(30,34,53,.05);
}
.stTabs [data-baseweb="tab"] {
    height:44px; border-radius:11px; padding:0 20px;
    font-weight:650; color:#49506A; background:transparent;
}
.stTabs [data-baseweb="tab"]:hover { background:#F1F3FF; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#4F46E5,#06B6D4); }
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] div { color:#FFFFFF !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { background:transparent; }

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

/* ---------- Status pills ---------- */
.rfp-pill {
    display:inline-flex; align-items:center; gap:.35rem;
    font-size:.78rem; font-weight:700; padding:.28rem .66rem; border-radius:999px;
}
.rfp-pill-ok   { background:#E7F8F0; color:#0A7A52; border:1px solid #A9E5CC; }
.rfp-pill-warn { background:#FFF3E0; color:#9A5B00; border:1px solid #FFD79A; }
.rfp-pill-info { background:#EEF1FF; color:#3A33B8; border:1px solid #C3CAFF; }

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
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea { color:#15182B !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:#FFFFFF; border-radius:10px; border-color:#FFFFFF;
}
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] span { color:#15182B !important; }
section[data-testid="stSidebar"] [data-baseweb="input"] { border-radius:10px; }
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


def _arc_chart(df, category, value, inner_radius, title, value_title, value_format=".1f"):
    base = alt.Chart(df).encode(
        theta=alt.Theta(f"{value}:Q", stack=True),
        color=alt.Color(
            f"{category}:N",
            scale=alt.Scale(range=PALETTE),
            legend=alt.Legend(title=None, orient="bottom", columns=2, labelLimit=190),
        ),
        order=alt.Order(f"{value}:Q", sort="descending"),
        tooltip=[
            alt.Tooltip(f"{category}:N", title=category),
            alt.Tooltip(f"{value}:Q", title=value_title, format=value_format),
        ],
    )
    arcs = base.mark_arc(
        innerRadius=inner_radius,
        outerRadius=118,
        stroke="#FFFFFF",
        strokeWidth=2,
        cornerRadius=3,
    )
    return arcs.properties(height=330, title=title)


def donut_chart(df, category, value, title, value_title="Value", value_format=".1f"):
    return _arc_chart(df, category, value, 68, title, value_title, value_format)


def pie_chart(df, category, value, title, value_title="Value", value_format=".1f"):
    return _arc_chart(df, category, value, 0, title, value_title, value_format)


def hbar_chart(df, category, value, title, value_title="Value", value_format=".1f"):
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=7, height=22)
        .encode(
            x=alt.X(f"{value}:Q", title=value_title, axis=alt.Axis(grid=True, gridDash=[2, 3])),
            y=alt.Y(f"{category}:N", sort="-x", title=None),
            color=alt.Color(f"{category}:N", scale=alt.Scale(range=PALETTE), legend=None),
            tooltip=[
                alt.Tooltip(f"{category}:N", title=category),
                alt.Tooltip(f"{value}:Q", title=value_title, format=value_format),
            ],
        )
        .properties(height=max(150, 44 * max(len(df), 1)), title=title)
    )


def grouped_bar_chart(df, x, y, color, title, value_title="Score"):
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(f"{x}:N", title=None, axis=alt.Axis(labelAngle=-25)),
            y=alt.Y(f"{y}:Q", title=value_title),
            color=alt.Color(f"{color}:N", scale=alt.Scale(range=PALETTE),
                            legend=alt.Legend(title=None, orient="bottom")),
            xOffset=alt.XOffset(f"{color}:N"),
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{color}:N"),
                     alt.Tooltip(f"{y}:Q", format=".2f", title=value_title)],
        )
        .properties(height=360, title=title)
    )
