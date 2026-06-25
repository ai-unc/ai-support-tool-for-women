"""
dashboard.py — Streamlit analytics dashboard for the Maternal Breastfeeding Support tool.

Run:
    streamlit run dashboard.py

Reads from analytics.db (created by analytics.py).
To populate with demo data first: python analytics.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

DB_PATH = os.environ.get("ANALYTICS_DB", "analytics.db")


st.set_page_config(
    page_title="Breastfeeding Support — Analytics",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Re-engineered Professional Dark-Mode CSS Layout ───────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

  /* Re-align primary viewport container */
  html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0F172A !important;
  }
  
  h1, h2, h3 { 
    font-family: 'DM Serif Display', serif; 
  }

  /* Centered, high-depth metric card layout grids */
  .metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  .metric-card:hover {
    border-color: #f472b6;
    transform: translateY(-2px);
  }
  .metric-card .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94A3B8;
    margin-bottom: 8px;
    font-weight: 500;
  }
  .metric-card .value {
    font-size: 38px;
    font-family: 'DM Serif Display', serif;
    color: #F8FAFC;
    line-height: 1.1;
    font-weight: 600;
  }
  .metric-card .sub {
    font-size: 12px;
    color: #64748B;
    margin-top: 6px;
  }

  /* Clean minimal section layout grids */
  .section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 24px;
    color: #F8FAFC;
    margin: 40px 0 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid #334155;
  }

  /* Structural Streamlit Overrides */
  [data-testid="stMetric"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 16px;
  }
  div[data-testid="stExpander"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    if not os.path.exists(DB_PATH):
        return None, None

    conn = sqlite3.connect(DB_PATH)
    turns = pd.read_sql("SELECT * FROM turns", conn, parse_dates=["timestamp"])
    sessions = pd.read_sql("SELECT * FROM sessions", conn, parse_dates=["started_at", "ended_at"])
    conn.close()
    return turns, sessions


turns, sessions = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 24px 0 4px'>
  <h1 style='font-size:40px; color:#F8FAFC; margin:0; font-weight:400;'>🌸 Breastfeeding Support</h1>
  <p style='color:#f472b6; font-size:14px; margin:6px 0 0; letter-spacing:0.15em; font-weight:500;'>
    CONVERSATION OBSERVABILITY & METRIC ANALYTICS
  </p>
</div>
""", unsafe_allow_html=True)

if turns is None or turns.empty:
    st.warning("No data yet. Run `python analytics.py` to seed demo data, or start the app and have some conversations.")
    st.stop()

# ── Date filter ───────────────────────────────────────────────────────────────
col_filter1, col_filter2, _ = st.columns([1.5, 1.5, 4])
with col_filter1:
    date_range = st.selectbox("Time range Scope", ["Last 7 days", "Last 30 days", "All time"], index=1)

cutoff = {
    "Last 7 days": datetime.utcnow() - timedelta(days=7),
    "Last 30 days": datetime.utcnow() - timedelta(days=30),
    "All time": datetime(2000, 1, 1),
}[date_range]

t = turns[turns["timestamp"] >= cutoff].copy()
s = sessions[sessions["started_at"] >= cutoff].copy()

# ── KPI Cards (Centered and Text-Balanced Layout Fixed) ───────────────────────
st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

total_sessions = len(s)
total_turns = len(t)
urgent_rate = s["had_urgent"].mean() * 100 if len(s) else 0
support_rate = s["had_support"].mean() * 100 if len(s) else 0
avg_turns = s["total_turns"].mean() if len(s) else 0
retrieval_hit_rate = t["retrieval_hit"].mean() * 100 if len(t) else 0

def kpi(col, label, value, sub=""):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

kpi(k1, "Total Sessions", total_sessions, f"Scope: {date_range.lower()}")
kpi(k2, "Total Turns", total_turns, f"Avg {avg_turns:.1f} turns / session")
kpi(k3, "Urgency Rate", f"{urgent_rate:.1f}%", "Flagged safe urgent redirections")
kpi(k4, "Support Rate", f"{support_rate:.1f}%", "Empathetic tracking sessions")
kpi(k5, "Retrieval Hit", f"{retrieval_hit_rate:.1f}%", "RAG baseline knowledge hits")


# ── Plotly Unified Styling Framework ──────────────────────────────────────────
# Re-usable theme properties to handle canvas rendering
DARK_THEME_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="DM Sans",
    font_color="#94A3B8",
    title_font_family="DM Serif Display",
    title_font_color="#F8FAFC",
    title_font_size=18,
    margin=dict(t=60, b=30, l=40, r=40),
)

colors = {
    "CLINICAL":       "#f472b6", # Modern Tailwinds Pink
    "SUPPORT":        "#a78bfa", # Light modern Violet
    "URGENT":         "#f87171", # Safe soft Coral red
    "QUESTION_FIRST": "#fb923c", # Amber orange
    "CLOSING":        "#34d399", # Emerald Mint Green
    "GENERAL":        "#60a5fa", # Indigo soft blue
    "URGENT_INFANT":  "#f87171",
    "URGENT_MATERNAL":"#fda4af"
}


# ── Route Distribution ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Conversation Routing Performance</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([1, 1])

with col_a:
    route_counts = t["route"].value_counts().reset_index()
    route_counts.columns = ["Route", "Count"]
    route_counts["color"] = route_counts["Route"].map(colors).fillna("#64748B")

    fig_pie = px.pie(
        route_counts, names="Route", values="Count",
        color="Route", color_discrete_map=colors,
        hole=0.6, # Slimmed modern donut design configuration
        title="Intent Routing Engine Distribution"
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
    fig_pie.update_layout(**DARK_THEME_LAYOUT)
    fig_pie.update_layout(showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    t["date"] = t["timestamp"].dt.date
    route_by_day = t.groupby(["date", "route"]).size().reset_index(name="count")

    fig_bar = px.bar(
        route_by_day, x="date", y="count", color="route",
        color_discrete_map=colors,
        title="Temporal Pipeline Volume Stacked by Route Labels",
        labels={"date": "", "count": "Processed Turns", "route": "Resolved Layer"},
    )
    fig_bar.update_layout(**DARK_THEME_LAYOUT)
    fig_bar.update_layout(
        legend_title_text="",
        xaxis=dict(showgrid=False, linecolor="#334155"),
        yaxis=dict(gridcolor="#1E293B", zeroline=False),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── Semantic Scores ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Real-Time Vector Similarity Observations</div>', unsafe_allow_html=True)

col_c, col_d = st.columns([1, 1])

with col_c:
    score_cols = ["score_pain", "score_latch", "score_supply", "score_stress", "score_urgency"]
    score_labels = ["Pain", "Latch", "Supply", "Stress", "Urgency"]
    avg_scores = [t[c].mean() for c in score_cols]

    fig_radar = go.Figure(go.Scatterpolar(
        r=avg_scores + [avg_scores[0]],
        theta=score_labels + [score_labels[0]],
        fill="toself",
        fillcolor="rgba(244, 114, 182, 0.15)",
        line=dict(color="#f472b6", width=2),
        marker=dict(color="#f472b6", size=6),
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 0.7], gridcolor="#334155", color="#94A3B8", tickvals=[0.2, 0.4, 0.6]),
            angularaxis=dict(gridcolor="#334155", color="#94A3B8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#94A3B8",
        title="Mean Semantic Vector Similarities Across Categories",
        title_font_family="DM Serif Display",
        title_font_color="#F8FAFC",
        title_font_size=18,
        margin=dict(t=70, b=40, l=60, r=60), # Padded to completely prevent layout clipping
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_d:
    flag_cols = ["flag_pain", "flag_latch", "flag_supply", "flag_stress", "flag_urgency"]
    flag_labels = ["Pain", "Latch", "Supply", "Stress", "Urgency"]
    flag_rates = [t[c].mean() * 100 for c in flag_cols]

    fig_flags = px.bar(
        x=flag_labels, y=flag_rates,
        labels={"x": "Extracted Domain Class", "y": "% of Turns Over Threshold"},
        title="Static Engine Threshold Execution Frequency",
        color=flag_labels,
        color_discrete_sequence=["#f472b6", "#a78bfa", "#fb923c", "#f87171", "#34d399"],
    )
    fig_flags.update_layout(**DARK_THEME_LAYOUT)
    fig_flags.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, linecolor="#334155"),
        yaxis=dict(gridcolor="#1E293B", ticksuffix="%", zeroline=False),
    )
    st.plotly_chart(fig_flags, use_container_width=True)


# ── Retrieval + Session Length ────────────────────────────────────────────────
st.markdown('<div class="section-header">Context Injection Performance & Observability Depth</div>', unsafe_allow_html=True)

col_e, col_f = st.columns([1, 1])

with col_e:
    retrieval_turns = t[t["retrieval_top_score"].notna()].copy()
    if not retrieval_turns.empty:
        fig_ret = px.histogram(
            retrieval_turns, x="retrieval_top_score",
            nbins=20,
            title="RAG Context Retrieval Confidence Distribution",
            labels={"retrieval_top_score": "FAISS Index Cosine Score Match", "count": "Resolved Logs"},
            color_discrete_sequence=["#f472b6"],
        )
        fig_ret.add_vline(x=0.5, line_dash="dash", line_color="#94A3B8",
                          annotation_text="Optimized Safety Cutoff (0.50)",
                          annotation_position="top right")
        fig_ret.update_layout(**DARK_THEME_LAYOUT)
        fig_ret.update_layout(
            xaxis=dict(showgrid=False, linecolor="#334155"),
            yaxis=dict(gridcolor="#1E293B", zeroline=False),
        )
        st.plotly_chart(fig_ret, use_container_width=True)
    else:
        st.info("No query retrieval data captured yet within scope criteria.")

with col_f:
    session_lengths = s["total_turns"].value_counts().sort_index().reset_index()
    session_lengths.columns = ["Turns", "Sessions"]

    fig_len = px.bar(
        session_lengths, x="Turns", y="Sessions",
        title="Structural Session Turn Depth Volatility",
        color_discrete_sequence=["#a78bfa"],
        labels={"Turns": "Exchanged Messages Per Active Instance", "Sessions": "Evaluated Count"},
    )
    fig_len.update_layout(**DARK_THEME_LAYOUT)
    fig_len.update_layout(
        xaxis=dict(showgrid=False, linecolor="#334155", dtick=1),
        yaxis=dict(gridcolor="#1E293B", zeroline=False),
    )
    st.plotly_chart(fig_len, use_container_width=True)


# ── Hourly Urgency Heatmap ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Temporal Stress & System Triage Maps</div>', unsafe_allow_html=True)

urgent_turns = t[t["route"].str.contains("URGENT", na=False)].copy()
if not urgent_turns.empty:
    urgent_turns["hour"] = urgent_turns["timestamp"].dt.hour
    urgent_turns["day"] = urgent_turns["timestamp"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = urgent_turns.groupby(["day", "hour"]).size().reset_index(name="count")
    heatmap_pivot = heatmap_data.pivot(index="day", columns="hour", values="count").fillna(0)
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])

    fig_heat = px.imshow(
        heatmap_pivot,
        color_continuous_scale=["#1E293B", "#f472b6"],
        title="Escalated Patient Risk Signals Plotted Chronologically",
        labels={"x": "Hour Profile (UTC)", "y": "", "color": "Triage Trigger Count"},
        aspect="auto",
    )
    fig_heat.update_layout(**DARK_THEME_LAYOUT)
    fig_heat.update_layout(coloraxis_showscale=True)
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.markdown("""
    <div style="background: #1E293B; border: 1px dashed #334155; padding: 24px; text-align: center; border-radius: 12px; color: #94A3B8;">
        No high-risk urgent safety mitigations triggered in this active timestamp window frame.
    </div>
    """, unsafe_allow_html=True)


# ── Raw data table ────────────────────────────────────────────────────────────
st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
with st.expander("📋 Audit Log Pipeline Records (Raw Execution Data Frames)"):
    display_cols = [
        "timestamp", "session_id", "turn_number", "route",
        "score_pain", "score_latch", "score_supply", "score_stress", "score_urgency",
        "retrieval_hit", "retrieval_top_score", "user_msg_len", "baby_age_known"
    ]
    available = [c for c in display_cols if c in t.columns]
    st.dataframe(
        t[available].sort_values("timestamp", ascending=False).head(200),
        use_container_width=True,
        hide_index=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:64px; padding:24px 0; border-top:1px solid #334155;
     color:#64748B; font-size:12px; text-align:center; letter-spacing: 0.05em;'>
  Maternal Clinical Support Analytics Engine v1.2.0 • Data Frame State Polling Active (30s Cache TTL)
</div>
""", unsafe_allow_html=True)