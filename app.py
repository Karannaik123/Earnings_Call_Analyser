import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# ── Page Config ────────────────────────────────
st.set_page_config(
    page_title="Earnings Call Analyzer",
    page_icon="📊",
    layout="wide"
)

# ── Load Data ───────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    mgmt = pd.read_csv(os.path.join(base, "data/management_sentiment.csv"))
    topics = pd.read_csv(os.path.join(base, "data/topic_assignments.csv"))
    structured = pd.read_csv(os.path.join(base, "data/structured_utterances.csv"))
    return mgmt, topics, structured

mgmt_df, topic_df, struct_df = load_data()

# ── Sidebar ─────────────────────────────────────
st.sidebar.title("📊 Earnings Call Analyzer")
st.sidebar.markdown("**MBA BA05 — NLP Project**")
st.sidebar.markdown("IIM Amritsar | FY25")
st.sidebar.divider()

companies = sorted(mgmt_df['ticker'].dropna().unique())
selected_companies = st.sidebar.multiselect(
    "Select Companies",
    options=companies,
    default=companies
)

quarters = ['Q1FY25', 'Q2FY25', 'Q3FY25', 'Q4FY25']
selected_quarters = st.sidebar.multiselect(
    "Select Quarters",
    options=quarters,
    default=quarters
)

st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 Sentiment Analysis",
     "🗂️ Topic Modelling", "🔍 Statement Explorer",
     "📋 Summary Report"]
)

# ── Filter Data ─────────────────────────────────
filtered_mgmt = mgmt_df[
    (mgmt_df['ticker'].isin(selected_companies)) &
    (mgmt_df['quarter'].isin(selected_quarters))
]
filtered_topics = topic_df[
    (topic_df['ticker'].isin(selected_companies)) &
    (topic_df['quarter'].isin(selected_quarters))
]

# ── PAGE 1: Overview ────────────────────────────
if page == "🏠 Overview":
    st.title("📊 Earnings Call Sentiment & Topic Analyzer")
    st.markdown("**Automated analysis of quarterly earnings call transcripts using FinBERT + BERTopic**")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies", len(selected_companies))
    col2.metric("Transcripts", len(selected_companies) * len(selected_quarters))
    col3.metric("Utterances", len(filtered_mgmt))
    col4.metric("Topics Found", topic_df['topic_label'].nunique())

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sent_counts = filtered_mgmt['sentiment'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        fig = px.pie(sent_counts, values='Count', names='Sentiment',
                     color='Sentiment', color_discrete_map=colors)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Utterances by Company")
        company_counts = filtered_mgmt['ticker'].value_counts().reset_index()
        company_counts.columns = ['Company', 'Count']
        fig = px.bar(company_counts, x='Company', y='Count',
                     color='Count', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

# ── PAGE 2: Sentiment Analysis ──────────────────
elif page == "📈 Sentiment Analysis":
    st.title("📈 Sentiment Analysis — FinBERT")
    st.divider()

    quarter_order = ['Q1FY25', 'Q2FY25', 'Q3FY25', 'Q4FY25']

    for company in selected_companies:
        data = filtered_mgmt[filtered_mgmt['ticker'] == company].groupby('quarter')[
            ['positive_score', 'negative_score', 'neutral_score']
        ].mean().reindex([q for q in quarter_order if q in selected_quarters])

        if data.empty:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['positive_score'],
                                  mode='lines+markers', name='Positive',
                                  line=dict(color='#2ecc71', width=2)))
        fig.add_trace(go.Scatter(x=data.index, y=data['negative_score'],
                                  mode='lines+markers', name='Negative',
                                  line=dict(color='#e74c3c', width=2)))
        fig.add_trace(go.Scatter(x=data.index, y=data['neutral_score'],
                                  mode='lines+markers', name='Neutral',
                                  line=dict(color='#95a5a6', width=2,
                                            dash='dash')))
        fig.update_layout(
            title=f'{company} — Management Sentiment Trend FY25',
            yaxis_range=[0, 1],
            height=300,
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Sentiment Comparison Across Companies")
    comparison = filtered_mgmt.groupby('ticker')[
        ['positive_score', 'negative_score', 'neutral_score']
    ].mean().round(3)
    st.dataframe(comparison, use_container_width=True)

# ── PAGE 3: Topic Modelling ─────────────────────
elif page == "🗂️ Topic Modelling":
    st.title("🗂️ Topic Distribution — BERTopic")
    st.divider()

    clean_topics = [t for t in filtered_topics['topic_label'].unique()
                    if t != 'General Discussion']
    clean_df = filtered_topics[filtered_topics['topic_label'].isin(clean_topics)]

    crosstab = pd.crosstab(clean_df['ticker'], clean_df['topic_label'])

    fig = px.imshow(
        crosstab,
        color_continuous_scale='YlOrRd',
        text_auto=True,
        title='Topic Distribution by Company'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Topic Distribution Breakdown")
    topic_counts = filtered_topics['topic_label'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig2 = px.bar(topic_counts, x='Count', y='Topic',
                  orientation='h', color='Count',
                  color_continuous_scale='Reds')
    st.plotly_chart(fig2, use_container_width=True)

# ── PAGE 4: Statement Explorer ──────────────────
elif page == "🔍 Statement Explorer":
    st.title("🔍 Statement Explorer")
    st.divider()

    col1, col2, col3 = st.columns(3)
    company_filter = col1.selectbox("Company", ["All"] + list(companies))
    sentiment_filter = col2.selectbox("Sentiment", ["All", "positive", "negative", "neutral"])
    search_term = col3.text_input("Search keyword", placeholder="e.g. margin, revenue, growth")

    filtered = filtered_mgmt.copy()
    if company_filter != "All":
        filtered = filtered[filtered['ticker'] == company_filter]
    if sentiment_filter != "All":
        filtered = filtered[filtered['sentiment'] == sentiment_filter]
    if search_term:
        filtered = filtered[filtered['utterance'].str.contains(
            search_term, case=False, na=False)]

    st.write(f"**{len(filtered)} statements found**")

    for _, row in filtered.head(20).iterrows():
        color = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
            row['sentiment'], "⚪")
        with st.expander(f"{color} {row['ticker']} | {row['quarter']} | {row['speaker'][:30]}"):
            st.write(row['utterance'])
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{row['positive_score']:.2f}")
            col2.metric("Negative", f"{row['negative_score']:.2f}")
            col3.metric("Confidence", f"{row['confidence']:.2f}")

# ── PAGE 5: Summary Report ──────────────────────
elif page == "📋 Summary Report":
    st.title("📋 Pipeline Summary Report")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Dataset")
        st.write(f"**Companies:** {', '.join(companies)}")
        st.write("**Period:** Q1–Q4 FY25")
        st.write("**Transcripts:** 32 PDFs")
        st.write(f"**Total Utterances:** {len(mgmt_df)}")

        st.subheader("🔧 Pipeline Stages")
        stages = [
            "✅ Stage 1 — Data Acquisition (32 PDFs)",
            "✅ Stage 2 — Text Extraction (pdfplumber)",
            "✅ Stage 3 — Pre-processing (Speaker Diarization)",
            "✅ Stage 4 — Sentiment Analysis (FinBERT + MPS)",
            "✅ Stage 5 — Topic Modelling (BERTopic)",
            "✅ Stage 6 — Evaluation (F1=0.80)"
        ]
        for s in stages:
            st.write(s)

    with col2:
        st.subheader("📊 Model Performance")
        metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'Weighted F1', 'Neutral F1', 'Positive F1'],
            'Score': [0.85, 0.80, 0.91, 0.80]
        })
        fig = px.bar(metrics, x='Metric', y='Score',
                     color='Score', color_continuous_scale='Greens',
                     range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔮 Key Findings")
        st.info("**SUNPHARMA** dominates Cost & Sales discussions (169 sentences)")
        st.info("**WIPRO** most active on Business Opportunities (27)")
        st.success("**TCS & Tata** highest positive sentiment in IT sector")
        st.warning("**BAJAJFIN** highest negative signals — credit risk narrative")