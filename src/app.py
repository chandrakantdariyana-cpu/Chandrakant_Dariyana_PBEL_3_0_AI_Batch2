import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from logger import logger
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


def main():
    st.set_page_config(
        page_title="CyberGuard AI | Intrusion Detection System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    logger.info("Streamlit Dashboard Started.")
    logger.info("Page configuration completed.")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    MODEL_PATH = 'models/best_model.pkl'
    SCALER_PATH = 'models/scaler.pkl'
    ENCODER_PATH = 'models/label_encoder.pkl'

    logger.info(f"Base Directory : {BASE_DIR}")
    logger.info(f"Model Directory : {MODEL_DIR}")
    logger.info(f"Model Path : {MODEL_PATH}")
    logger.info(f"Scaler Path : {SCALER_PATH}")
    logger.info(f"Encoder Path : {ENCODER_PATH}")

    with open('style.txt', 'r') as file:
        style = file.read()

    logger.info("Loading dashboard style file.")
    logger.info("Dashboard style loaded successfully.")
    logger.info("Loading dashboard style file.")

    with open('style.txt', 'r') as file:
        style = file.read()

    logger.info("Dashboard style loaded successfully.")

    st.markdown(style, unsafe_allow_html=True)

    logger.info("Custom CSS applied successfully.")

    @st.cache_resource
    def load_artifacts():
        model, scaler, encoder, feature_columns = None, None, None, None

        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)

        if os.path.exists(ENCODER_PATH):
            encoder = joblib.load(ENCODER_PATH)

        if os.path.exists(FEATURE_COLUMNS_PATH):
            feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

        return model, scaler, encoder, feature_columns


    model, scaler, label_encoder, feature_columns = load_artifacts()

    if "activity_feed" not in st.session_state:
        st.session_state.activity_feed = []
    if "predictions_df" not in st.session_state:
        st.session_state.predictions_df = None

    with st.sidebar:
        st.markdown("### 🛡️ CyberGuard AI")
        st.caption("Intrusion Detection System")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Overview", "Threat Detection", "Real-time Monitor", "Security Analytics"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**System Status**")
        if model is not None:
            st.success("Model loaded ✔️")
        else:
            st.error("best_model.pkl not found in /models")
        st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.markdown("## Security Dashboard")
        st.caption("Real-time monitoring and threat analysis powered by AI")
    with col_h2:
        st.markdown(
            "<div style='text-align:right; padding-top:20px;'>"
            "🟢 <span style='color:#4ade80; font-weight:600;'>System Secure</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV (CICIDS2017-format) for threat analysis",
        type=["csv"],
    )

    if uploaded_file is not None and model is not None:
        with st.spinner("Analyzing traffic with AI model..."):
            raw_df = pd.read_csv(uploaded_file)
            df = raw_df.copy()

            for col in ["Label", "label", "Attack Type"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            df.columns = df.columns.str.strip()

            df = df.select_dtypes(include=[np.number])

            if feature_columns is not None:
                df = df.reindex(columns=feature_columns, fill_value=0)

            df = df.fillna(0)

            try:
                X = scaler.transform(df) if scaler is not None else df.values
                preds = model.predict(X)

                if label_encoder is not None:
                    pred_labels = label_encoder.inverse_transform(preds)
                else:
                    pred_labels = preds

                raw_df["Prediction"] = pred_labels
                st.session_state.predictions_df = raw_df

                for label in pred_labels[:15]:
                    is_attack = str(label).upper() not in ("BENIGN", "NORMAL", "0")
                    st.session_state.activity_feed.insert(0, {
                        "label": label,
                        "critical": is_attack,
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })
                st.session_state.activity_feed = st.session_state.activity_feed[:20]

                st.success(f"Analysis complete — {len(raw_df)} flows scanned.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Check that the CSV columns match the features used during training.")

    pred_df = st.session_state.predictions_df

    if pred_df is not None and "Prediction" in pred_df.columns:
        total = len(pred_df)
        attack_mask = ~pred_df["Prediction"].astype(str).str.upper().isin(["BENIGN", "NORMAL", "0"])
        active_threats = int(attack_mask.sum())
        blocked_attacks = active_threats
        unread_alerts = int(min(active_threats, 9))
        system_health = round(100 - (active_threats / total * 100 if total else 0), 1)
    else:
        active_threats, blocked_attacks, unread_alerts, system_health = 0, 0, 0, 100.0

    c1, c2, c3, c4 = st.columns(4)

    def metric_card(col, label, value, sub, badge_text, badge_class, icon):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{icon} &nbsp;{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
                <div class="badge {badge_class}">{badge_text}</div>
            </div>
            """, unsafe_allow_html=True)

    metric_card(c1, "Active Threats", active_threats, "Detected in this scan",
                "MEDIUM" if active_threats else "LOW",
                "badge-medium" if active_threats else "badge-low", "⚠️")

    metric_card(c2, "Blocked Attacks", blocked_attacks, "Flagged by AI model",
                "CRITICAL" if blocked_attacks else "LOW",
                "badge-critical" if blocked_attacks else "badge-low", "🛡️")

    metric_card(c3, "Unread Alerts", unread_alerts, "Require attention",
                "MEDIUM" if unread_alerts else "LOW",
                "badge-medium" if unread_alerts else "badge-low", "👁️")

    metric_card(c4, "System Health", f"{system_health}%", "All systems operational",
                "LOW" if system_health > 90 else "MEDIUM",
                "badge-low" if system_health > 90 else "badge-medium", "📈")

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="section-title">Threat Detection Timeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Attacks vs total traffic over time</div>', unsafe_allow_html=True)

        if pred_df is not None:
            timeline = pred_df.copy()
            timeline["idx"] = range(len(timeline))
            timeline["is_attack"] = attack_mask.values.astype(int)
            bucket_size = max(1, len(timeline) // 30)
            timeline["bucket"] = timeline["idx"] // bucket_size
            agg = timeline.groupby("bucket").agg(
                total=("is_attack", "count"),
                attacks=("is_attack", "sum"),
            ).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=agg["bucket"], y=agg["total"], name="Total Traffic",
                                     line=dict(color="#f87171"), fill="tozeroy",
                                     fillcolor="rgba(248,113,113,0.15)"))
            fig.add_trace(go.Scatter(x=agg["bucket"], y=agg["attacks"], name="Attacks",
                                     line=dict(color="#4ade80"), fill="tozeroy",
                                     fillcolor="rgba(74,222,128,0.15)"))
            fig.update_layout(
                paper_bgcolor="#0b1120", plot_bgcolor="#0b1120",
                font_color="#94a3b8", margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.15),
                xaxis=dict(showgrid=False, title="Batch"), yaxis=dict(gridcolor="#1e293b"),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload a CSV above to generate the live threat timeline.")

    with chart_col2:
        st.markdown('<div class="section-title">Attack Type Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Breakdown of detected classes</div>', unsafe_allow_html=True)

        if pred_df is not None:
            dist = pred_df["Prediction"].value_counts().reset_index()
            dist.columns = ["Type", "Count"]
            fig2 = px.bar(dist, x="Count", y="Type", orientation="h",
                          color="Count", color_continuous_scale=["#4ade80", "#f87171"])
            fig2.update_layout(
                paper_bgcolor="#0b1120", plot_bgcolor="#0b1120",
                font_color="#94a3b8", margin=dict(l=10, r=10, t=10, b=10),
                height=320, coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Upload a CSV above to see attack type breakdown.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔴 Real-time Activity Feed</div>', unsafe_allow_html=True)

    if st.session_state.activity_feed:
        for item in st.session_state.activity_feed:
            css_class = "feed-critical" if item["critical"] else "feed-normal"
            badge = "🔴 CRITICAL" if item["critical"] else "🟢 NORMAL"
            st.markdown(f"""
            <div class="feed-item {css_class}">
                <div>
                    <b>{item['label']}</b> traffic pattern detected
                    <div style="color:#64748b; font-size:12px;">AI Detection System • {item['time']}</div>
                </div>
                <div>{badge}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No activity yet. Upload traffic data to start monitoring.")

    if pred_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
        st.dataframe(pred_df, use_container_width=True, height=300)

        csv_out = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results as CSV", csv_out,
                           file_name="threat_predictions.csv", mime="text/csv")

if __name__ == '__main__':
    main()