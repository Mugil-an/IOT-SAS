import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import os
from datetime import datetime, timedelta

# 1. Page Config must be the FIRST streamlit command
st.set_page_config(page_title="Classroom Monitor", layout="wide")

# 2. Refresh logic - ONLY CALL THIS ONCE
st_autorefresh(interval=5000, key="datarefresh_timer")

CSV_PATH = "attention_results.csv"
IMG_PATH = "latest_frame.jpg"

st.title("Classroom Attention Dashboard")

# 3. Check if file exists
if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size < 50:
    st.info("Waiting for data from watcher.py... (Ensure watcher is running)")
    if os.path.exists(IMG_PATH):
        st.image(IMG_PATH, caption="Live Preview", use_container_width=True)
    st.stop()

# 4. Load Data safely
try:
    # We use on_bad_lines to ignore half-written rows from the watcher
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Clean column names just in case of spaces
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# 5. Filter for recent data (15 mins)
df_recent = df[df["timestamp"] > (datetime.now() - timedelta(minutes=15))]

# 6. Top Row Metrics
col1, col2, col3, col4 = st.columns(4)
if not df_recent.empty:
    with col1:
        st.metric("Active Students", df_recent["student_id"].nunique())
    with col2:
        avg = df_recent["attention_score"].mean()
        st.metric("Avg Attention", f"{avg:.1f}%")
    with col3:
        # Check if hand_raised column exists before summing
        hands = df_recent["hand_raised"].sum() if "hand_raised" in df_recent.columns else 0
        st.metric("Hands Raised", int(hands))
    with col4:
        last_t = df["timestamp"].iloc[-1].strftime("%H:%M:%S")
        st.metric("Last Update", last_t)
else:
    st.warning("No data in the last 15 minutes.")

st.divider()

# 7. Visuals Row
left, right = st.columns([1, 1])

with left:
    st.subheader("Live Processing View")
    if os.path.exists(IMG_PATH):
        st.image(IMG_PATH, use_container_width=True)

with right:
    st.subheader("Current Engagement Status")
    if not df_recent.empty:
        # Get latest score for each unique student ID
        latest = df_recent.sort_values("timestamp").groupby("student_id").last().reset_index()
        fig = px.bar(latest, x="student_id", y="attention_score", 
                     color="attention_score", range_y=[0, 110],
                     color_continuous_scale="RdYlGn",
                     text="attention_score")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# 8. Timeline View
st.divider()
st.subheader("Attention Timeline")
if not df_recent.empty:
    fig2 = px.line(df_recent, x="timestamp", y="attention_score", 
                   color="student_id", markers=True)
    fig2.update_layout(yaxis_range=[0, 110])
    st.plotly_chart(fig2, use_container_width=True)

# 9. Raw Data Table
with st.expander("View Raw Log"):
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)