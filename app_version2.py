# app.py
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------- Page & Style --------------------
st.set_page_config(page_title="AWS Bedrock Token Cost Calculator", page_icon="🧮", layout="wide")

# Minimal inline CSS to get the card-like look
st.markdown(
    """
    <style>
    .gradient-banner {
        background: linear-gradient(90deg, #ff7a18, #ffb347);
        color: white; padding: 18px 22px; border-radius: 14px; font-weight: 700;
        margin-bottom: 18px; border: 1px solid rgba(255,255,255,.3);
    }
    .subcard {
        background: #fff; border-radius: 14px; border: 1px solid #ececec;
        padding: 18px; box-shadow: 0 8px 18px rgba(0,0,0,0.05);
    }
    .big-total {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white; border-radius: 16px; padding: 26px; text-align: center;
        box-shadow: 0 8px 22px rgba(0,0,0,0.12); border: 1px solid rgba(255,255,255,.2);
        margin: 12px 0 16px 0;
    }
    .pill {
        display:inline-block; padding:4px 10px; background:#f6f7fb; border:1px solid #ececec;
        border-radius:999px; font-size:12px; color:#666; margin-left:6px;
    }
    .muted { color:#666; font-size: 13px; }
    .metric-label { font-size: 14px; color:#6b7280; margin-bottom: 2px; }
    .metric-value { font-size: 26px; font-weight: 800; margin-bottom: 0; }
    .metric-sub { font-size: 12px; color:#9ca3af; }
    .section-title { font-weight: 800; font-size: 20px; margin: 4px 0 12px 0; }
    .hint { font-size: 12px; color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Pricing (USD per 1K tokens) --------------------
PRICING = [
    ("Claude Opus 4.1",         0.015,   0.075,   None,     None,     0.01875, 0.0015),
    ("Claude Opus 4",           0.015,   0.075,   None,     None,     0.01875, 0.0015),
    ("Claude Sonnet 4",         0.003,   0.015,   None,     None,     0.00375, 0.0003),
    ("Claude Sonnet 4 - Long Context", 0.006, 0.0225, None, None,     0.0075,  0.0006),
    ("Claude 3.7 Sonnet",       0.003,   0.015,   None,     None,     0.00375, 0.0003),
    ("Claude 3.5 Sonnet",       0.003,   0.015,   0.0015,   0.0075,   None,    None),
    ("Claude 3.5 Haiku",        0.0008,  0.004,   0.0004,   0.002,    0.001,   0.00008),
    ("Claude 3.5 Sonnet v2",    0.0
