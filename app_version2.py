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
    ("Claude 3.5 Sonnet v2",    0.003,   0.015,   0.0015,   0.0075,   0.00375, 0.0003),
    ("Claude 3 Opus",           0.015,   0.075,   0.0075,   0.0375,   None,    None),
    ("Claude 3 Haiku",          0.00025, 0.00125, 0.000125, 0.000625, None,    None),
    ("Claude 3 Sonnet",         0.003,   0.015,   0.0015,   0.0075,   None,    None),
    ("Claude 2.1",              0.008,   0.024,   None,     None,     None,    None),
    ("Claude 2.0",              0.008,   0.024,   None,     None,     None,    None),
    ("Claude Instant",          0.0008,  0.0024,  None,     None,     None,    None),
]
df = pd.DataFrame(PRICING, columns=[
    "Model", "in_per_1k", "out_per_1k", "batch_in_per_1k", "batch_out_per_1k", "cache_write_per_1k", "cache_read_per_1k"
]).set_index("Model")

# -------------------- Helpers --------------------
def pick_price(row, use_batch, key_normal, key_batch):
    if use_batch and not math.isnan(row[key_batch]) if row[key_batch] is not None else False:
        return row[key_batch]
    return row[key_normal]

def compute_costs(row, use_batch, in_tokens, out_tokens, use_cache=False, cache_w=0, cache_r=0):
    in_price  = pick_price(row, use_batch, "in_per_1k",  "batch_in_per_1k")
    out_price = pick_price(row, use_batch, "out_per_1k", "batch_out_per_1k")
    cache_w_price = row["cache_write_per_1k"] if pd.notna(row["cache_write_per_1k"]) else None
    cache_r_price = row["cache_read_per_1k"]  if pd.notna(row["cache_read_per_1k"])  else None

    cw = max(0, min(cache_w, in_tokens)) if use_cache else 0
    cr = max(0, min(cache_r, in_tokens - cw)) if use_cache else 0
    normal_in = max(0, in_tokens - cw - cr)

    cost_in   = (normal_in / 1000.0) * in_price
    cost_out  = (out_tokens / 1000.0) * out_price
    cost_cw   = (cw / 1000.0) * (cache_w_price or 0)
    cost_cr   = (cr / 1000.0) * (cache_r_price or 0)
    cache_cost = (0 if not use_cache else (cost_cw + cost_cr))

    breakdown = [
        ("Input tokens (billed)", normal_in, in_price, cost_in),
        ("Output tokens", out_tokens, out_price, cost_out),
    ]
    if use_cache:
        breakdown.append(("Cache write (subset of input)", cw, cache_w_price, cost_cw if cache_w_price else float("nan")))
        breakdown.append(("Cache read (subset of input)",  cr, cache_r_price, cost_cr if cache_r_price else float("nan")))

    total = cost_in + cost_out + (cache_cost if use_cache else 0)
    return total, cost_in, cost_out, cache_cost, breakdown, dict(
        in_price=in_price, out_price=out_price, cw_price=cache_w_price, cr_price=cache_r_price,
        normal_in=normal_in, cw=cw, cr=cr
    )

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Model Selection")
    selected_model = st.selectbox("Choose Claude Model", list(df.index), index=list(df.index).index("Claude Sonnet 4"))

    row = df.loc[selected_model]

    st.markdown(
        f"""
        <div class="subcard" style="padding:10px">
        <b>{selected_model}</b><br>
        <span class="muted">
        Input: ${row['in_per_1k']:.6f}/1K • Output: ${row['out_per_1k']:.6f}/1K
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Pricing Options")
    use_batch = st.toggle("Use Batch Pricing", value=False)
    st.caption("If batch price isn't available, normal price is used.")

    st.subheader("Token Usage")
    col_in, col_out = st.columns(2)
    with col_in:
        input_tokens = st.number_input("Input Tokens", min_value=0, value=10000, step=100)
    with col_out:
        output_tokens = st.number_input("Output Tokens", min_value=0, value=2000, step=100)

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        if st.button("Sm", use_container_width=True):
            input_tokens, output_tokens = 2000, 500
    with pcol2:
        if st.button("Medium", use_container_width=True):
            input_tokens, output_tokens = 10000, 2000
    with pcol3:
        if st.button("Large", use_container_width=True):
            input_tokens, output_tokens = 50000, 10000

    st.subheader("Cache Settings")
    enable_cache = st.toggle("Enable Cache Calculation", value=False)
    if enable_cache:
        cache_w = st.number_input("Cache write tokens (subset of input)", min_value=0, value=0, step=100)
        cache_r = st.number_input("Cache read tokens (subset of input)", min_value=0, value=0, step=100)
    else:
        cache_w = cache_r = 0

    st.subheader("Monthly Projection")
    requests_per_day = st.number_input("Requests per day", min_value=0, value=100, step=10)

# -------------------- Header --------------------
st.markdown(
    """
    <div class="gradient-banner">
        <div style="font-size:20px">💰 AWS Bedrock Token Cost Calculator</div>
        <div class="muted">Anthropic Claude Models • US East (N. Virginia) Region<span class="pill">pricing per 1K tokens</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Calculations --------------------
total_cost, cost_in, cost_out, cache_cost, breakdown_list, ctx = compute_costs(
    row, use_batch, input_tokens, output_tokens, enable_cache, cache_w, cache_r
)

# -------------------- Top Metrics & Chart --------------------
lcol, rcol = st.columns([2.2, 1.2])

with lcol:
    st.markdown('<div class="section-title">📊 Cost Analysis</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="subcard">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">🧾 Input Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${cost_in:,.6f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-sub">{ctx["normal_in"]:,} tokens</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="subcard">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">🧠 Output Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${cost_out:,.6f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-sub">{output_tokens:,} tokens</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="subcard">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">🗄️ Cache Cost</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${(cache_cost if enable_cache else 0):,.6f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-sub">{(ctx["cw"]+ctx["cr"]) if enable_cache else 0:,} tokens</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="subcard">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">⏱️ Per Request</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${total_cost:,.6f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-sub">Total cost</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="big-total">', unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:18px;font-weight:700'>💡 Total Estimated Cost</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:28px;font-weight:900'>${total_cost:,.6f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hint'>For {(ctx['normal_in'] + (ctx['cw'] if enable_cache else 0) + (ctx['cr'] if enable_cache else 0) + output_tokens):,} total tokens</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with rcol:
    st.markdown('<div class="section-title">📈 Cost Visualization</div>', unsafe_allow_html=True)
    with st.container():
        fig, ax = plt.subplots()
        values = [cost_in, cost_out]
        labels = ["Input", "Output"]
        if enable_cache and cache_cost > 0:
            values.append(cache_cost)
            labels.append("Cache")
        if sum(values) == 0:
            values = [1]  # avoid zero pie
            labels = ["No cost"]
        ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig, use_container_width=True)

# -------------------- Breakdown Table --------------------
st.markdown('<div class="section-title">📄 Detailed Breakdown</div>', unsafe_allow_html=True)
breakdown_df = pd.DataFrame(
    [{"Component": n, "Tokens": t, "Rate per 1K": ("" if r is None or pd.isna(r) else f"${r:,.6f}"), "Cost (USD)": ("" if pd.isna(c) else f"${c:,.6f}")} for n, t, r, c in breakdown_list]
)
st.dataframe(breakdown_df, use_container_width=True)

# -------------------- Usage Scenarios --------------------
scenarios = [
    ("eChatbot (2k in, 200 out)", 2000, 200),
    ("Content Gen (5k in, 2k out)", 5000, 2000),
    ("Analysis (10k in, 1k out)", 10000, 1000),
    ("Summarization (20k in, 500 out)", 20000, 500),
]
rows = []
for name, IN, OUT in scenarios:
    sc_total, *_ = compute_costs(row, use_batch, IN, OUT, enable_cache=False)
    rows.append({"Scenario": name, "Cost": f"${sc_total:,.6f}"})
sc_df = pd.DataFrame(rows)

uc1, uc2 = st.columns([2, 1])
with uc1:
    st.markdown('<div class="section-title">💡 Usage Scenarios</div>', unsafe_allow_html=True)
    st.dataframe(sc_df, use_container_width=True)
with uc2:
    st.markdown('<div class="section-title">📤 Export Options</div>', unsafe_allow_html=True)
    csv = breakdown_df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", data=csv, file_name="bedrock_cost_breakdown.csv", mime="text/csv", use_container_width=True)

# -------------------- Monthly Projection --------------------
st.markdown('<div class="section-title">📅 Monthly Projection</div>', unsafe_allow_html=True)
monthly_cost = total_cost * requests_per_day * 30
st.write(f"**Requests per day:** {requests_per_day:,}")
st.write(f"### Monthly Cost: **${monthly_cost:,.2f}**")

# -------------------- Quick Compare --------------------
st.markdown('<div class="section-title">⚖️ Quick Compare</div>', unsafe_allow_html=True)
alt_model = st.selectbox("Compare with", list(df.index), index=list(df.index).index("Claude Opus 4.1"))
alt_row = df.loc[alt_model]
alt_total, *_ = compute_costs(alt_row, use_batch, input_tokens, output_tokens, enable_cache, cache_w, cache_r)
delta = alt_total - total_cost
if delta >= 0:
    st.success(f"✅ You save **${delta:,.6f}** vs {alt_model}")
else:
    st.warning(f"⚠️ {alt_model} is **${-delta:,.6f}** cheaper for this request")

# -------------------- Footnotes --------------------
st.markdown(
    """
    <hr>
    <div class="hint">
    Pro Tip: Cache tokens can reduce costs for repeated content. |
    Prices are for US East (N. Virginia). |
    This calculator provides estimates—always verify with official AWS pricing.
    </div>
    """,
    unsafe_allow_html=True,
)
