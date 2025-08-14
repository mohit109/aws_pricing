# app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Bedrock Token Cost Calculator (Anthropic)", page_icon="🧮", layout="centered")

st.title("🧮 Bedrock Token Cost Calculator — Anthropic (us-east-1)")
st.caption(
    "Pricing copied from the screenshot you provided (US East / N. Virginia). "
    "If AWS updates prices or you use a different region, adjust the table below."
)

# ---------------------------
# Pricing table (USD per 1K tokens)
# Keys are model names as shown in the screenshot.
# batch_* uses the batch columns when available (N/A shown as None).
# cache_* applies to input tokens when writing to or reading from the cache.
# ---------------------------
PRICING = [
    # name, in, out, batch_in, batch_out, cache_write, cache_read
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

with st.sidebar:
    st.subheader("Inputs")
    model = st.selectbox("Model", df.index.tolist(), index=0)
    use_batch = st.toggle("Use batch pricing (where available)", value=False)
    st.caption("If a batch price is N/A for the selected model, the normal price will be used.")

    total_input_tokens = st.number_input("Total input tokens", min_value=0, value=1000, step=100)
    total_output_tokens = st.number_input("Total output tokens", min_value=0, value=1000, step=100)

    st.markdown("---")
    st.subheader("Cache (optional)")
    st.caption(
        "Specify how many input tokens are written to cache and read from cache.\n"
        "These tokens are billed at cache rates and should be a subset of your input tokens."
    )
    cache_write_tokens = st.number_input("Cache write tokens (subset of input)", min_value=0, value=0, step=100)
    cache_read_tokens  = st.number_input("Cache read tokens (subset of input)",  min_value=0, value=0, step=100)

    normalize = st.toggle("Automatically cap cache tokens so they don't exceed total input", value=True)

# Normalize / safety checks
if normalize:
    # ensure cache tokens don't exceed total input tokens
    cw = min(cache_write_tokens, total_input_tokens)
    cr = min(cache_read_tokens, max(0, total_input_tokens - cw))
else:
    cw, cr = cache_write_tokens, cache_read_tokens

# Any remaining input tokens that are NOT cache-read or cache-written
normal_input_tokens = max(0, total_input_tokens - cw - cr)

row = df.loc[model]
in_price  = (row["batch_in_per_1k"]  if use_batch and pd.notna(row["batch_in_per_1k"])  else row["in_per_1k"])
out_price = (row["batch_out_per_1k"] if use_batch and pd.notna(row["batch_out_per_1k"]) else row["out_per_1k"])

cache_write_price = row["cache_write_per_1k"] if pd.notna(row["cache_write_per_1k"]) else None
cache_read_price  = row["cache_read_per_1k"]  if pd.notna(row["cache_read_per_1k"])  else None

# Compute costs
cost_normal_input = (normal_input_tokens / 1000.0) * in_price
cost_output       = (total_output_tokens / 1000.0) * out_price

if cache_write_price is not None:
    cost_cache_write = (cw / 1000.0) * cache_write_price
else:
    cost_cache_write = 0.0 if cw == 0 else float("nan")

if cache_read_price is not None:
    cost_cache_read = (cr / 1000.0) * cache_read_price
else:
    cost_cache_read = 0.0 if cr == 0 else float("nan")

components = [
    ("Normal input", normal_input_tokens, in_price, cost_normal_input),
    ("Output",       total_output_tokens, out_price, cost_output),
    ("Cache write",  cw, cache_write_price, cost_cache_write),
    ("Cache read",   cr, cache_read_price,  cost_cache_read),
]

breakdown_rows = []
total_cost = 0.0
for label, tokens, price_per_1k, cost in components:
    if tokens == 0 and (pd.isna(price_per_1k) or price_per_1k is None):
        continue
    breakdown_rows.append({
        "Component": label,
        "Tokens": tokens,
        "Price / 1K": ("N/A" if (price_per_1k is None or pd.isna(price_per_1k)) else f"${price_per_1k:,.6f}"),
        "Cost (USD)": ("" if pd.isna(cost) else f"${cost:,.6f}")
    })
    if not pd.isna(cost):
        total_cost += cost

st.markdown("### Selection")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Model:** {model}")
    st.write(f"**Batch pricing:** {'Yes' if use_batch else 'No'}")
with col2:
    st.write(f"**Input tokens:** {total_input_tokens:,}")
    st.write(f"**Output tokens:** {total_output_tokens:,}")
st.write(f"**Cache write tokens:** {cw:,}  |  **Cache read tokens:** {cr:,}")

st.markdown("### Cost breakdown")
breakdown_df = pd.DataFrame(breakdown_rows)
st.dataframe(breakdown_df, use_container_width=True)

st.markdown(f"## Total estimated cost: **${total_cost:,.6f}**")

# Download breakdown
csv = breakdown_df.to_csv(index=False)
st.download_button(
    "Download breakdown (CSV)",
    data=csv,
    file_name="bedrock_cost_breakdown.csv",
    mime="text/csv"
)

# Show the raw price row for quick reference
st.markdown("#### Pricing row (per 1K tokens)")
display_row = row.copy()
display_row = display_row.rename({
    "in_per_1k": "input",
    "out_per_1k": "output",
    "batch_in_per_1k": "batch_input",
    "batch_out_per_1k": "batch_output",
    "cache_write_per_1k": "cache_write_input",
    "cache_read_per_1k": "cache_read_input",
})
st.table(display_row.to_frame("USD").style.format({"USD": "{:.6f}"}))

st.caption(
    "Notes: Cache prices apply only to input tokens. "
    "This calculator treats cache-write and cache-read tokens as subsets of your total input tokens."
)
