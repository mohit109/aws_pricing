# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration with custom theme
st.set_page_config(
    page_title="Bedrock Token Cost Calculator", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #FF6B35;
        margin: 0.5rem 0;
    }
    
    .cost-breakdown {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .total-cost {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d1f2eb;
        border: 1px solid #a7e6cc;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💰 AWS Bedrock Token Cost Calculator</h1>
    <p>Anthropic Claude Models • US East (N. Virginia) Region</p>
</div>
""", unsafe_allow_html=True)

# Pricing data (same as before)
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

# Sidebar with improved styling
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Model selection with better formatting
    st.markdown("### 🤖 Model Selection")
    model = st.selectbox(
        "Choose Claude Model", 
        df.index.tolist(), 
        index=2,  # Default to Claude Sonnet 4
        help="Select the Claude model you want to calculate costs for"
    )
    
    # Show model info
    row = df.loc[model]
    st.info(f"📊 **{model}**\n\nInput: ${row['in_per_1k']:.6f}/1K tokens\nOutput: ${row['out_per_1k']:.6f}/1K tokens")
    
    st.markdown("---")
    
    # Pricing options
    st.markdown("### 💼 Pricing Options")
    use_batch = st.toggle(
        "🔄 Use Batch Pricing", 
        value=False,
        help="Enable batch pricing if available for the selected model"
    )
    
    if use_batch and pd.isna(row["batch_in_per_1k"]):
        st.warning("⚠️ Batch pricing not available for this model. Using standard pricing.")
    
    st.markdown("---")
    
    # Token inputs with better UX
    st.markdown("### 🎯 Token Usage")
    
    col1, col2 = st.columns(2)
    with col1:
        total_input_tokens = st.number_input(
            "📥 Input Tokens", 
            min_value=0, 
            value=10000, 
            step=1000,
            help="Total number of input tokens"
        )
    
    with col2:
        total_output_tokens = st.number_input(
            "📤 Output Tokens", 
            min_value=0, 
            value=2000, 
            step=500,
            help="Total number of output tokens"
        )
    
    # Quick presets
    st.markdown("**Quick Presets:**")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    with preset_col1:
        if st.button("📝 Small", help="1K in, 500 out"):
            st.session_state.preset_input = 1000
            st.session_state.preset_output = 500
    
    with preset_col2:
        if st.button("📊 Medium", help="10K in, 2K out"):
            st.session_state.preset_input = 10000
            st.session_state.preset_output = 2000
    
    with preset_col3:
        if st.button("🚀 Large", help="50K in, 10K out"):
            st.session_state.preset_input = 50000
            st.session_state.preset_output = 10000
    
    # Apply presets if clicked
    if 'preset_input' in st.session_state:
        total_input_tokens = st.session_state.preset_input
        total_output_tokens = st.session_state.preset_output
        del st.session_state.preset_input
        del st.session_state.preset_output
        st.rerun()
    
    st.markdown("---")
    
    # Cache settings
    st.markdown("### 🧠 Cache Settings")
    use_cache = st.toggle("Enable Cache Calculation", value=False)
    
    if use_cache:
        st.caption("Cache tokens are billed at special rates and should be a subset of input tokens")
        
        cache_write_tokens = st.number_input(
            "✍️ Cache Write Tokens", 
            min_value=0, 
            value=0, 
            step=100,
            help="Tokens written to cache (subset of input)"
        )
        
        cache_read_tokens = st.number_input(
            "📖 Cache Read Tokens", 
            min_value=0, 
            value=0, 
            step=100,
            help="Tokens read from cache (subset of input)"
        )
        
        normalize = st.checkbox("Auto-adjust cache tokens to not exceed input", value=True)
    else:
        cache_write_tokens = 0
        cache_read_tokens = 0
        normalize = True

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Calculate costs
    if normalize and use_cache:
        cw = min(cache_write_tokens, total_input_tokens)
        cr = min(cache_read_tokens, max(0, total_input_tokens - cw))
    else:
        cw, cr = cache_write_tokens, cache_read_tokens

    normal_input_tokens = max(0, total_input_tokens - cw - cr)

    # Get pricing
    in_price = (row["batch_in_per_1k"] if use_batch and pd.notna(row["batch_in_per_1k"]) else row["in_per_1k"])
    out_price = (row["batch_out_per_1k"] if use_batch and pd.notna(row["batch_out_per_1k"]) else row["out_per_1k"])
    
    cache_write_price = row["cache_write_per_1k"] if pd.notna(row["cache_write_per_1k"]) else None
    cache_read_price = row["cache_read_per_1k"] if pd.notna(row["cache_read_per_1k"]) else None

    # Compute costs
    cost_normal_input = (normal_input_tokens / 1000.0) * in_price
    cost_output = (total_output_tokens / 1000.0) * out_price

    if cache_write_price is not None and cw > 0:
        cost_cache_write = (cw / 1000.0) * cache_write_price
    else:
        cost_cache_write = 0.0

    if cache_read_price is not None and cr > 0:
        cost_cache_read = (cr / 1000.0) * cache_read_price
    else:
        cost_cache_read = 0.0

    total_cost = cost_normal_input + cost_output + cost_cache_write + cost_cache_read

    # Display results
    st.markdown("## 📊 Cost Analysis")
    
    # Key metrics in cards
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📥 Input Cost</h4>
            <h2>${cost_normal_input:.6f}</h2>
            <p>{normal_input_tokens:,} tokens</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📤 Output Cost</h4>
            <h2>${cost_output:.6f}</h2>
            <p>{total_output_tokens:,} tokens</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧠 Cache Cost</h4>
            <h2>${(cost_cache_write + cost_cache_read):.6f}</h2>
            <p>{(cw + cr):,} tokens</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⏱️ Per Request</h4>
            <h2>${total_cost:.6f}</h2>
            <p>Total cost</p>
        </div>
        """, unsafe_allow_html=True)

    # Total cost highlight
    st.markdown(f"""
    <div class="total-cost">
        <h1>💰 Total Estimated Cost</h1>
        <h1>${total_cost:.6f}</h1>
        <p>For {(total_input_tokens + total_output_tokens):,} total tokens</p>
    </div>
    """, unsafe_allow_html=True)

    # Cost breakdown table
    st.markdown("### 📋 Detailed Breakdown")
    
    breakdown_data = []
    if normal_input_tokens > 0:
        breakdown_data.append({
            "Component": "📥 Input Tokens (Normal)",
            "Tokens": f"{normal_input_tokens:,}",
            "Rate per 1K": f"${in_price:.6f}",
            "Cost": f"${cost_normal_input:.6f}"
        })
    
    if total_output_tokens > 0:
        breakdown_data.append({
            "Component": "📤 Output Tokens",
            "Tokens": f"{total_output_tokens:,}",
            "Rate per 1K": f"${out_price:.6f}",
            "Cost": f"${cost_output:.6f}"
        })
    
    if cw > 0:
        breakdown_data.append({
            "Component": "✍️ Cache Write",
            "Tokens": f"{cw:,}",
            "Rate per 1K": f"${cache_write_price:.6f}" if cache_write_price else "N/A",
            "Cost": f"${cost_cache_write:.6f}"
        })
    
    if cr > 0:
        breakdown_data.append({
            "Component": "📖 Cache Read",
            "Tokens": f"{cr:,}",
            "Rate per 1K": f"${cache_read_price:.6f}" if cache_read_price else "N/A",
            "Cost": f"${cost_cache_read:.6f}"
        })

    if breakdown_data:
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

with col2:
    # Cost visualization
    st.markdown("### 📈 Cost Visualization")
    
    # Pie chart of cost components
    if total_cost > 0:
        cost_data = []
        labels = []
        colors = []
        
        if cost_normal_input > 0:
            cost_data.append(cost_normal_input)
            labels.append("Input")
            colors.append("#FF6B35")
        
        if cost_output > 0:
            cost_data.append(cost_output)
            labels.append("Output")
            colors.append("#F7931E")
        
        if cost_cache_write > 0:
            cost_data.append(cost_cache_write)
            labels.append("Cache Write")
            colors.append("#4ECDC4")
        
        if cost_cache_read > 0:
            cost_data.append(cost_cache_read)
            labels.append("Cache Read")
            colors.append("#45B7D1")
        
        if cost_data:
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=cost_data,
                hole=0.4,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title="Cost Distribution",
                height=400,
                showlegend=True,
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Usage scenarios
    st.markdown("### 💡 Usage Scenarios")
    
    scenarios = [
        ("🤖 Chatbot (1K in, 200 out)", 1000, 200),
        ("📝 Content Gen (5K in, 2K out)", 5000, 2000),
        ("🔍 Analysis (10K in, 1K out)", 10000, 1000),
        ("📚 Summarization (20K in, 500 out)", 20000, 500),
    ]
    
    scenario_costs = []
    for name, inp, outp in scenarios:
        scenario_cost = (inp / 1000 * in_price) + (outp / 1000 * out_price)
        scenario_costs.append({
            "Scenario": name,
            "Cost": f"${scenario_cost:.4f}"
        })
    
    scenario_df = pd.DataFrame(scenario_costs)
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

# Additional features
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    # Monthly cost calculator
    st.markdown("### 📅 Monthly Projection")
    requests_per_day = st.number_input("Requests per day", min_value=0, value=100, step=10)
    monthly_cost = total_cost * requests_per_day * 30
    st.metric("Monthly Cost", f"${monthly_cost:.2f}")

with col2:
    # Export options
    st.markdown("### 📥 Export Options")
    if breakdown_data:
        breakdown_df = pd.DataFrame(breakdown_data)
        csv = breakdown_df.to_csv(index=False)
        st.download_button(
            "📄 Download CSV",
            data=csv,
            file_name="bedrock_cost_breakdown.csv",
            mime="text/csv",
            type="primary"
        )

with col3:
    # Model comparison
    st.markdown("### ⚖️ Quick Compare")
    compare_model = st.selectbox("Compare with", [m for m in df.index if m != model])
    if compare_model:
        compare_row = df.loc[compare_model]
        compare_cost = (total_input_tokens / 1000 * compare_row['in_per_1k']) + (total_output_tokens / 1000 * compare_row['out_per_1k'])
        savings = compare_cost - total_cost
        if savings > 0:
            st.success(f"💰 Save ${savings:.6f}")
        else:
            st.error(f"📈 Costs ${abs(savings):.6f} more")

# Footer with additional info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>💡 <strong>Pro Tip:</strong> Cache tokens can significantly reduce costs for repeated content</p>
    <p>📊 Prices are for US East (N. Virginia) region • Last updated: January 2025</p>
    <p>⚠️ This calculator provides estimates. Always verify with official AWS pricing</p>
</div>
""", unsafe_allow_html=True)
