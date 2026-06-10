import streamlit as st
import pickle
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="YouTube Ad Revenue Predictor", page_icon="🎬", layout="centered")

st.title("🎬 YouTube Ad Revenue Predictor")
st.markdown("Predict estimated ad revenue based on video performance metrics.")
st.divider()

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# ── Sidebar Inputs ───────────────────────────────────────────
st.sidebar.header("📊 Enter Video Details")

views        = st.sidebar.number_input("Views",                  key="views",        min_value=1,           max_value=100_000_000, value=500_000, step=1000)
likes        = st.sidebar.number_input("Likes",                  key="likes",        min_value=0,           max_value=10_000_000,  value=20_000,  step=100)
comments     = st.sidebar.number_input("Comments",               key="comments",     min_value=0,           max_value=10_000_000,  value=5_000,   step=100)
watch_time   = st.sidebar.number_input("Watch Time (minutes)",   key="watch_time",   min_value=0,           max_value=10_000_000,  value=100_000, step=500)
video_length = st.sidebar.number_input("Video Length (minutes)", key="video_length", min_value=1,           max_value=1_000,       value=10,      step=1)
subscribers  = st.sidebar.number_input("Subscribers",            key="subscribers",  min_value=0,           max_value=100_000_000, value=50_000,  step=1000)

category = st.sidebar.selectbox("Category", ['Entertainment', 'Music', 'Gaming', 'Education', 'Tech', 'Lifestyle'])
device   = st.sidebar.selectbox("Device",   ['Mobile', 'Desktop', 'Tablet', 'TV'])
country  = st.sidebar.selectbox("Country",  ['CA', 'DE', 'IN', 'UK', 'US', 'AU'])

# ── Auto-calculate derived features ──────────────────────────
engagement_rate = (likes + comments) / views
watch_ratio     = watch_time / video_length

# ── Show calculated values ────────────────────────────────────
st.subheader("⚙️ Auto-Calculated Features")
col1, col2 = st.columns(2)
col1.metric("Engagement Rate", f"{engagement_rate:.4f}", help="(Likes + Comments) / Views")
col2.metric("Watch Ratio",     f"{watch_ratio:.4f}",     help="Watch Time / Video Length")
st.divider()

# ── Build input DataFrame ─────────────────────────────────────
input_data = pd.DataFrame([{
    'views':              views,
    'likes':              likes,
    'watch_time_minutes': watch_time,
    'subscribers':        subscribers,
    'category':           category,
    'device':             device,
    'country':            country,
    'engagement_rate':    engagement_rate,
    'watch_ratio':        watch_ratio
}])

# ── Show input summary ────────────────────────────────────────
st.subheader("📋 Input Summary")
st.dataframe(input_data, use_container_width=True)
st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("🚀 Predict Ad Revenue", use_container_width=True):
    prediction = model.predict(input_data)[0]
    st.success("✅ Prediction Complete!")
    st.metric(label="💰 Estimated Monthly Ad Revenue", value=f"${prediction:,.2f}")
    st.divider()
    st.caption("⚠️ Prediction is based on your trained Random Forest model.")