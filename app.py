"""
app.py  —  Streamlit web application for fruit recognition and price lookup.

Usage:
    streamlit run app.py

Requirements:
    - At least one trained model in saved_models/
    - prices.json in the project root
    - saved_models/class_names.json (auto-created by train.py)
"""

import os
import json

import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image

from dataset import prepare_image
from models import MODEL_REGISTRY

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="🍎 Fruit Recognizer",
    page_icon="🍎",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); }

    .result-box {
        background: linear-gradient(135deg, #1e3a5f, #12284a);
        border: 1px solid #3a7bd5;
        border-radius: 16px;
        padding: 24px 32px;
        margin: 16px 0;
        text-align: center;
    }
    .result-label {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .result-confidence {
        font-size: 1rem;
        color: #8ab4f8;
        margin-top: 4px;
    }
    .price-box {
        background: linear-gradient(135deg, #1a4a2e, #0d2b1a);
        border: 1px solid #34d399;
        border-radius: 16px;
        padding: 20px 32px;
        margin: 16px 0;
        text-align: center;
    }
    .price-label {
        font-size: 1rem;
        color: #6ee7b7;
        margin: 0;
    }
    .price-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: #34d399;
        margin: 4px 0 0 0;
    }
    .top-preds {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 16px 24px;
        margin: 12px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3a7bd5, #00d2ff);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────
@st.cache_resource
def load_class_names():
    path = "saved_models/class_names.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_prices():
    with open("prices.json") as f:
        return json.load(f)


@st.cache_resource
def load_model_cached(model_name: str, num_classes: int):
    path = f"saved_models/{model_name}_best.pth"
    if not os.path.exists(path):
        return None
    device = torch.device("cpu")   # CPU for serving (safe on all machines)
    model = MODEL_REGISTRY[model_name](num_classes=num_classes, pretrained=False)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def get_available_models():
    names = []
    for m in MODEL_REGISTRY.keys():
        if os.path.exists(f"saved_models/{m}_best.pth"):
            names.append(m.upper())
    return names


# ── UI ─────────────────────────────────────────────────────────
st.title("🍎 Fruit Recognizer")
st.markdown("Upload a photo of a fruit — the model will identify it and show its price.")
st.markdown("---")

# Sidebar — model selector
st.sidebar.title("⚙️ Settings")
available = get_available_models()

if not available:
    st.error(
        "❌ No trained models found in `saved_models/`.\n\n"
        "Run training first:\n```\npython train.py --model resnet --epochs 10\n```"
    )
    st.stop()

selected_display = st.sidebar.selectbox("Choose model", available)
selected_key = selected_display.lower()

class_names = load_class_names()
if class_names is None:
    st.error("❌ `saved_models/class_names.json` not found. Re-run `train.py`.")
    st.stop()

prices = load_prices()
model = load_model_cached(selected_key, num_classes=len(class_names))

st.sidebar.markdown(f"**Classes:** {len(class_names)}")
st.sidebar.markdown(f"**Model loaded:** ✅ {selected_display}")

# File uploader
uploaded = st.file_uploader(
    "Choose an image (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            tensor = prepare_image(image)
            with torch.no_grad():
                outputs = model(tensor)
                probs = F.softmax(outputs, dim=1)[0]

            top5_probs, top5_idx = torch.topk(probs, k=min(5, len(class_names)))
            top5_probs = top5_probs.numpy()
            top5_idx   = top5_idx.numpy()

        # Main prediction
        pred_class = class_names[top5_idx[0]]
        pred_conf  = top5_probs[0]
        price      = prices.get(pred_class, None)

        st.markdown(f"""
        <div class="result-box">
            <p class="result-label">🍓 {pred_class}</p>
            <p class="result-confidence">Confidence: {pred_conf:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

        if price is not None:
            st.markdown(f"""
            <div class="price-box">
                <p class="price-label">Price per unit</p>
                <p class="price-value">₸ {price}</p>
            </div>
            """, unsafe_allow_html=True)

            # --- Shopping Cart Feature ---
            if "cart" not in st.session_state:
                st.session_state.cart = []
            
            if st.button("➕ Add to Bill"):
                st.session_state.cart.append({"item": pred_class, "price": price})
                st.toast(f"Added {pred_class} to bill!")
        else:
            st.warning(f"Price for '{pred_class}' not found in prices.json")

    # Display the current bill
    if "cart" in st.session_state and st.session_state.cart:
        st.markdown("---")
        st.markdown("### 🛒 Your Bill")
        total_cost = 0
        for i, entry in enumerate(st.session_state.cart):
            col_a, col_b = st.columns([3, 1])
            col_a.write(f"{i+1}. **{entry['item']}**")
            col_b.write(f"₸ {entry['price']}")
            total_cost += entry['price']
        
        st.markdown(f"#### **Total Combined Price: ₸ {total_cost}**")
        if st.button("🗑️ Clear Bill"):
            st.session_state.cart = []
            st.rerun()

    # Top-5 predictions table
    st.markdown("#### 📊 Top-5 Predictions")
    for i, (idx, prob) in enumerate(zip(top5_idx, top5_probs)):
        name = class_names[idx]
        bar_color = "#3a7bd5" if i == 0 else "#555"
        st.markdown(f"""
        <div style="margin: 6px 0; display: flex; align-items: center; gap: 12px;">
            <span style="width: 130px; color: {'#fff' if i==0 else '#aaa'}; font-weight: {'700' if i==0 else '400'};">
                {name}
            </span>
            <div style="flex:1; background:#222; border-radius:8px; height:18px;">
                <div style="width:{prob*100:.1f}%; background:{bar_color};
                            border-radius:8px; height:100%;"></div>
            </div>
            <span style="width:52px; text-align:right; color:#ccc; font-size:0.9rem;">
                {prob:.1%}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"Model: **{selected_display}** | Classes: {len(class_names)}")

else:
    st.info("👆 Upload a fruit photo to get started!")
    st.markdown("**Supported fruits:**")
    prices_data = load_prices()
    cols = st.columns(3)
    for i, (fruit, price) in enumerate(prices_data.items()):
        cols[i % 3].markdown(f"🍏 **{fruit}** — ₸{price}")
