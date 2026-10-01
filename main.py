import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# 1. Page Configuration
# =========================================================

st.set_page_config(
    page_title="Next Word AI",
    page_icon="✨",
    layout="centered"
)


# =========================================================
# 2. Modern Premium & Centered CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
   GLOBAL
===================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99,102,241,0.20),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(168,85,247,0.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(14,165,233,0.12),
            transparent 35%
        ),
        #080b18;
    color: #f8fafc;
}

/* Main container */
.block-container {
    max-width: 750px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}


/* =====================================================
   HEADER
===================================================== */

.main-title {
    text-align: center;
    font-size: 3.4rem;
    font-weight: 900;
    background: linear-gradient(
        90deg,
        #60a5fa,
        #8b5cf6,
        #d946ef
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 8px;
    text-shadow: 0 0 35px rgba(139,92,246,0.25);
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2.5rem;
}

.subtitle b {
    color: #c4b5fd;
}


/* =====================================================
   INPUT LABEL (Custom Centered)
===================================================== */

.custom-input-label {
    text-align: center;
    color: #cbd5e1;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 12px;
}


/* =====================================================
   TEXT INPUT
===================================================== */

.stTextInput input {
    background: rgba(15,23,42,0.85) !important;
    color: #f8fafc !important;
    text-align: center !important; /* Centered text inside input */
    border: 1px solid rgba(148,163,184,0.20) !important;
    border-radius: 16px !important;
    padding: 17px !important;
    font-size: 17px !important;
    transition: all 0.3s ease;
    box-shadow:
        inset 0 0 20px rgba(0,0,0,0.15),
        0 8px 30px rgba(0,0,0,0.15);
}

.stTextInput input::placeholder {
    color: #64748b !important;
    text-align: center !important;
}

.stTextInput input:focus {
    border: 1px solid #8b5cf6 !important;
    box-shadow:
        0 0 0 3px rgba(139,92,246,0.15),
        0 0 30px rgba(139,92,246,0.12);
}


/* =====================================================
   WORD COUNT TITLE
===================================================== */

.count-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 15px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 15px;
}


/* =====================================================
   RADIO BUTTON
===================================================== */

div[role="radiogroup"] {
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    background: rgba(15,23,42,0.55);
    padding: 12px;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.12);
}

div[role="radiogroup"] label {
    background: rgba(30,41,59,0.75);
    border-radius: 12px;
    padding: 8px 14px;
    transition: all 0.25s ease;
}

div[role="radiogroup"] label:hover {
    background: rgba(99,102,241,0.25);
    transform: translateY(-2px);
}


/* =====================================================
   GENERATE BUTTON (Centered)
===================================================== */

.stButton {
    display: flex;
    justify-content: center; /* Center the button wrapper */
    margin-top: 30px;
}

.stButton > button {
    width: 100%;
    max-width: 350px; /* Limits width on big screens, keeps it centered */
    border: none;
    border-radius: 16px;
    padding: 15px 20px;
    font-size: 17px;
    font-weight: 800;
    color: white;
    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed,
            #c026d3
        );
    box-shadow: 0 10px 30px rgba(124,58,237,0.30);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(168,85,247,0.45);
    filter: brightness(1.08);
}

.stButton > button:active {
    transform: scale(0.98);
}


/* =====================================================
   PREDICTION CARD
===================================================== */

.prediction-box {
    position: relative;
    text-align: center; /* Center result text */
    margin-top: 35px;
    padding: 30px;
    border-radius: 24px;
    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.85),
            rgba(15,23,42,0.92)
        );
    border: 1px solid rgba(139,92,246,0.25);
    box-shadow:
        0 20px 60px rgba(0,0,0,0.35),
        0 0 40px rgba(99,102,241,0.08);
    overflow: hidden;
}

.prediction-label {
    color: #a78bfa;
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 14px;
}

.prediction-text {
    color: #f8fafc;
    font-size: 22px;
    line-height: 1.8;
    font-weight: 600;
    word-wrap: break-word;
}


/* =====================================================
   SUCCESS MESSAGE
===================================================== */

div[data-testid="stAlert"] {
    text-align: center;
    border-radius: 14px;
    border: 1px solid rgba(34,197,94,0.25);
    background: rgba(22,101,52,0.15);
}


/* =====================================================
   FOOTER
===================================================== */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid rgba(148,163,184,0.08);
}

.footer span {
    color: #a78bfa;
    font-weight: 700;
}


/* =====================================================
   MOBILE RESPONSIVENESS
===================================================== */

@media (max-width: 600px) {
    .main-title {
        font-size: 2.5rem;
    }
    .subtitle {
        font-size: 0.95rem;
    }
    .stButton > button {
        max-width: 100%; /* Full width button on small phones */
    }
    .prediction-box {
        padding: 22px;
    }
    .prediction-text {
        font-size: 18px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. Load Model, Tokenizer and max_len
# =========================================================

@st.cache_resource
def load_assets():
    model = load_model("BiGRU_Model.keras")
    with open("tokenizer (1).pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

try:
    model, tokenizer, max_len = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()


# =========================================================
# 4. Generate Multiple Words
# =========================================================

def generate_text(model, tokenizer, text, max_len, num_words):
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = sequence[-max_len:]
        sequence = pad_sequences(
            [sequence],
            maxlen=max_len,
            padding="post"
        )
        pred = model.predict(sequence, verbose=0)
        pred_index = np.argmax(pred)
        next_word = tokenizer.index_word.get(pred_index, "")
        
        if not next_word:
            break
            
        text += " " + next_word
        
    return text


# =========================================================
# 5. Header
# =========================================================

st.markdown(
    '<div class="main-title">✨ Next Word AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Generate intelligent text using your '
    '<b>GRU model</b>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 6. Input
# =========================================================

# Custom centered label
st.markdown(
    '<div class="custom-input-label">Enter your text</div>',
    unsafe_allow_html=True
)

input_text = st.text_input(
    "Enter your text",
    placeholder="Try something like: why is a...",
    label_visibility="collapsed" # Hides the default left-aligned label
)


# =========================================================
# 7. Word Count Selector
# =========================================================

st.markdown(
    '<div class="count-title">'
    '⚡ Choose how many words to generate'
    '</div>',
    unsafe_allow_html=True
)

word_count = st.radio(
    "Select number of words",
    options=list(range(1, 11)),
    horizontal=True,
    label_visibility="collapsed"
)


# =========================================================
# 8. Generate Button
# =========================================================

generate_button = st.button(
    f"✨ Generate {word_count} Word"
    + ("s" if word_count > 1 else "")
)


# =========================================================
# 9. Prediction
# =========================================================

if generate_button:
    if not input_text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner(
            f"Generating {word_count} word"
            + ("s..." if word_count > 1 else "...")
        ):
            result = generate_text(
                model,
                tokenizer,
                input_text,
                max_len,
                word_count
            )

        # Result card - Flush left to prevent Markdown code block rendering
        st.markdown(
            f"""
<div class="prediction-box">
    <div class="prediction-label">
        ✨ AI Generated Text
    </div>
    <div class="prediction-text">
        {result}
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        st.success(
            f"🎉 Successfully generated "
            f"{word_count} "
            + ("words!" if word_count > 1 else "word!")
        )


# =========================================================
# 10. Footer
# =========================================================

st.markdown(
    """
<div class="footer">
    Built with <span>❤️</span> using
    <span>Streamlit</span> +
    <span>TensorFlow</span> +
    <span>GRU model</span>
</div>
""",
    unsafe_allow_html=True
)