import streamlit as st
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import re

# Page configuration
st.set_page_config(
    page_title="AI Text Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for dark theme and smaller fonts
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Make all text light colored */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #fafafa !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        color: #fafafa;
    }

    [data-testid="stSidebar"] * {
        color: #fafafa !important;
    }

    /* Labels and text */
    label, .stTextArea label, .stTextInput label {
        color: #fafafa !important;
    }

    /* Smaller font for statistics */
    .small-font {
        font-size: 14px !important;
        color: #fafafa !important;
    }

    .stats-header {
        font-size: 16px !important;
        font-weight: bold;
        margin-bottom: 8px;
        color: #fafafa !important;
    }

    .stats-item {
        font-size: 13px !important;
        margin: 4px 0;
        color: #fafafa !important;
    }

    /* Adjust metric font sizes and colors */
    [data-testid="stMetricValue"] {
        font-size: 18px !important;
        color: #fafafa !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        color: #b0b0b0 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 11px !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #fafafa !important;
    }
</style>
""", unsafe_allow_html=True)

# Model configurations
MODELS = {
    "Fakespot RoBERTa": {
        "name": "fakespot-ai/roberta-base-ai-text-detection-v1",
        "description": "Fine-tuned for AI text detection by Fakespot",
        "labels": {0: "Human", 1: "AI-Generated"}
    },
    "OpenAI RoBERTa": {
        "name": "openai-community/roberta-base-openai-detector",
        "description": "OpenAI's GPT-2 output detector",
        "labels": {0: "Human", 1: "AI-Generated"}
    }
}


@st.cache_resource
def load_model(model_key):
    """Load model and tokenizer with caching based on selected model."""
    model_info = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
    model = AutoModelForSequenceClassification.from_pretrained(model_info["name"])
    model.eval()
    return tokenizer, model


def calculate_text_statistics(text):
    """Calculate various text statistics including approximations of perplexity and burstiness."""
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Split into words
    words = text.split()

    if not sentences or not words:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "burstiness": 0,
            "vocabulary_richness": 0,
        }

    # Basic counts
    word_count = len(words)
    sentence_count = len(sentences)

    # Average sentence length
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0

    # Burstiness: measures variance in sentence lengths
    if len(sentence_lengths) > 1:
        mean_len = np.mean(sentence_lengths)
        std_len = np.std(sentence_lengths)
        if (std_len + mean_len) > 0:
            burstiness = (std_len - mean_len) / (std_len + mean_len)
            burstiness = (burstiness + 1) / 2
        else:
            burstiness = 0.5
    else:
        burstiness = 0.5

    # Vocabulary richness (Type-Token Ratio)
    unique_words = set(word.lower() for word in words)
    vocabulary_richness = len(unique_words) / word_count if word_count > 0 else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "burstiness": round(burstiness, 4),
        "vocabulary_richness": round(vocabulary_richness, 4),
    }


def estimate_perplexity(text, model, tokenizer):
    """Estimate a perplexity-like metric using the model's confidence."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = softmax(logits.numpy()[0])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    perplexity_proxy = np.exp(entropy)

    return round(perplexity_proxy, 4)


def analyze_text(text, model, tokenizer):
    """Main function to analyze text for AI detection."""
    if not text or not text.strip():
        return None

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert to probabilities
    probs = softmax(logits.numpy()[0])

    # Get labels
    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: "Human", 1: "AI"}

    # Determine prediction
    predicted_class = np.argmax(probs)
    predicted_label = id2label.get(predicted_class, f"Class {predicted_class}")
    confidence = probs[predicted_class] * 100

    # Calculate statistics
    stats = calculate_text_statistics(text)
    perplexity_proxy = estimate_perplexity(text, model, tokenizer)

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probs": probs,
        "id2label": id2label,
        "stats": stats,
        "perplexity_proxy": perplexity_proxy
    }


# Sample texts
AI_SAMPLE = """Artificial intelligence has revolutionized numerous industries in recent years. The technology enables machines to learn from experience, adjust to new inputs, and perform human-like tasks. From healthcare to finance, AI applications are transforming how businesses operate and deliver value to customers.

Machine learning, a subset of AI, allows systems to automatically learn and improve from experience without being explicitly programmed. This capability has led to breakthroughs in areas such as natural language processing, computer vision, and predictive analytics.

As AI continues to evolve, it presents both opportunities and challenges for society. While the technology promises increased efficiency and new capabilities, it also raises important questions about privacy, employment, and ethical considerations."""

HUMAN_SAMPLE = """I never thought I'd say this, but gardening has become my obsession. Started last spring - just a few tomato plants, nothing fancy. Now? My backyard looks like a jungle exploded.

The thing nobody tells you about tomatoes is how MUCH they produce. We're drowning in them! Made salsa yesterday. Gave bags to neighbors. Still have more.

My mom keeps asking when I'll grow "real vegetables." Whatever that means, Mom. Cherry tomatoes ARE real. They're small but mighty. Like me, I guess?

Anyway, thinking about trying peppers next year. Or maybe herbs? Basil would be nice with all these tomatoes..."""


# Main app
def main():
    st.title("🔍 AI Text Detection")

    # Model selection in sidebar (before other sidebar elements)
    st.sidebar.header("Model Selection")
    selected_model = st.sidebar.radio(
        "Choose detection model:",
        options=list(MODELS.keys()),
        index=0,
        help="Select which RoBERTa model to use for detection"
    )

    # Show model info
    model_info = MODELS[selected_model]
    st.sidebar.markdown(f"""
    <p class="small-font">
    <strong>Model:</strong> {model_info['name']}<br>
    <em>{model_info['description']}</em>
    </p>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    st.markdown(f"""
    <p class="small-font">Using <strong>{selected_model}</strong> model to detect whether text is AI-generated or human-written.</p>
    """, unsafe_allow_html=True)

    # Load model based on selection
    with st.spinner(f"Loading {selected_model} model..."):
        tokenizer, model = load_model(selected_model)

    # Sidebar with sample texts
    st.sidebar.header("Sample Texts")
    if st.sidebar.button("Load AI Sample", use_container_width=True):
        st.session_state.input_text = AI_SAMPLE
    if st.sidebar.button("Load Human Sample", use_container_width=True):
        st.session_state.input_text = HUMAN_SAMPLE

    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.markdown("""
    <p class="small-font">
    <strong>Available Models:</strong><br>
    • <strong>Fakespot RoBERTa:</strong> Fine-tuned for general AI text detection<br>
    • <strong>OpenAI RoBERTa:</strong> Trained to detect GPT-2 generated text<br><br>
    <strong>Metrics Explained:</strong><br>
    • <strong>Burstiness:</strong> Measures sentence length variation (0-1). Low = uniform (AI-like), High = varied (human-like)<br>
    • <strong>Perplexity Proxy:</strong> Model confidence metric<br>
    • <strong>Vocabulary Richness:</strong> Ratio of unique words to total words
    </p>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""

    # Main input area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Text")
        input_text = st.text_area(
            "Enter text to analyze (can be multiple paragraphs):",
            value=st.session_state.input_text,
            height=500,
            placeholder="Paste your text here..."
        )

        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

    with col2:
        st.subheader("Results")

        if analyze_button and input_text:
            with st.spinner("Analyzing..."):
                result = analyze_text(input_text, model, tokenizer)

            if result:
                # Main result - handle different label formats from different models
                # Fakespot: "human" vs "machine", OpenAI: "Real" vs "Fake"
                label_lower = result["predicted_label"].lower()
                is_ai = any(term in label_lower for term in ["ai", "machine", "fake", "generated"])

                if is_ai:
                    st.error("### 🤖 AI-Generated Text")
                else:
                    st.success("### ✍️ Human-Written Text")

                st.markdown(f"<p class='small-font'><strong>Prediction:</strong> {result['predicted_label']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='small-font'><strong>Confidence:</strong> {result['confidence']:.2f}%</p>", unsafe_allow_html=True)

                # Progress bar for confidence
                st.progress(float(result['confidence']) / 100)

                st.markdown("---")

                # Probability breakdown
                st.markdown("<p class='stats-header'>Probability Breakdown</p>", unsafe_allow_html=True)
                for idx, prob in enumerate(result['probs']):
                    label = result['id2label'].get(idx, f"Class {idx}")
                    st.markdown(f"<p class='stats-item'><strong>{label}:</strong> {prob*100:.2f}%</p>", unsafe_allow_html=True)
                    st.progress(float(prob))

                st.markdown("---")

                # Statistics section - now in the right column
                st.markdown("<p class='stats-header'>📊 Text Statistics & AI Detection Metrics</p>", unsafe_allow_html=True)

                # Basic Statistics
                st.markdown("<p class='stats-item'><strong>Basic Statistics:</strong></p>", unsafe_allow_html=True)
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Words", result['stats']['word_count'])
                with stat_col2:
                    st.metric("Sentences", result['stats']['sentence_count'])
                with stat_col3:
                    st.metric("Avg Length", f"{result['stats']['avg_sentence_length']}")

                # AI Detection Metrics
                st.markdown("<p class='stats-item'><strong>AI Detection Metrics:</strong></p>", unsafe_allow_html=True)
                metric_col1, metric_col2, metric_col3 = st.columns(3)

                burstiness = result['stats']['burstiness']
                burstiness_label = "Low (AI-like)" if burstiness < 0.4 else "Medium" if burstiness < 0.6 else "High (Human)"

                perplexity = result['perplexity_proxy']
                perplexity_label = "Low" if perplexity < 1.5 else "Medium" if perplexity < 2.0 else "High"

                vocab_richness = result['stats']['vocabulary_richness']

                with metric_col1:
                    st.metric("Burstiness", f"{burstiness:.3f}", burstiness_label)
                with metric_col2:
                    st.metric("Perplexity", f"{perplexity:.3f}", perplexity_label)
                with metric_col3:
                    st.metric("Vocab Rich.", f"{vocab_richness:.1%}")

                # Interpretation guide
                with st.expander("ℹ️ Understanding the Metrics"):
                    st.markdown("""
                    <p class="small-font">
                    <strong>Burstiness (0-1 scale):</strong> Measures variation in sentence lengths. Low (<0.4) = uniform (AI-like), High (>0.6) = varied (human-like)<br><br>
                    <strong>Perplexity Proxy:</strong> Based on model's prediction confidence. Lower = more confident.<br><br>
                    <strong>Vocabulary Richness:</strong> Ratio of unique words to total words. Higher = more diverse vocabulary.
                    </p>
                    """, unsafe_allow_html=True)

        elif analyze_button and not input_text:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
