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

# Model configuration
MODEL_NAME = "fakespot-ai/roberta-base-ai-text-detection-v1"


@st.cache_resource
def load_model():
    """Load model and tokenizer with caching."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
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
    st.markdown("""
    This tool uses the **fakespot-ai/roberta-base-ai-text-detection-v1** model to detect whether text is
    AI-generated or human-written.
    """)

    # Load model
    with st.spinner("Loading model..."):
        tokenizer, model = load_model()

    # Sidebar with sample texts
    st.sidebar.header("Sample Texts")
    if st.sidebar.button("Load AI Sample", use_container_width=True):
        st.session_state.input_text = AI_SAMPLE
    if st.sidebar.button("Load Human Sample", use_container_width=True):
        st.session_state.input_text = HUMAN_SAMPLE

    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.markdown("""
    **Model:** RoBERTa-base fine-tuned for AI text detection

    **Metrics Explained:**
    - **Burstiness:** Measures sentence length variation (0-1). Low = uniform (AI-like), High = varied (human-like)
    - **Perplexity Proxy:** Model confidence metric
    - **Vocabulary Richness:** Ratio of unique words to total words
    """)

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
            height=400,
            placeholder="Paste your text here..."
        )

        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

    with col2:
        st.subheader("Results")

        if analyze_button and input_text:
            with st.spinner("Analyzing..."):
                result = analyze_text(input_text, model, tokenizer)

            if result:
                # Main result
                is_ai = "ai" in result["predicted_label"].lower() or "machine" in result["predicted_label"].lower() or "fake" in result["predicted_label"].lower()

                if is_ai:
                    st.error(f"### 🤖 AI-Generated Text")
                else:
                    st.success(f"### ✍️ Human-Written Text")

                st.markdown(f"**Prediction:** {result['predicted_label']}")
                st.markdown(f"**Confidence:** {result['confidence']:.2f}%")

                # Progress bar for confidence
                st.progress(result['confidence'] / 100)

                st.markdown("---")

                # Probability breakdown
                st.markdown("#### Probability Breakdown")
                for idx, prob in enumerate(result['probs']):
                    label = result['id2label'].get(idx, f"Class {idx}")
                    st.markdown(f"**{label}:** {prob*100:.2f}%")
                    st.progress(float(prob))

        elif analyze_button and not input_text:
            st.warning("Please enter some text to analyze.")

    # Statistics section (shown below when there are results)
    if analyze_button and input_text:
        result = analyze_text(input_text, model, tokenizer)
        if result:
            st.markdown("---")
            st.subheader("📊 Text Statistics & AI Detection Metrics")

            col_stats1, col_stats2, col_stats3 = st.columns(3)

            with col_stats1:
                st.markdown("#### Basic Statistics")
                st.metric("Word Count", result['stats']['word_count'])
                st.metric("Sentence Count", result['stats']['sentence_count'])
                st.metric("Avg. Sentence Length", f"{result['stats']['avg_sentence_length']} words")

            with col_stats2:
                st.markdown("#### AI Detection Metrics")
                burstiness = result['stats']['burstiness']
                burstiness_label = "Low (AI-like)" if burstiness < 0.4 else "Medium" if burstiness < 0.6 else "High (Human-like)"
                st.metric("Burstiness", f"{burstiness:.4f}", burstiness_label)

                perplexity = result['perplexity_proxy']
                perplexity_label = "Low uncertainty" if perplexity < 1.5 else "Medium" if perplexity < 2.0 else "High uncertainty"
                st.metric("Perplexity Proxy", f"{perplexity:.4f}", perplexity_label)

            with col_stats3:
                st.markdown("#### Vocabulary Analysis")
                vocab_richness = result['stats']['vocabulary_richness']
                st.metric("Vocabulary Richness (TTR)", f"{vocab_richness:.2%}")

            # Interpretation guide
            with st.expander("ℹ️ Understanding the Metrics"):
                st.markdown("""
                ### Burstiness (0-1 scale)
                - Measures variation in sentence lengths
                - **Low burstiness (<0.4):** More uniform sentence structure, often associated with AI text
                - **High burstiness (>0.6):** More varied sentence structure, often associated with human writing

                ### Perplexity Proxy
                - Based on model's prediction confidence
                - **Lower values:** Model is more confident in its prediction
                - **Higher values:** More uncertainty in the prediction

                ### Vocabulary Richness (Type-Token Ratio)
                - Ratio of unique words to total words
                - **Higher values:** More diverse vocabulary
                - Human text often shows more vocabulary variation

                *Note: These metrics are heuristic approximations. The primary AI detection is performed by the RoBERTa model.*
                """)


if __name__ == "__main__":
    main()
