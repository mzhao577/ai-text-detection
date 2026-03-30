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

    /* Segment highlighting for AI detection */
    .ai-segment {
        background-color: rgba(255, 99, 99, 0.3);
        border-left: 3px solid #ff6b6b;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
    }

    .human-segment {
        background-color: rgba(99, 255, 132, 0.2);
        border-left: 3px solid #63ff84;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
    }

    .segment-label {
        font-size: 11px;
        font-weight: bold;
        margin-bottom: 4px;
    }

    .segment-text {
        font-size: 13px;
        color: #fafafa;
    }

    .segment-confidence {
        font-size: 11px;
        color: #b0b0b0;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Model configurations
MODELS = {
    "Fakespot RoBERTa": {
        "name": "fakespot-ai/roberta-base-ai-text-detection-v1",
        "description": "Fine-tuned for AI text detection by Fakespot",
        "labels": {0: "Human", 1: "AI-Generated"}
    }
    # OpenAI RoBERTa disabled to reduce storage usage on Streamlit Cloud
    # "OpenAI RoBERTa": {
    #     "name": "openai-community/roberta-base-openai-detector",
    #     "description": "OpenAI's GPT-2 output detector",
    #     "labels": {0: "Human", 1: "AI-Generated"}
    # }
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


def split_into_chunks(text, target_words=200):
    """Split text into chunks of approximately target_words each, respecting paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    if not paragraphs:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_word_count + para_words <= target_words * 1.3:
            current_chunk.append(para)
            current_word_count += para_words
        else:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_word_count = para_words

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def analyze_segments(text, model, tokenizer, min_words=200):
    """Analyze text segment by segment. Only performs segmentation if text has >= min_words."""
    total_words = len(text.split())

    if total_words < min_words:
        return None

    chunks = split_into_chunks(text, target_words=200)

    if len(chunks) <= 1:
        return None

    results = []
    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: "Human", 1: "AI"}

    for i, chunk in enumerate(chunks):
        chunk_words = len(chunk.split())

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probs = softmax(logits.numpy()[0])
        predicted_class = np.argmax(probs)
        predicted_label = id2label.get(predicted_class, f"Class {predicted_class}")
        confidence = probs[predicted_class] * 100

        label_lower = predicted_label.lower()
        is_ai = any(term in label_lower for term in ["ai", "machine", "fake", "generated"])

        results.append({
            "chunk_num": i + 1,
            "text": chunk,
            "word_count": chunk_words,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "is_ai": is_ai
        })

    return results


def analyze_sentences(text, model, tokenizer, min_words=5):
    """Analyze each sentence individually for AI detection."""
    # Split text into sentences
    sentences = re.split(r'([.!?]+)', text)

    # Reconstruct sentences with their punctuation
    reconstructed = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i + 1] if i + 1 < len(sentences) else ''
        if sentence:
            reconstructed.append(sentence + punct)
    # Handle last sentence if no punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        reconstructed.append(sentences[-1].strip())

    if not reconstructed:
        return None

    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: "Human", 1: "AI"}

    results = []
    for sentence in reconstructed:
        word_count = len(sentence.split())

        if word_count < min_words:
            # Too short for reliable prediction - mark as neutral
            results.append({
                'text': sentence,
                'is_short': True,
                'confidence': 0,
                'is_ai': None,
                'ai_prob': 0,
                'human_prob': 0
            })
        else:
            # Run prediction
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = softmax(outputs.logits.numpy()[0])

            # Determine AI vs Human based on label
            ai_idx = None
            human_idx = None
            for idx, label in id2label.items():
                label_lower = label.lower()
                if any(term in label_lower for term in ["ai", "machine", "fake", "generated"]):
                    ai_idx = idx
                else:
                    human_idx = idx

            # Default if labels not recognized
            if ai_idx is None:
                ai_idx = 1
            if human_idx is None:
                human_idx = 0

            ai_prob = float(probs[ai_idx])
            human_prob = float(probs[human_idx])
            is_ai = ai_prob > human_prob
            confidence = ai_prob if is_ai else human_prob

            results.append({
                'text': sentence,
                'is_short': False,
                'confidence': confidence,
                'is_ai': is_ai,
                'ai_prob': ai_prob,
                'human_prob': human_prob
            })

    return results


def render_highlighted_text(sentence_results):
    """Generate HTML with color-coded sentences based on AI/Human classification."""
    html_parts = []

    for result in sentence_results:
        # Escape HTML in text
        text = result['text'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        if result['is_short']:
            # Neutral styling for short sentences (gray)
            html_parts.append(
                f'<span style="background-color: rgba(128,128,128,0.2); padding: 2px 4px; border-radius: 3px;" '
                f'title="Too short to analyze ({len(result["text"].split())} words)">{text}</span>'
            )
        elif result['is_ai']:
            # Red for AI - opacity based on confidence (0.5-1.0 maps to 0.2-0.7)
            opacity = 0.2 + (result['confidence'] - 0.5) * 1.0
            opacity = max(0.2, min(0.7, opacity))  # Clamp between 0.2 and 0.7
            html_parts.append(
                f'<span style="background-color: rgba(255,99,99,{opacity:.2f}); padding: 2px 4px; border-radius: 3px;" '
                f'title="AI-Generated: {result["confidence"]*100:.1f}% confidence">{text}</span>'
            )
        else:
            # Green for Human - opacity based on confidence
            opacity = 0.2 + (result['confidence'] - 0.5) * 1.0
            opacity = max(0.2, min(0.7, opacity))  # Clamp between 0.2 and 0.7
            html_parts.append(
                f'<span style="background-color: rgba(99,255,132,{opacity:.2f}); padding: 2px 4px; border-radius: 3px;" '
                f'title="Human-Written: {result["confidence"]*100:.1f}% confidence">{text}</span>'
            )

    return ' '.join(html_parts)


# Color palette for duplicate groups
DUPLICATE_COLORS = [
    ("Orange", "rgba(255, 165, 0, 0.4)"),
    ("Purple", "rgba(138, 43, 226, 0.4)"),
    ("Cyan", "rgba(0, 191, 255, 0.4)"),
    ("Yellow", "rgba(255, 255, 0, 0.4)"),
    ("Pink", "rgba(255, 105, 180, 0.4)"),
    ("Lime", "rgba(50, 205, 50, 0.4)"),
]


def detect_duplicates(text):
    """Detect exact duplicate paragraphs in the text."""
    # Split into paragraphs (by newlines)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not paragraphs:
        return None

    # Track paragraphs by their normalized content
    # Key: normalized text, Value: list of (index, original_text)
    paragraph_map = {}

    for i, para in enumerate(paragraphs):
        # Normalize: strip whitespace and convert to lowercase for comparison
        normalized = ' '.join(para.split()).lower()

        if normalized not in paragraph_map:
            paragraph_map[normalized] = []
        paragraph_map[normalized].append({
            'index': i + 1,  # 1-based index
            'text': para
        })

    # Find duplicate groups (groups with more than one paragraph)
    duplicate_groups = []
    for normalized, paras in paragraph_map.items():
        if len(paras) > 1:
            duplicate_groups.append(paras)

    # Calculate duplication percentage
    total_chars = sum(len(p) for p in paragraphs)
    duplicated_chars = 0
    for group in duplicate_groups:
        # Count all but the first occurrence as duplicated
        for para in group[1:]:
            duplicated_chars += len(para['text'])

    duplication_percentage = (duplicated_chars / total_chars * 100) if total_chars > 0 else 0

    # Create paragraph info list with duplicate group assignments
    paragraphs_with_info = []
    for i, para in enumerate(paragraphs):
        normalized = ' '.join(para.split()).lower()

        # Find which duplicate group this paragraph belongs to
        group_index = None
        for gi, group in enumerate(duplicate_groups):
            if any(p['index'] == i + 1 for p in group):
                group_index = gi
                break

        paragraphs_with_info.append({
            'index': i + 1,
            'text': para,
            'duplicate_group': group_index  # None if not a duplicate
        })

    return {
        'paragraphs': paragraphs_with_info,
        'duplicate_groups': duplicate_groups,
        'duplication_percentage': duplication_percentage,
        'total_paragraphs': len(paragraphs)
    }


def render_duplicate_highlighted_text(duplicate_result):
    """Generate HTML with color-coded duplicate paragraphs."""
    if not duplicate_result:
        return ""

    html_parts = []

    for para in duplicate_result['paragraphs']:
        # Escape HTML in text
        text = para['text'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        if para['duplicate_group'] is not None:
            # This is a duplicate - apply color from the group
            color_index = para['duplicate_group'] % len(DUPLICATE_COLORS)
            color_name, color_rgba = DUPLICATE_COLORS[color_index]

            html_parts.append(
                f'<div style="background-color: {color_rgba}; padding: 8px 12px; margin: 4px 0; '
                f'border-radius: 4px; border-left: 3px solid {color_rgba.replace("0.4", "1.0")};" '
                f'title="Duplicate Group {para["duplicate_group"] + 1} ({color_name})">'
                f'<span style="font-size: 11px; color: #888;">P{para["index"]}</span> {text}</div>'
            )
        else:
            # Not a duplicate - no highlighting
            html_parts.append(
                f'<div style="padding: 8px 12px; margin: 4px 0; border-radius: 4px; '
                f'background-color: rgba(255,255,255,0.05);">'
                f'<span style="font-size: 11px; color: #888;">P{para["index"]}</span> {text}</div>'
            )

    return '\n'.join(html_parts)


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
    <strong>Model:</strong><br>
    • <strong>Fakespot RoBERTa:</strong> Fine-tuned for general AI text detection<br><br>
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

                # Segment-level analysis (only if 200+ words)
                st.markdown("---")
                segment_results = analyze_segments(input_text, model, tokenizer, min_words=200)

                if segment_results:
                    st.markdown("<p class='stats-header'>📝 Segment-by-Segment Analysis</p>", unsafe_allow_html=True)
                    st.markdown("<p class='small-font'><em>Text split into ~200 word chunks for detailed analysis</em></p>", unsafe_allow_html=True)

                    ai_count = sum(1 for s in segment_results if s['is_ai'])
                    human_count = len(segment_results) - ai_count
                    st.markdown(f"<p class='small-font'><strong>Summary:</strong> {ai_count} AI-detected segments, {human_count} human-detected segments</p>", unsafe_allow_html=True)

                    for seg in segment_results:
                        css_class = "ai-segment" if seg['is_ai'] else "human-segment"
                        label_icon = "🤖 AI" if seg['is_ai'] else "✍️ Human"
                        preview = seg['text'][:300] + "..." if len(seg['text']) > 300 else seg['text']
                        preview = preview.replace('\n', ' ')

                        st.markdown(f"""
                        <div class="{css_class}">
                            <div class="segment-label">Segment {seg['chunk_num']} - {label_icon} ({seg['confidence']:.1f}% confidence)</div>
                            <div class="segment-text">{preview}</div>
                            <div class="segment-confidence">{seg['word_count']} words</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<p class='small-font'><em>Segment analysis requires 200+ words with multiple paragraphs.</em></p>", unsafe_allow_html=True)

                # Sentence-by-sentence analysis
                st.markdown("---")
                st.markdown("<p class='stats-header'>🔬 Sentence-by-Sentence Analysis</p>", unsafe_allow_html=True)

                sentence_results = analyze_sentences(input_text, model, tokenizer, min_words=5)

                if sentence_results:
                    # Legend
                    st.markdown("""
                    <p class='small-font'>
                    <strong>Legend:</strong>
                    <span style="background-color: rgba(99,255,132,0.5); padding: 2px 6px; border-radius: 3px;">Human-written</span>
                    <span style="background-color: rgba(255,99,99,0.5); padding: 2px 6px; border-radius: 3px; margin-left: 8px;">AI-generated</span>
                    <span style="background-color: rgba(128,128,128,0.3); padding: 2px 6px; border-radius: 3px; margin-left: 8px;">Too short</span>
                    <br><em>(Darker shade = higher confidence. Hover for details.)</em>
                    </p>
                    """, unsafe_allow_html=True)

                    # Render highlighted text
                    highlighted_html = render_highlighted_text(sentence_results)
                    st.markdown(f"""
                    <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; line-height: 1.8; margin: 10px 0;">
                    {highlighted_html}
                    </div>
                    """, unsafe_allow_html=True)

                    # Summary statistics
                    ai_sentences = [s for s in sentence_results if s['is_ai'] is True]
                    human_sentences = [s for s in sentence_results if s['is_ai'] is False]
                    short_sentences = [s for s in sentence_results if s['is_short']]

                    st.markdown(f"""
                    <p class='small-font'>
                    <strong>Summary:</strong> {len(ai_sentences)} AI-detected, {len(human_sentences)} human-detected, {len(short_sentences)} too short to analyze
                    </p>
                    """, unsafe_allow_html=True)

                    # Detailed breakdown in expander
                    with st.expander("📋 View Sentence Details"):
                        for i, sent in enumerate(sentence_results, 1):
                            if sent['is_short']:
                                icon = "⚪"
                                label = "Too short"
                                conf = "N/A"
                            elif sent['is_ai']:
                                icon = "🔴"
                                label = "AI"
                                conf = f"{sent['confidence']*100:.1f}%"
                            else:
                                icon = "🟢"
                                label = "Human"
                                conf = f"{sent['confidence']*100:.1f}%"

                            preview = sent['text'][:100] + "..." if len(sent['text']) > 100 else sent['text']
                            st.markdown(f"<p class='small-font'>{icon} <strong>S{i}</strong> [{label}, {conf}]: {preview}</p>", unsafe_allow_html=True)

                # Duplicate Detection
                st.markdown("---")
                st.markdown("<p class='stats-header'>📋 Duplicate Detection</p>", unsafe_allow_html=True)

                duplicate_result = detect_duplicates(input_text)

                if duplicate_result and duplicate_result['duplicate_groups']:
                    # Show duplication percentage
                    st.markdown(f"""
                    <p class='small-font'>
                    <strong>Duplication:</strong> {duplicate_result['duplication_percentage']:.1f}% of content is duplicated
                    ({len(duplicate_result['duplicate_groups'])} duplicate group(s) found)
                    </p>
                    """, unsafe_allow_html=True)

                    # Legend for duplicate colors
                    legend_items = []
                    for i, group in enumerate(duplicate_result['duplicate_groups']):
                        if i < len(DUPLICATE_COLORS):
                            color_name, color_rgba = DUPLICATE_COLORS[i]
                            legend_items.append(
                                f'<span style="background-color: {color_rgba}; padding: 2px 6px; border-radius: 3px; margin-right: 8px;">Group {i+1} ({color_name})</span>'
                            )

                    st.markdown(f"""
                    <p class='small-font'>
                    <strong>Legend:</strong> {''.join(legend_items)}
                    </p>
                    """, unsafe_allow_html=True)

                    # Render highlighted text
                    duplicate_html = render_duplicate_highlighted_text(duplicate_result)
                    st.markdown(f"""
                    <div style="background-color: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px; margin: 10px 0;">
                    {duplicate_html}
                    </div>
                    """, unsafe_allow_html=True)

                    # Show duplicate groups details
                    with st.expander("📋 View Duplicate Groups"):
                        for i, group in enumerate(duplicate_result['duplicate_groups']):
                            color_name, color_rgba = DUPLICATE_COLORS[i % len(DUPLICATE_COLORS)]
                            para_indices = [str(p['index']) for p in group]
                            st.markdown(f"""
                            <p class='small-font'>
                            <span style="background-color: {color_rgba}; padding: 2px 6px; border-radius: 3px;">Group {i+1}</span>
                            Paragraphs {', '.join(para_indices)} are identical
                            </p>
                            """, unsafe_allow_html=True)

                            # Show the duplicated text (preview)
                            preview = group[0]['text'][:150] + "..." if len(group[0]['text']) > 150 else group[0]['text']
                            st.markdown(f"<p class='small-font' style='color: #888; margin-left: 20px;'><em>\"{preview}\"</em></p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p class='small-font'>No duplicate paragraphs detected.</p>", unsafe_allow_html=True)

        elif analyze_button and not input_text:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
