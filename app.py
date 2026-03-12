import gradio as gr
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import re

# Model configuration
MODEL_NAME = "fakespot-ai/roberta-base-ai-text-detection-v1"

# Load model and tokenizer globally
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully!")


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
    # AI-generated text tends to have more uniform sentence lengths (lower burstiness)
    # Human text tends to have more varied sentence lengths (higher burstiness)
    if len(sentence_lengths) > 1:
        mean_len = np.mean(sentence_lengths)
        std_len = np.std(sentence_lengths)
        # Burstiness formula: (std - mean) / (std + mean), normalized to 0-1 range
        if (std_len + mean_len) > 0:
            burstiness = (std_len - mean_len) / (std_len + mean_len)
            # Normalize to 0-1 range (original range is -1 to 1)
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
    """
    Estimate a perplexity-like metric using the model's confidence.
    Note: This is not true perplexity (which requires a language model),
    but rather a confidence-based metric.
    """
    # Tokenize the text
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

    # Get probabilities
    probs = softmax(logits.numpy()[0])

    # Calculate entropy as a proxy for perplexity
    # Higher entropy = more uncertainty = higher "perplexity"
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Convert to a perplexity-like score (exponential of entropy)
    perplexity_proxy = np.exp(entropy)

    return round(perplexity_proxy, 4)


def analyze_text(text):
    """Main function to analyze text for AI detection."""
    if not text or not text.strip():
        return "Please enter some text to analyze.", "", ""

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

    # Get labels (model typically has labels: 0=Human, 1=AI)
    # Check the model's config for label mapping
    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: "Human", 1: "AI"}

    # Determine prediction
    predicted_class = np.argmax(probs)
    predicted_label = id2label.get(predicted_class, f"Class {predicted_class}")
    confidence = probs[predicted_class] * 100

    # Create result summary
    if "ai" in predicted_label.lower() or "machine" in predicted_label.lower() or "fake" in predicted_label.lower():
        result_emoji = "AI-Generated Text"
        result_color = "red"
    else:
        result_emoji = "Human-Written Text"
        result_color = "green"

    # Format main result
    main_result = f"""
## Detection Result: {result_emoji}

**Prediction:** {predicted_label}
**Confidence:** {confidence:.2f}%

### Probability Breakdown:
"""
    for idx, prob in enumerate(probs):
        label = id2label.get(idx, f"Class {idx}")
        main_result += f"- **{label}:** {prob*100:.2f}%\n"

    # Calculate statistics
    stats = calculate_text_statistics(text)
    perplexity_proxy = estimate_perplexity(text, model, tokenizer)

    # Format statistics
    statistics = f"""
## Text Statistics

| Metric | Value |
|--------|-------|
| Word Count | {stats['word_count']} |
| Sentence Count | {stats['sentence_count']} |
| Avg. Sentence Length | {stats['avg_sentence_length']} words |
| Vocabulary Richness (TTR) | {stats['vocabulary_richness']:.2%} |

## AI Detection Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Burstiness** | {stats['burstiness']:.4f} | {'Low (AI-like: uniform sentences)' if stats['burstiness'] < 0.4 else 'Medium' if stats['burstiness'] < 0.6 else 'High (Human-like: varied sentences)'} |
| **Perplexity Proxy** | {perplexity_proxy:.4f} | {'Low uncertainty' if perplexity_proxy < 1.5 else 'Medium uncertainty' if perplexity_proxy < 2.0 else 'High uncertainty'} |
"""

    # Interpretation guide
    interpretation = """
## Understanding the Metrics

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
"""

    return main_result, statistics, interpretation


# Create Gradio interface
with gr.Blocks(title="AI Text Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # AI Text Detection Tool

    This tool uses the **fakespot-ai/roberta-base-ai-text-detection-v1** model to detect whether text is
    AI-generated or human-written.

    Enter your text below and click "Analyze" to get the detection results along with various text statistics.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Enter Text to Analyze",
                placeholder="Paste your text here (can be multiple paragraphs)...",
                lines=15,
                max_lines=30
            )
            analyze_btn = gr.Button("Analyze Text", variant="primary", size="lg")
            clear_btn = gr.Button("Clear", variant="secondary")

    with gr.Row():
        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Detection Result")
        with gr.Column(scale=1):
            stats_output = gr.Markdown(label="Statistics")

    with gr.Row():
        interpretation_output = gr.Markdown(label="Interpretation Guide")

    # Sample texts for demonstration
    gr.Markdown("### Sample Texts for Testing")
    with gr.Row():
        sample1_btn = gr.Button("Load AI Sample")
        sample2_btn = gr.Button("Load Human Sample")

    # AI-generated sample
    ai_sample = """Artificial intelligence has revolutionized numerous industries in recent years. The technology enables machines to learn from experience, adjust to new inputs, and perform human-like tasks. From healthcare to finance, AI applications are transforming how businesses operate and deliver value to customers.

Machine learning, a subset of AI, allows systems to automatically learn and improve from experience without being explicitly programmed. This capability has led to breakthroughs in areas such as natural language processing, computer vision, and predictive analytics.

As AI continues to evolve, it presents both opportunities and challenges for society. While the technology promises increased efficiency and new capabilities, it also raises important questions about privacy, employment, and ethical considerations."""

    # Human-written sample (more varied style)
    human_sample = """I never thought I'd say this, but gardening has become my obsession. Started last spring - just a few tomato plants, nothing fancy. Now? My backyard looks like a jungle exploded.

The thing nobody tells you about tomatoes is how MUCH they produce. We're drowning in them! Made salsa yesterday. Gave bags to neighbors. Still have more.

My mom keeps asking when I'll grow "real vegetables." Whatever that means, Mom. Cherry tomatoes ARE real. They're small but mighty. Like me, I guess?

Anyway, thinking about trying peppers next year. Or maybe herbs? Basil would be nice with all these tomatoes..."""

    # Event handlers
    analyze_btn.click(
        fn=analyze_text,
        inputs=[input_text],
        outputs=[result_output, stats_output, interpretation_output]
    )

    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        inputs=[],
        outputs=[input_text, result_output, stats_output, interpretation_output]
    )

    sample1_btn.click(
        fn=lambda: ai_sample,
        inputs=[],
        outputs=[input_text]
    )

    sample2_btn.click(
        fn=lambda: human_sample,
        inputs=[],
        outputs=[input_text]
    )


if __name__ == "__main__":
    print("Starting AI Text Detection Web App...")
    print("The model will be downloaded automatically on first run.")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
