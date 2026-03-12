# AI Text Detection Web App

A web application for detecting AI-generated text using the **fakespot-ai/roberta-base-ai-text-detection-v1** model.

## Features

- **AI/Human Text Detection**: Uses a fine-tuned RoBERTa model to classify text as AI-generated or human-written
- **Probability Scores**: Shows confidence percentages for both classifications
- **Text Statistics**:
  - Word count
  - Sentence count
  - Average sentence length
  - Vocabulary richness (Type-Token Ratio)
- **AI Detection Metrics**:
  - **Burstiness**: Measures sentence length variation (AI text tends to be more uniform)
  - **Perplexity Proxy**: Based on model prediction confidence
- **Sample Texts**: Includes pre-loaded AI and human text samples for testing

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-text-detection.git
   cd ai-text-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:7860`

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Gradio

## Model

This app uses the [fakespot-ai/roberta-base-ai-text-detection-v1](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1) model from Hugging Face. The model is automatically downloaded on first run.

## Understanding the Metrics

### Burstiness (0-1 scale)
- Measures variation in sentence lengths
- **Low burstiness (<0.4)**: More uniform sentence structure, often associated with AI text
- **High burstiness (>0.6)**: More varied sentence structure, often associated with human writing

### Perplexity Proxy
- Based on model's prediction confidence
- **Lower values**: Model is more confident in its prediction
- **Higher values**: More uncertainty in the prediction

### Vocabulary Richness (Type-Token Ratio)
- Ratio of unique words to total words
- Higher values indicate more diverse vocabulary

## License

MIT License
