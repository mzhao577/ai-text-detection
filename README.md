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
- **Segment-by-Segment Analysis**: Analyzes longer texts (~200 word chunks) to identify which portions are AI vs human-written
- **Sentence-by-Sentence Analysis**: Color-coded highlighting of individual sentences
  - Green = Human-written, Red = AI-generated, Gray = Too short to analyze
  - Shade intensity indicates confidence level
- **Duplicate Detection**: Identifies exact duplicate paragraphs with color-coded highlighting
  - Shows duplication percentage
  - Groups identical paragraphs with matching colors
- **Sample Texts**: Includes pre-loaded AI and human text samples for testing

## Live Demo

Try the app here: [https://ai-text-detection.streamlit.app](https://ai-text-detection.streamlit.app)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mzhao577/ai-text-detection.git
   cd ai-text-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   **Option 1: Streamlit (Recommended for deployment)**
   ```bash
   streamlit run streamlit_app.py
   ```
   Then open `http://localhost:8501`

   **Option 2: Gradio**
   ```bash
   python app.py
   ```
   Then open `http://localhost:7860`

## Deploy to Streamlit Cloud

1. Fork or push this repository to your GitHub account
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to `streamlit_app.py`
6. Click "Deploy"

The app will be publicly available at `https://your-app-name.streamlit.app`

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Streamlit or Gradio

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
