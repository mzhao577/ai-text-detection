---
title: AI Text Detection
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Detect AI-generated vs human-written text
---

# AI Text Detection

This app uses the **fakespot-ai/roberta-base-ai-text-detection-v1** model to detect whether text is AI-generated or human-written.

## Features

- **AI/Human Text Detection**: Uses a fine-tuned RoBERTa model
- **Probability Scores**: Shows confidence percentages
- **Text Statistics**: Word count, sentence count, vocabulary richness
- **AI Detection Metrics**: Burstiness and perplexity proxy

## How to Use

1. Enter or paste text in the input box
2. Click "Analyze Text"
3. View the detection results and statistics

## Model

Uses [fakespot-ai/roberta-base-ai-text-detection-v1](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1) from Hugging Face.
