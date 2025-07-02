# Thai Text Classification Examples

This directory contains practical examples for different text classification scenarios using Thai language models.

## Available Examples

### 1. Basic Sentiment Analysis
- **File**: `basic_sentiment_analysis.py`
- **Description**: Simple positive/negative sentiment classification
- **Use Cases**: Product reviews, social media sentiment, customer feedback

### 2. Multi-class Topic Classification
- **File**: `topic_classification.py`
- **Description**: Classify text into multiple topic categories
- **Use Cases**: News categorization, document organization, content tagging

### 3. Intent Classification
- **File**: `intent_classification.py`
- **Description**: Classify user intents for chatbots and virtual assistants
- **Use Cases**: Customer service bots, voice assistants, automated routing

### 4. Custom Domain Classification
- **File**: `custom_domain_example.py`
- **Description**: Template for domain-specific classification tasks
- **Use Cases**: Medical text classification, legal document categorization, etc.

## Quick Start

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Run a basic example:
```python
python basic_sentiment_analysis.py
```

3. Customize for your use case by modifying the data and labels

## Data Format

All examples expect data in the following CSV format:
```csv
text,label
"Your Thai text here","positive"
"Another text sample","negative"
```

## Performance Tips

- Use GPU when available for faster training
- Start with smaller learning rates (1e-5 to 5e-5)
- Monitor validation metrics to prevent overfitting
- Use early stopping for optimal performance
