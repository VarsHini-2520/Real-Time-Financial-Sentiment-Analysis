# Stock Sentiment Analysis using FinBERT & LSTM

This project implements an **AI-powered sentiment analysis system** to classify financial news, reports, and market statements as **Positive, Negative, or Neutral**, providing actionable investment insights. The system leverages a **hybrid FinBERT + LSTM model** for robust domain-specific sentiment detection and includes an interactive **Streamlit application** for real-time predictions and visualization.

---

## Features

- **Financial text processing**:
  - Data curation and preprocessing of large-scale financial datasets
  - Oversampling to address class imbalance
- **Hybrid model architecture**:
  - FinBERT for contextual embeddings
  - LSTM for sequential pattern recognition
- **Real-time predictions**:
  - Interactive Streamlit web app for end-users
  - Visualization of sentiment results
- **Performance metrics**:
  - Train Accuracy: 78.58%
  - Validation Accuracy: 80.09%
  - Test Accuracy: 80.40%
  - Balanced F1 scores — Negative: 0.87, Neutral: 0.76, Positive: 0.77


## Dependencies

1. Clone the repository:
```bash
git clone https://github.com/VarsHini-2520/Real-Time-Financial-Sentiment-Analysis
cd Stock-Sentiment-Analysis

2.Libraries 

pip install numpy pandas torch transformers streamlit scikit-learn matplotlib seaborn

