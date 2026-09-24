# 📱 Spam Message Detector

A simple machine learning web app that checks whether a text message is **Spam** or **Normal (Ham)**. Type or paste a message, click a button, and get an instant result.

Built with Python, scikit-learn, and Streamlit, and trained on the UCI SMS Spam Collection dataset.

## Overview

Spam messages such as fake prizes, scam links, and unwanted ads are a daily annoyance and can lead to fraud. This project teaches a computer to recognize spam by learning from thousands of labeled SMS messages, then exposes the trained model through a simple web interface.

## How It Works

1. **Load data:** the UCI SMS Spam dataset is loaded from Hugging Face (about 5,500 labeled messages).
2. **Split data:** 80% for training and 20% for testing.
3. **Vectorize text:** messages are converted into numbers using TF-IDF, which gives more weight to distinctive words.
4. **Train the model:** a Multinomial Naive Bayes classifier learns which words and patterns are common in spam.
5. **Evaluate and save:** accuracy is measured on unseen test messages, and the model and vectorizer are saved with joblib.
6. **Serve the model:** the Streamlit app loads the saved files and predicts on any message the user enters.

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| Hugging Face Datasets | Loading the SMS dataset |
| scikit-learn | TF-IDF vectorizer and Naive Bayes model |
| Joblib | Saving and loading the trained model |
| Streamlit | Web app interface |

## Project Structure

```
app.py              - Streamlit web app
train_model.py      - Trains the model and saves it
Load_dataset.py     - Quick script to explore the dataset
spam_model.pkl      - Saved trained model
vectorizer.pkl      - Saved TF-IDF vectorizer
requirements.txt    - Python dependencies
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the app:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually http://localhost:8501).

To retrain the model (optional):

```bash
python train_model.py
```

Note: run all commands from the project folder so the app can find the `.pkl` files.

## Example

| Message | Result |
|---|---|
| Congratulations! You won a free prize. Click the link to claim now. | 🚨 Spam |
| Hey, are we meeting for lunch tomorrow? | ✅ Normal (Ham) |

## Future Improvements

- Show a confidence percentage with each prediction
- Add more evaluation metrics (precision, recall, confusion matrix)
- Support messages in other languages
- Try other models (e.g. Logistic Regression, SVM)
- Deploy online (e.g. Streamlit Community Cloud)

## Dataset

[UCI SMS Spam Collection](https://huggingface.co/datasets/ucirvine/sms_spam)
