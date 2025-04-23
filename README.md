
# SmartDoc: Medical PDF Summarization & Classification System

SmartDoc is a medical document intelligence system that extracts text from PDF and image files, generates summaries, and classifies the content into illness categories using an AI model. Built with a Flask backend and a lightweight HTML/CSS frontend, this project is powered by FLAN-T5 for NLP tasks and is fine-tuning ready.

## Features

- Text Extraction from PDFs
- OCR for Image Files (JPG/PNG)
- Abstractive Summarization using FLAN-T5
- Medical Text Classification
- English Words Filtering
- Redundancy Removal for Cleaner Summaries
- Support for 4 Illness Categories:
  - 0: Acute  
  - 1: Mental  
  - 2: Chronic  
  - 3: Physical
- Easy Frontend Integration via HTML/CSS
- Fine-tune Ready with Structured Illness Dataset (12,500+ samples)
- Multi-label Classification Recommended for improved accuracy in diagnosis history

## Tech Stack

| Layer     | Tools & Frameworks              |
|-----------|---------------------------------|
| Backend   | Python, Flask                   |
| Frontend  | HTML, CSS                       |
| AI Model  | FLAN-T5 (transformers)          |
| OCR       | Tesseract via `pytesseract`     |
| Dataset   | Custom illness classification dataset (CSV) |
| Model I/O | HuggingFace Transformers        |

## Project Structure

```
SmartDoc/
│
├── app.py                   # Flask app with routing logic
├── TextSummarizer.py       # AI model for text processing
├── static/
│   ├── style.css           # Frontend styling
│   └── script.js           # Frontend scripting (optional)
├── templates/
│   └── index.html          # Web interface
├── uploads/                # File storage for user uploads
├── model/                  # (Optional) Fine-tuned model directory
└── requirements.txt
```

## Setup & Run

1. Install dependencies:

```

2. Start the Flask server:

```
python app.py
```

3. Access the app:
Open http://localhost:5000 in your browser.

## Dataset

A fine-tuning ready dataset with over 2500 labeled illness names in 4 categories. Available in multiple structured versions including multi-label combinations.
the data was genrated using AI .

## Recommendation

For real-world usage, we recommend switching to a multi-label classification approach. This allows the model to:
- Identify multiple diagnoses per document
- Extend support to additional illness categories
- Handle historical and comorbid conditions better

##Auther 
identities department team 
