import pdfplumber
import re
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class TextSummarizer:
    def __init__(self, model_path="google/flan-t5-large", fine_tuned_path=None):
        """Initialize the summarization model and tokenizer."""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load fine-tuned model if provided
        self.fine_tuned_model = None
        self.fine_tuned_tokenizer = None
        if fine_tuned_path:
            self.fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_path)
            self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)

    def clean_text(self, text):
        """Clean the extracted text."""
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"()]+', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()

    def extract_text_from_pdf(self, pdf_path):
        """Extract and clean text from a PDF file using pdfplumber."""
        extracted_text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
        except Exception as e:
            print(f"An error occurred while processing the PDF: {e}")

        return self.clean_text(extracted_text.strip())  # Clean and return the text

    def abstractive_summarization(self, text, max_length=1000, min_length=50):
        """Generate a summary using the large FLAN-T5 model."""
        inputs = self.tokenizer(
            "summarize: " + text, return_tensors="pt", max_length=1000, truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def remove_redundancy(self, text):
        """Remove redundant words/phrases from the text for cleaner display."""
        sentences = text.split(". ")
        unique_sentences = list(dict.fromkeys(sentences))  # Remove duplicate sentences
        cleaned_text = ". ".join(unique_sentences)
        return cleaned_text.strip()

    def classify_summary(self, summary):
        """Use the fine-tuned model (if available) to classify the summary."""
        if not self.fine_tuned_model:
            return "Fine-tuned model not loaded."

        inputs = self.fine_tuned_tokenizer(summary, return_tensors="pt").to(self.device)
        outputs = self.fine_tuned_model.generate(**inputs)
        answer = self.fine_tuned_tokenizer.decode(outputs[0])

        str_to_int = {'0': "An Acute - not Accepted ", '1': "A Chronic Disease - Accepted ",
                      '2': "A Mental Illness - not Accepted ", '3': "A Physical Disability - Accepted "}
        return str_to_int.get(answer[6], "Unknown")

    def save_to_excel(self, summary, file_path="C:/Users/yasmalotaibi/Desktop/SmartDoc/submission.xlsx"):
        """Save the summary and classification to an Excel file."""
        data = [{"Summary": summary, "Classification": self.classify_summary(summary)}]
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)
        print("Summary saved to Excel:", file_path)
