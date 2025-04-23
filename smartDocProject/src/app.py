from flask import Flask, request, jsonify, render_template, send_from_directory
from TextSummarization import TextSummarizer
import os
from PIL import Image
import pytesseract

app = Flask(__name__)

# Initialize the summarizer
summarizer = TextSummarizer(
    model_path="google/flan-t5-large",
    fine_tuned_path=r"./uploads/checkpoint-708"
    #fine_tuned_path=r"C:\Users\yasmalotaibi\Desktop\SmartDoc\results\checkpoint-102"

)

# Set the path for tesseract executable for OCR
#C:\Users\yasmalotaibi\AppData\Local\Programs\Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\yasmalotaibi\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'  # Update this to your Tesseract path

# Sample data for requests
requests = {
    1: {"id": 1, "name": "من تقارير ابشر داون  ", "type": "PDF", "date": "2025-02-16", "attachment": "/static/pdfs/reaDataDown.pdf"},
    2: {"id": 2, "name": "طلب مرض خرف ", "type": "PDF", "date": "2025-02-17", "attachment": "/static/pdfs/Mental.pdf"},
    3: {"id": 3, "name": " من تقارير ابشر صورة", "type": "JPG", "date": "2025-02-18", "attachment": "/static/pdfs/img5.jpg"},
    4: {"id": 4, "name": "ملف شيخة ي", "type": "PDF", "date": "2025-02-19", "attachment": "/static/pdfs/anemia.pdf"},
    5: {"id": 5, "name": " إصابة ما بعد الصدمة", "type": "PDF", "date": "2024-05-23", "attachment": "/static/pdfs/Patient Medical Reportph v2 .pdf"}

}

# Route to serve static PDFs and Images
@app.route('/static/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

# Home page route
@app.route('/')
def index():
    return render_template("index.html", requests=requests)

# Route for request details page and processing the file (PDF or Image)
@app.route('/request-details/<int:request_id>', methods=['GET', 'POST'])
def request_details(request_id):
    request_data = requests.get(request_id)

    if not request_data:
        return "Request not found", 404

    if request.method == 'POST':
        file_path = request.json.get('file_path')
        print(f"Received file path: {file_path}")  # Debugging the received file path

        # Construct the full path to the file (PDF or Image)
        full_file_path = os.path.join(app.root_path, file_path.strip('/'))

        print(f"Full file path: {full_file_path}")  # Log the full path for debugging

        # Check if the file exists
        if not os.path.exists(full_file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 400

        try:
            # Handle PDF or Image file
            if file_path.lower().endswith('.pdf'):
                # Extract text from PDF
                extracted_text = summarizer.extract_text_from_pdf(full_file_path)
            else:
                # Handle Image (OCR)
                image = Image.open(full_file_path)
                extracted_text = pytesseract.image_to_string(image)

            # Generate the summary and classification
            summarized_text = summarizer.abstractive_summarization(extracted_text)
            cleaned_summary = summarizer.remove_redundancy(summarized_text)
            classification = summarizer.classify_summary(summarized_text)

            # Return the summary and classification
            response = {
                "summary": cleaned_summary,
                "classification": classification,
                "original_length": len(extracted_text),
                "summary_length": len(cleaned_summary)
            }

            return jsonify(response)

        except Exception as e:
            print(f"Error during processing: {e}")
            return jsonify({"error": "Error processing file."}), 500

    return render_template("request_details.html", request=request_data)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
