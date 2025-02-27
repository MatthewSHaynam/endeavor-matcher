from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize the sentence transformer model.
model = SentenceTransformer('all-MiniLM-L6-v1')

# Global variables to hold the uploaded CSV data and precomputed embeddings.
data_df = None
embeddings = None

@app.route('/')
def index():
    return "Backend is running!", 200

@app.route('/upload', methods=['POST'])
def upload_csv():
    # Debug: Print the received keys for both files and form data
    print("Request files keys:", list(request.files.keys()))
    print("Request form data:", request.form)
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        print("CSV Columns received:", list(df.columns))
        
        # Ensure that the CSV has the required columns.
        if not {'Id', 'description'}.issubset(df.columns):
            return jsonify({'error': 'CSV must contain Id and description columns'}), 400
        
        # Compute embeddings for each description.
        global data_df, embeddings
        data_df = df.copy()
        embeddings = model.encode(data_df['description'].tolist())
        return jsonify({'message': 'File uploaded and processed successfully.'})
    except Exception as e:
        print("Error processing CSV:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/match', methods=['POST'])
def find_match():
    global data_df, embeddings
    if data_df is None or embeddings is None:
        return jsonify({'error': 'No data uploaded yet.'}), 400
    data = request.get_json()
    if not data or 'input' not in data:
        return jsonify({'error': 'Please provide input text.'}), 400
    input_text = data['input']
    input_embedding = model.encode([input_text])
    # Compute cosine similarities between the input and all CSV descriptions.
    sims = cosine_similarity(input_embedding, embeddings)[0]
    best_index = int(np.argmax(sims))
    best_row = data_df.iloc[best_index]
    return jsonify({
        'Id': int(best_row['Id']),         # Convert numpy.int64 to native int
        'description': best_row['description'],
        'score': float(sims[best_index])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
