from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize the sentence transformer model.
model = SentenceTransformer('all-MiniLM-L6-v1')

# Global variables to hold the uploaded CSV data and precomputed embeddings.
data_df = None
embeddings = None

@app.route('/upload', methods=['POST'])
def upload_csv():
    global data_df, embeddings
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        data_df = pd.read_csv(file)
        # Ensure that the CSV has the required columns.
        if not {'Id', 'description'}.issubset(data_df.columns):
            return jsonify({'error': 'CSV must contain Id and description columns'}), 400
        # Compute embeddings for each description.
        embeddings = model.encode(data_df['description'].tolist())
        return jsonify({'message': 'File uploaded and processed successfully.'})
    except Exception as e:
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
    best_index = np.argmax(sims)
    best_row = data_df.iloc[best_index]
    return jsonify({
        'Id': best_row['Id'],
        'description': best_row['description'],
        'score': float(sims[best_index])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
