from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
from model import load_model_and_generate_caption

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Paths to model and vocabulary
MODEL_PATH = 'best_tested_model.pt'
VOCAB_PATH = 'vocabulary.json'

@app.route('/')
def index():
    return render_template('index.html', caption=None, image_path=None, error=None)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('index', error="No image uploaded."))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index', error="No image selected."))

    if file:
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        try:
            # Generate caption using the model
            caption = load_model_and_generate_caption(image_path, MODEL_PATH, VOCAB_PATH)
            return render_template('index.html', caption=caption, image_path=image_path, error=None)
        except Exception as e:
            return render_template('index.html', caption=None, image_path=image_path, error=f"Error processing image: {str(e)}")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api')
def api():
    return render_template('api.html')

if __name__ == '__main__':
    # Only run the development server locally
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))