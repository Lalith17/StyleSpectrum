from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to get the dominant color of an image using K-means
def get_dominant_color(image_path, k=1):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img.width // 5, img.height // 5))  # Resize to speed up processing
    img = np.array(img)

    pixels = img.reshape(-1, 3)  # Flatten the image to a 2D array of RGB values
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    return kmeans.cluster_centers_[0].astype(int)

def clear_upload_folder():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove a directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Function to match the shirt color with compatible pants colors
def match_color(color):
    color_rules = {
        "White": ["Black", "Navy", "Denim", "Grey", "Beige"],
        "Blue": ["Beige", "Grey", "Black", "Denim", "White"],
        "Red": ["Black", "Beige", "White", "Navy"],
        "Black": ["Grey", "Beige", "Dark", "White"],
        "Green": ["Beige", "Grey", "Neutral", "White"],
        "Yellow": ["Grey", "Black", "Beige", "Dark Blue"],
        "Orange": ["Grey", "Black", "Beige", "White"],
        "Purple": ["Black", "Grey", "Beige", "Denim"],
        "Brown": ["Black", "White", "Beige", "Denim"],
        "Pink": ["Grey", "Black", "White", "Denim"],
        "Grey": ["Black", "White", "Beige", "Denim"]
    }
    color_str = find_closest_color(color)
    return color_rules.get(color_str, [])

# Function to identify the closest color name based on RGB values
def find_closest_color(rgb):
    color_names = {
        "White": np.array([255, 255, 255]),
        "Blue": np.array([0, 0, 255]),
        "Red": np.array([255, 0, 0]),
        "Black": np.array([0, 0, 0]),
        "Green": np.array([0, 255, 0]),
        "Yellow": np.array([255, 255, 0]),
        "Orange": np.array([255, 165, 0]),
        "Purple": np.array([128, 0, 128]),
        "Brown": np.array([139, 69, 19]),
        "Pink": np.array([255, 192, 203]),
        "Grey": np.array([169, 169, 169])
    }

    min_distance = float('inf')
    closest_color = ""
    for color, color_value in color_names.items():
        distance = np.linalg.norm(rgb - color_value)
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

# Function to recommend the best matching clothing item
def recommend_clothing(image_path, options_paths):
    dominant_color = get_dominant_color(image_path)
    matching_pants_colors = match_color(dominant_color)

    for option_path in options_paths:
        option_color = get_dominant_color(option_path)
        pant_color = find_closest_color(option_color)

        if pant_color in matching_pants_colors:
            return option_path
    return options_paths[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    clear_upload_folder()
    uploaded_files = request.files.getlist('options')  # Get multiple files
    base_image = request.files['base_image']          # Get the shirt image

    filenames = []

    # Save base image (shirt)
    if base_image:
        base_filename = os.path.join(app.config['UPLOAD_FOLDER'], base_image.filename)
        base_image.save(base_filename)

    # Save all pants images
    for file in uploaded_files:
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filenames.append(filepath)
    suggested=recommend_clothing(base_filename, filenames)

    return render_template('result.html',input=base_filename, filename=suggested)

if __name__ == '__main__':
    app.run(debug=True)