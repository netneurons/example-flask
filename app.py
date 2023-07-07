from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

app = Flask(__name__)

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Please upload two images.'})

        image1 = request.files['image1']
        image2 = request.files['image2']

        similarity = calculate_similarity(image1, image2)

        return jsonify({'similarity': similarity})

    # Render the HTML form
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image Similarity Comparison</title>
        </head>
        <body>
            <h1>Image Similarity Comparison</h1>
            <form action="/" method="post" enctype="multipart/form-data">
                <label for="image1">Image 1:</label>
                <input type="file" id="image1" name="image1" accept="image/*"><br><br>
                <label for="image2">Image 2:</label>
                <input type="file" id="image2" name="image2" accept="image/*"><br><br>
                <input type="submit" value="Compare Images">
            </form>
        </body>
        </html>
    ''')

def calculate_similarity(image1_path, image2_path):
    img1 = image.load_img(image1_path, target_size=(224, 224))
    img2 = image.load_img(image2_path, target_size=(224, 224))

    x1 = image.img_to_array(img1)
    x2 = image.img_to_array(img2)

    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)

    x1 = preprocess_input(x1)
    x2 = preprocess_input(x2)

    feature1 = model.predict(x1)
    feature2 = model.predict(x2)

    similarity = np.sum((feature1 - feature2) ** 2)

    return similarity

if __name__ == '__main__':
    app.run()
