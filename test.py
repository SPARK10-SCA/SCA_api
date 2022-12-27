from flask import Flask, request, Response
from flask_cors import CORS
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)
CORS(app)

@app.route('/api/test', methods=['POST'])
def react_to_flask():
    response = Response()

    if request.method == 'POST':
        if request.files:
            imgData = request.files.get('img')
            origImage = Image.open(imgData)
            # origImage = origImage.resize((256, 256))

            origImage.save('output.png')
    
        return ("true")


if __name__ == '__main__':
    app.run(debug=True)