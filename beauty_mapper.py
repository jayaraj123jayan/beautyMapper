
#!pip install dlib
#!pip install opencv-python
#!pip install webcolors
#!pip install flask
#!pip install ngrok

import dlib
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import webcolors
import json
from types import SimpleNamespace
import texture as texture


from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests

predictor_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.isfile(predictor_path):
    
    print("Downloading dilb model...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    response = requests.get(url, stream=True)
    with open("shape_predictor_68_face_landmarks.dat.bz2", "wb") as f:
        f.write(response.content)

    # Extract the bz2 file
    print("Extracting model ...")
    import bz2
    with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2") as fr, open(predictor_path, "wb") as fw:
        fw.write(fr.read())

landmark_predictor = dlib.shape_predictor(predictor_path)

def delectFaces(img):
  face_detector = dlib.get_frontal_face_detector()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = face_detector(gray)
  return faces,gray

def extract_region(image, points):
    mask = np.zeros_like(image)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], (255, 255, 255))
    result = cv2.bitwise_and(image, mask)
    rect = cv2.boundingRect(points)
    cropped = result[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    return cropped

def get_dominant_color(image, k=1):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Flatten the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Remove black pixels (assuming black is [0, 0, 0])
    pixels = pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    
    # Check if pixels array is empty after filtering
    if len(pixels) == 0:
        return [0, 0, 0]  # Return black or a default color if no valid pixels

    # Apply KMeans clustering
    clt = KMeans(n_clusters=k, n_init=10)
    clt.fit(pixels)

    # Get the centroid of the cluster with the most pixels
    unique, counts = np.unique(clt.labels_, return_counts=True)
    most_common_cluster = unique[np.argmax(counts)]
    dominant_color = clt.cluster_centers_[most_common_cluster].astype(int)

    return dominant_color

def display_clusters(image, labels, centers):
    segmented_image = centers[labels].reshape(image.shape)
    cv2.imshow("Segmented Image", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
def extractLips(image):
    faces,gray = delectFaces(image)
    face = faces[0]
    landmarks = landmark_predictor(gray, face)
    landmarks_points = [(p.x, p.y) for p in landmarks.parts()]
    # Define points for lips (48-67)
    lips_points = landmarks_points[48:67]
    return extract_region(image, lips_points)

def delectProducts(faces,gray,image):
  if len(faces) !=1 :
      return None
  

  for face in faces:
    landmarks = landmark_predictor(gray, face)
    landmarks_points = [(p.x, p.y) for p in landmarks.parts()]

    left_eye_points = landmarks_points[36:42]
    left_eye_color = get_dominant_color(extract_region(image,left_eye_points))

# Define points for right eye (42-47)
    right_eye_points = landmarks_points[42:48]
    right_eye_color = get_dominant_color(extract_region(image,right_eye_points))

# Define points for left cheek (approximate using points 1, 2, 3, 4, 31)
    left_cheek_points = [landmarks_points[i] for i in [1, 2, 3, 4, 31]]
    left_cheek_color = get_dominant_color(extract_region(image,left_cheek_points),)

# Define points for right cheek (approximate using points 15, 14, 13, 12, 35)
    right_cheek_points = [landmarks_points[i] for i in [15, 14, 13, 12, 35]]
    right_cheek_color = get_dominant_color(extract_region(image,right_cheek_points))

    # Define points for lips (48-67)
    lips_points = landmarks_points[48:67]
    lips_image = extract_region(image, lips_points)
    print('getting lipstick texture')
    lipTexture = texture.predict_texture(lips_image)
    # lipTexture = ""
    print('lipstick texture' + lipTexture)

    # Get the predominant color of the lips
    lipstick_color = get_dominant_color(lips_image)
    # color_name = closest_color(dominant_color)
    # print("Lipstick color:" getRgbStr(lipstick_color))
    lipstickColorName = getcolor(getRgbStr(lipstick_color))
    foundationColorName = getcolor(getRgbStr(right_cheek_color))
    lenseColorName = getcolor(getRgbStr(right_eye_color))
    products = 'lipstick - color' + lipstickColorName+ 'texture' +lipTexture
    products+= 'cheek foundation - color' + foundationColorName
    products+= 'eye lesne - color' + lenseColorName
    seph ="https://www.sephora.com/search?keyword="
    itemLinks = [
        {
            "name": "lipstick",
            "link" : seph+ "lipstick"+ lipstickColorName  +lipTexture
            
        },
        {
            "name": "Foundation",
            "link" : seph+ "Foundation"+ foundationColorName
        },
        {
            "name": "Lense",
            "link" : seph+ "EyeLens"+ lenseColorName
        }
    ]
    return products ,itemLinks

def getRgbStr (color) :
    color = color.tolist()
    return "rgb ("+ str(color[0]) +','+ str(color[1]) +','+ str(color[2]) +")"

# !set NGROK_AUTHTOKEN=2KRH4rvDLeAgu8wjsLZCIO6bo89_6zu9ewBupf1ZtSrAY13TC

app = Flask(__name__)

def resize_image_aspect_ratio(image, target_width):
    # Get the original dimensions
    (h, w) = image.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = w / h
    
    # Calculate the new dimensions
    new_width = target_width
    new_height = int(new_width / aspect_ratio)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return resized_image


@app.route('/upload', methods=['POST'])
def predict_image():
    print("received image")
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
    
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        
        # Save the file
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        print("saved image")
        img = resize_image_aspect_ratio(cv2.imread("uploads/" + filename),400)
        print("resized")
        faces,gray = delectFaces(img)
        print("face detected")
        # return delectLips(faces,gray,img).tolist()
        #return jsonify(result)
        products, itms = delectProducts(faces,gray,img);
        print("products detected")
        return genInstGoogle(products, itms)
apiKey = 'AIzaSyCakukvTpn88bVTGJhD_SNplALWNq-C6Uc'
def genInstGoogle(text, itms):
    text = "Please provide instructions(in less than 128 words) to apply following makeup products to my face. Start with the peparation of skin. ignore the rgb color . products are "+ text 
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=' + apiKey
    data='{"contents":[{"parts":[{"text": "'+text+'"}]}]}'
    # print(data)
    response = requests.post(url,data=data, stream=True)
    inst = jsonToObj(response.content.decode("utf-8")).candidates[0].content.parts[0].text

    return {
        "instructions": inst,
        "products": itms
    }
def getcolor(rgb):
    text = "decode this rgb code to closest color name (respond woth the color only)"+ rgb 
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=' + apiKey
    data='{"contents":[{"parts":[{"text": "'+text+'"}]}]}'
    # print(data)
    response = requests.post(url,data=data, stream=True)
    # print(jsonToObj(response.content.decode("utf-8")))
    return jsonToObj(response.content.decode("utf-8")).candidates[0].content.parts[0].text
def jsonToObj(str) :
    return json.loads(str, object_hook=lambda d: SimpleNamespace(**d))

@app.route('/instuctions', methods=['GET'])
def getInstructions():
    return genInstGoogle(["lipstick"])

@app.route('/')
def index():
    return render_template('index.html') 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

