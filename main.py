import cv2
import random
from cv_utilities import get_bounding_boxes, blur_background
from flask import Flask, render_template, Response
import time 
import torch
import torchvision.models as models
import numpy as np

from torch import nn
from torchvision import transforms
from PIL import Image

CLASSES = [    
    "glass",
    "paper",
    "cardboard",
    "plastic",
    "metal",
    "trash"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

app = Flask(__name__)

vid = cv2.VideoCapture(0)
marginHorizontal = 160
marginVertical = 100

model = models.resnet18()
model.fc = nn.Linear(512,6)
model.load_state_dict(torch.load("model.pth"))
model.eval()

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

def getObject(frame):
    
    global model, CLASSES, data_transforms

    with torch.no_grad():

        frame = Image.fromarray(frame)
        frame = data_transforms(frame)
        frame = frame.float()
        pred = model(frame.unsqueeze(0))
        print("Prediction: ", pred)
        predicted = CLASSES[pred[0].argmax(0)]
        return predicted



# a function to continuously run camera loop and get frames
def run_camera_loop():

    global vid, marginHorizontal, marginVertical

    # get video feed "forever"
    while True:

        try:
            
            time.sleep(.05)
            ret, frame = vid.read()

            # break if nothing is returned from video feed
            if not ret:
                break  
            
            # Previous approach: get bounding boxes of frame
            # this was deemed inneffective, since there was a lot of noise in live frames
            # bounding_boxes = get_bounding_boxes(frame)
            
            color = (255, 0, 0)

            # define coordinates based on frame size 
            width, height = frame.shape[1], frame.shape[0]

            x1, y1 = marginHorizontal, marginVertical
            x2, y2 = width - marginHorizontal, height - marginVertical

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            frame_crop=frame[y1:y2, x1:x2]
            frame_crop= blur_background(frame_crop)

            object = getObject(frame_crop)
            cv2.putText(frame, object, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            # concat frame one by one and show result
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

        except:
            continue

            
# a function that returns matplotlib figure showing preprocessed frames
def return_preprocessed_frames():

    global vid, marginHorizontal, marginVertical

     # get video feed "forever"
    while True:
        try:
            time.sleep(.1)
            ret, frame = vid.read()

            # break if nothing is returned from video feed
            if not ret:
                break

            # define coordinates based on frame size 
            width, height = frame.shape[1], frame.shape[0]

            x1, y1 = marginHorizontal, marginVertical
            x2, y2 = width - marginHorizontal, height - marginVertical

            frame=frame[y1:y2, x1:x2]
            frame= blur_background(frame)

            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            # concat frame one by one and show result
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

        except: 
            continue


@app.route('/video_feed')
def video_feed():
    return Response(run_camera_loop(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plot_preprocessed_image')
def plot_preprocessed_image():
    return Response(return_preprocessed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# run the main function in root directory
@app.route("/")
def index():
    return render_template('index.html')

# run the flask app
if __name__ == '__main__':
    app.run(debug=True)