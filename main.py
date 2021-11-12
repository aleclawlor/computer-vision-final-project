import cv2
from cv_utilities import get_bounding_boxes
from flask import Flask, render_template, Response


app = Flask(__name__)

vid = cv2.VideoCapture(0)

# a function to continuously run camera loop and get frames
def run_camera_loop():

    # dummy = cv2.imread('bird.jpg')

    while True:

        ret, frame = vid.read()

        # break if nothing is returned from video feed
        if not ret:
            break 

        bounding_boxes = get_bounding_boxes(frame)
        
        ret, buffer = cv2.imencode('.jpg', bounding_boxes)
        frame = buffer.tobytes()

        # concat frame one by one and show result
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

    #     cv2.imshow('frame', frame)
      
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # vid.release()
    # cv2.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    return Response(run_camera_loop(), mimetype='multipart/x-mixed-replace; boundary=frame')

# run the main function in root directory
@app.route("/")
def index():
    return render_template('index.html')

# run the flask app
if __name__ == '__main__':
    app.run(debug=False)