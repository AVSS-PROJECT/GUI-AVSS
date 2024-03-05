from flask import Flask, render_template, Response
import cv2
import torch

app = Flask(__name__)

frame_width = 1280
frame_height = 720

# Load the YOLOv8 model
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)
model = model.autoshape()

def pothole_detection(frame):
    # Perform YOLOv8 object detection
    results = model(frame)

    # Extract bounding boxes and scores
    boxes = results.xyxy[0].cpu().numpy()
    scores = results.xyxy[0][:, 4].cpu().numpy()

    # Filter high confidence pothole detections
    high_conf_detections = [
        (box[:4].astype(int), score)
        for box, score in zip(boxes, scores)
        if score > 0.7  # adjust confidence threshold as needed
    ]

    print("High confidence detections:", high_conf_detections)

    # Annotate the frame with bounding boxes
    for (x_min, y_min, x_max, y_max), score in high_conf_detections:
        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        frame = cv2.putText(frame, f"Score: {score:.2f}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform pothole detection
        frame = pothole_detection(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
