import os
import numpy as np
import cv2
import onnxruntime as ort
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You might want to restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ONNX model
onnx_model_path = os.path.join(os.getcwd(), "best.onnx")
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"Model file not found at: {onnx_model_path}")
session = ort.InferenceSession(onnx_model_path)

# Define thresholds
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
DETECTION_THRESHOLD = 20  # Spillage area threshold in percent
ALERT_COUNT_THRESHOLD = 10  # Number of times threshold is exceeded
detection_counter = 0
alert_triggered = False

# Classes for the model
CLASSES = [
    'Aluminium foil', 'Bottle cap', 'Bottle', 'Broken glass', 'Can', 'Carton',
    'Cigarette', 'Cup', 'Lid', 'Other litter', 'Other plastic', 'Paper',
    'Plastic bag - wrapper', 'Plastic container', 'Pop tab', 'Straw',
    'Styrofoam piece', 'Unlabeled litter'
]

# Function to perform non-maximum suppression (NMS)
def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

# Function to calculate spillage area percentage
def calculate_spillage_area(bboxes, frame_size):
    frame_area = frame_size[0] * frame_size[1]
    trash_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes)
    return (trash_area / frame_area) * 100

# Function to process frame through ONNX model and get bounding boxes
def detect_trash(frame):
    input_size = (640, 640)
    img = cv2.resize(frame, input_size).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # CHW format with batch dim
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})[0].squeeze()

    bboxes, confidences, class_ids, detected_classes = [], [], [], []
    for detection in output:
        x_center, y_center, width, height = detection[:4]
        confidence = detection[4]
        class_scores = detection[5:]
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = np.argmax(class_scores)
            if class_scores[class_id] > CONFIDENCE_THRESHOLD:
                x1 = int((x_center - width / 2) * frame.shape[1])
                y1 = int((y_center - height / 2) * frame.shape[0])
                x2 = int((x_center + width / 2) * frame.shape[1])
                y2 = int((y_center + height / 2) * frame.shape[0])
                bboxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(class_id)
                detected_classes.append(CLASSES[class_id])

    indices = non_max_suppression(bboxes, confidences, NMS_THRESHOLD)
    return ([bboxes[i] for i in indices], [confidences[i] for i in indices],
            [class_ids[i] for i in indices], [detected_classes[i] for i in indices])

# Function to draw bounding boxes on the frame
def draw_boxes(frame, bboxes, class_ids, confidences):
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{CLASSES[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# WebSocket route with alert system
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global detection_counter, alert_triggered

    await websocket.accept()
    logger.info("WebSocket connected")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Unable to open webcam.")
        await websocket.send_text("Error: Unable to open webcam.")
        await websocket.close()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, confidences, class_ids, detected_classes = detect_trash(frame)
        spillage_percentage = calculate_spillage_area(bboxes, frame.shape[:2])
        if spillage_percentage > DETECTION_THRESHOLD:
            detection_counter += 1
        else:
            detection_counter = 0

        if detection_counter >= ALERT_COUNT_THRESHOLD and not alert_triggered:
            alert_triggered = True
            await websocket.send_text("Alert: Trash spillage exceeded threshold!")

        draw_boxes(frame, bboxes, class_ids, confidences)
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        await websocket.send_bytes(buffer.tobytes())
        await websocket.send_text(f"Detected: {', '.join(detected_classes)}")
        await asyncio.sleep(0.03)

    cap.release()
    logger.info("WebSocket closed")

# Serve a styled HTML page
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_content = """
    <html>
        <head>
            <title>Trash Detection</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; background-color: #f4f4f4; }
                h1 { color: #333; }
                img { border: 2px solid #ddd; border-radius: 8px; margin-top: 20px; }
                #detected-classes { font-size: 18px; margin-top: 10px; }
                .alert { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Real-Time Trash Detection</h1>
            <img id="video-feed" src="" alt="Webcam feed" width="640" height="640">
            <p id="detected-classes">Detected: None</p>
            <script>
                const videoFeed = document.getElementById('video-feed');
                const detectedClasses = document.getElementById('detected-classes');
                const ws = new WebSocket(`wss://${window.location.host}/ws`);
                ws.onmessage = event => {
                    if (typeof event.data === 'string') {
                        if (event.data.startsWith("Alert")) {
                            alert(event.data);
                        } else {
                            detectedClasses.textContent = event.data;
                        }
                    } else {
                        const blob = new Blob([event.data], { type: 'image/jpeg' });
                        videoFeed.src = URL.createObjectURL(blob);
                    }
                };
                ws.onerror = () => alert("WebSocket connection error.");
                ws.onclose = () => alert("WebSocket connection closed.");
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Main execution
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render uses dynamic port
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
