from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Path to the trained model
model_path = "last.pt"

# Load the custom model
model = YOLO(model_path)

# Lower the threshold for detection
threshold = 0.6

@app.get('/')
def res():
    return {'message': "Server running on PORT 127.0.0.1"}
    
@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Convert file to an OpenCV image
        contents = await image.read()
        in_memory_file = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1280, 1280))

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to load image")

        results = model(img)[0]

        detection_made = len(results.boxes.data) > 0
        bounding_boxes = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                bounding_boxes.append({"xmin": int(x1), "ymin": int(y1), "xmax": int(x2), "ymax": int(y2), "score": score})
                # Draw the bounding box and label on the image
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        message = "Logo detected" if detection_made else "Logo not detected"
        return JSONResponse(content={"status": detection_made, "message": message, "bounding_boxes": bounding_boxes})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
