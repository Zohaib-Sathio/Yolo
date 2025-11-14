from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64

# Fix PyTorch 2.6+ serialization issue with YOLO models
# Patch torch.load to allow loading YOLO models (which contain custom classes)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # If weights_only is not explicitly set, default to False for YOLO compatibility
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Try to add safe globals (PyTorch 2.6+ recommended approach)
try:
    import ultralytics.nn.tasks
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
except (ImportError, AttributeError):
    pass

from ultralytics import YOLO

app = FastAPI(title="YOLO Image Detection")

# Load YOLO model (using extra large version - highest accuracy)
model = YOLO('yolov8n.pt')  # extra large version for highest accuracy

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the homepage"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image and get YOLO predictions
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run YOLO prediction
        results = model(img)
        
        # Draw predictions on image with bounding boxes and labels
        annotated_img = results[0].plot()
        
        # Convert annotated image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence * 100, 2),  # Convert to percentage
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })
        
        return JSONResponse({
            "success": True,
            "detections": detections,
            "total_detections": len(detections),
            "annotated_image": f"data:image/jpeg;base64,{img_base64}"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict_with_image")
async def predict_with_image(file: UploadFile = File(...)):
    """
    Upload an image, get predictions, and return annotated image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run YOLO prediction with visualization
        results = model(img)
        
        # Draw predictions on image
        annotated_img = results[0].plot()
        
        # Convert to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence * 100, 2)
                })
        
        return JSONResponse({
            "success": True,
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "total_detections": len(detections)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict_frame")
async def predict_frame(file: UploadFile = File(...)):
    """
    Process a single frame from camera and return annotated frame with detections
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run YOLO prediction
        results = model(img)
        
        # Draw predictions on image
        annotated_img = results[0].plot()
        
        # Convert to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence * 100, 2)
                })
        
        return JSONResponse({
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "detections": detections,
            "total_detections": len(detections)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

