# YOLO Image Detection Web Application

A web application that uses YOLO (You Only Look Once) model to detect objects in uploaded images. Built with FastAPI backend and a modern, responsive frontend.

## Features

- 🖼️ Upload images via drag-and-drop or file browser
- 🔍 Real-time object detection using YOLOv8n (nano version - smallest model)
- 📊 Display detected classes with confidence scores
- 🎨 Beautiful, modern UI with gradient design
- ⚡ Fast inference with optimized YOLO model

## Setup Instructions

### 1. Install Dependencies

Make sure you have Python 3.8+ installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

Start the FastAPI server:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload
```

The application will be available at `http://localhost:8000`

### 3. Access the Homepage

Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

1. Click "Choose Image" or drag and drop an image onto the upload area
2. Wait for the YOLO model to process the image
3. View the detected objects with their confidence scores displayed as percentage bars

## Model Information

- **Model**: YOLOv8n (nano version)
- **Size**: Smallest YOLO model for faster inference
- **Classes**: 80 COCO dataset classes (person, car, dog, etc.)
- The model will be automatically downloaded on first run

## API Endpoints

- `GET /` - Homepage with image upload interface
- `POST /predict` - Upload image and get detection results (JSON)

## Future Enhancements

- Support for larger YOLO models (YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x)
- Annotated image display with bounding boxes
- Batch image processing
- Export results as JSON/CSV

## Requirements

- Python 3.8+
- FastAPI
- Ultralytics YOLO
- OpenCV
- NumPy
- Pillow

