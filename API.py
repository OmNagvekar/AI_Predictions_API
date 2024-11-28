from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uuid
import imghdr
import uvicorn
import os
import cv2
import numpy as np
import base64
from garbage import Predictions
from garbage_video import PredictionsVideo

app = FastAPI()
UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post('/process-image')
async def process_image(file: UploadFile = File(...)):
    # Check if uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    # Read file content
    file_content = await file.read()
    
    # Validate file as an image using imghdr
    file_type = imghdr.what(None, h=file_content)
    if file_type not in ["jpeg", "png", "bmp", "tiff", "jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Initialize predictions class and process image
    try:
        prediction = Predictions(file_path)
        results = prediction.predict_all()

        # Extract object percentages from both the type and intensity models
        type_percentages = results['type'][3]  # From model_type
        intensity_percentages = results['intensity'][3]  # From model_intensity
        garbage_percentage = results['intensity'][2]  # Total garbage percentage

        # Merge bounding boxes from all models for visualization
        processed_image = results['intensity'][0]  # Use intensity image as base
        
        # Encode processed image to Base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_bytes = base64.b64encode(buffer).decode('utf-8')

        # Prepare the response
        response_data = {
            "type_percentages": type_percentages,
            "intensity_percentages": intensity_percentages,
            "garbage_percentage": garbage_percentage,
            "processed_image": processed_image_bytes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
    
    # Return JSON response with image and data
    return JSONResponse(content=response_data)


@app.post('/process-video')
async def process_video(file: UploadFile = File(...)):
    # Check if uploaded file is a video
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")

    # Read file content
    file_content = await file.read()

    # Save the video file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Initialize predictions class and process video
    try:
        prediction = PredictionsVideo(file_path)
        processed_frames, results = prediction.process_video()

        # Save processed frames as image files
        frame_urls = []
        for i, frame in enumerate(processed_frames):
            # Generate a unique filename for each frame
            frame_filename = f"{uuid.uuid4()}.jpg"
            frame_path = os.path.join(UPLOAD_DIR, frame_filename)
            
            # Save the frame as an image file
            cv2.imwrite(frame_path, frame)

            # Generate URL or file path to the saved frame
            frame_urls.append(f"/frames/{frame_filename}")

        # Prepare the response with URLs to the processed frames
        response_data = {
            "intensity_results": results['intensity'],
            "type_results": results['type'],
            "litter_results": results['litter'],
            "processed_frames": frame_urls,  # Provide URLs to the frames
        }
        if isinstance(response_data, np.ndarray):
            response_data = response_data.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the video: {str(e)}")

    # Return JSON response with URLs to the processed frames and data
    return JSONResponse(content=response_data)

# Endpoint to serve the processed frames
@app.get("/frames/{filename}")
async def get_frame(filename: str):
    frame_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(frame_path):
        return FileResponse(frame_path)
    raise HTTPException(status_code=404, detail="Frame not found")

if __name__ == '__main__':
    uvicorn.run('API:app', host='localhost', port=8000, reload=True)
