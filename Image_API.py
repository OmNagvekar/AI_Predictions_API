from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import imghdr
import uvicorn
import os
import cv2
import numpy as np
from garbage import Predictions

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

        # Merge bounding boxes from all models
        processed_image = results['intensity'][0] # Use intensity image as base
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
    
    # Encode processed image to bytes
    _, buffer = cv2.imencode('.jpg', processed_image)
    processed_image_bytes = buffer.tobytes()

    # Return the processed image
    return StreamingResponse(
        iter([processed_image_bytes]),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=processed_{file.filename}"}
    )

if __name__ == '__main__':
    uvicorn.run('Image_API:app', host='localhost', port=8000, reload=True)
