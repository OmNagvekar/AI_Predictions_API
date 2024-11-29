from fastapi import FastAPI, File, UploadFile, HTTPException,WebSocket
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
from concurrent.futures import ThreadPoolExecutor
import asyncio
app = FastAPI()
UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)
CCTV_STREAM_URL = '/home/chinu_tensor/Downloads/action3.mp4'

executor = ThreadPoolExecutor(max_workers=2)



def process_predictions(frame):
    """Process a single frame for predictions."""
    prediction = Predictions(file=None,stream=True)
    prediction.image = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
    results = prediction.predict_all()

    # Extract prediction data and ensure serialization
    intensity_data = results['intensity']
    type_data = results['type']
    litter_data = results['litter']
    print('this is litter',type_data[3])
    # Ensure all outputs are JSON-serializable
    response_data = {
        "intensity": {
            "garbage_percentage": float(intensity_data[2]),
            "object_percentages": [float(val) for val in intensity_data[3].values()],
        },
        "type": {
            "object_percentages": [float(val) for val in type_data[3].values()],
        },
        "litter": {
            "object_percentages": [float(val) for val in litter_data[3].values()],
        },
    }

    # Encode processed image to Base64
    _, buffer = cv2.imencode('.jpg', intensity_data[0])
    response_data["processed_image"] = base64.b64encode(buffer).decode('utf-8')

    return response_data
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
    print(file_path)
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
    model = Predictions(file_path,True)
    # Initialize predictions class and process video
    try:
        processed_frames ,_,_,_= model.predict_over_video()
        write_video(file_path,processed_frames,UPLOAD_DIR,"processd") 
        response_data = {'satus':'succeed'}

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


@app.websocket("/ws")
async def stream_cctv(websocket: WebSocket):
    await websocket.accept()

    video_capture = cv2.VideoCapture(CCTV_STREAM_URL)
    if not video_capture.isOpened():
        await websocket.close()
        print("Unable to access CCTV stream.")
        return

    frame_skip = 10
    frame_count = 0

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to read frame from CCTV.")
                break

            if frame_count % frame_skip == 0:
                loop = asyncio.get_event_loop()
                response_data = await loop.run_in_executor(executor, process_predictions, frame)

                # Send processed data to WebSocket client
                await websocket.send_json(response_data)

            frame_count += 1
            await asyncio.sleep(0.01)  # 20 FPS
    except Exception as e:
        print(f"Error in WebSocket stream: {e}")
    finally:
        video_capture.release()
        await websocket.close()

def write_video(file_path, processed_frame_list, output_dir, file_name):
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return 
    output_path = f'./{output_dir}/processed_{file_name}.mp4'
    print(f"Output Path: {output_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = 640
    frame_height = 640
    
    # Setup VideoWriter
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    # Read and write processed frames
    for frame in processed_frame_list:
        output_video.write(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    
    # Release resources
    cap.release()
    output_video.release()
    
    print(f"Video written successfully to {output_path}")
if __name__ == '__main__':
    uvicorn.run('API:app', host='localhost', port=8000, reload=True)
