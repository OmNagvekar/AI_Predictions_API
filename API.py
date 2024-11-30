from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse, FileResponse
import uuid
import imghdr
import uvicorn
import os
import cv2
import numpy as np
import base64
from garbage import Predictions
from fastapi import BackgroundTasks
from garbage_video import PredictionsVideo
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
from collections import defaultdict

# FastAPI app initialization
app = FastAPI()

# Directory for uploaded files
UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# CCTV stream URL
CCTV_STREAM_URL = '/home/chinu_tensor/Downloads/action3.mp4'

# Thread pool for concurrent frame processing
executor = ThreadPoolExecutor(max_workers=2)

def process_predictions(frame):
    """
    Process a single frame for predictions. This function runs predictions on the given frame 
    and returns prediction results including intensity, type, litter, and a processed image.
    """
    prediction = Predictions(file=None, stream=True)
    prediction.image = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
    results = prediction.predict_all()

    # Extract prediction data and ensure serialization
    intensity_data = results['intensity']
    type_data = results['type']
    litter_data = results['litter']
    
    # Prepare the response data
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
async def process_image(background_tasks: BackgroundTasks,file: UploadFile = File(...)):
    """
    Endpoint to process an image uploaded by the user. 
    The image is processed for predictions, and the results are returned.
    """
    # Check if uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    # Read and validate file content
    file_content = await file.read()
    file_type = imghdr.what(None, h=file_content)
    if file_type not in ["jpeg", "png", "bmp", "tiff", "jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    try:
        # Initialize predictions and process the image
        prediction = Predictions(file_path)
        results = prediction.predict_all()

        # Extract and prepare data for the response
        type_percentages = results['type'][3]
        intensity_percentages = results['intensity'][3]
        garbage_percentage = results['intensity'][2]

        # Process image for Base64 encoding and store the image
        processed_image = results['intensity'][0]
        cv2.imwrite('./upload/processed_image.jpg',processed_image)
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_bytes = base64.b64encode(buffer).decode('utf-8')

        # Prepare the response data
        response_data = {
            "type_percentages": type_percentages,
            "intensity_percentages": intensity_percentages,
            "garbage_percentage": garbage_percentage,
            "processed_image": processed_image_bytes,
            "processed_image_download_URL":"http://127.0.0.1:8000/upload/processed_image.jpg"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
    background_tasks.add(cleanup_temp_files,[file_path,"./upload/processed_image.jpg"])
    return JSONResponse(content=response_data)

@app.post('/process-video')
async def process_video(file: UploadFile = File(...)):
    """
    Endpoint to process a video uploaded by the user. The video is processed frame-by-frame for predictions,
    and results are returned.
    """
    # Check if uploaded file is a video
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    
    # Read and save the video file
    file_content = await file.read()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    model = Predictions(file_path, True)
    
    try:
        # Process the video and get the results
        processed_frames, results, avg_garbage_percentage, type_sum = model.predict_over_video()
        len_frames = len(processed_frames)
        
        # Check if video frames were processed
        if len_frames == 0:
            raise HTTPException(status_code=500, detail="No frames processed in the video.")

        # Write processed frames to a new video file
        write_video(file_path, processed_frames, UPLOAD_DIR, "processed")

        # Initialize counters for type and litter
        total_percentages = defaultdict(float)
        counts = defaultdict(int)
        
        # Calculate average percentages for object types
        for instance in type_sum:
            for obj, percentage in instance.items():
                total_percentages[obj] += percentage
                counts[obj] += 1
        
        # Compute averages
        average_percentages = {obj: total / counts[obj] for obj, total in total_percentages.items()}

        # Count object frequencies
        object_frequencies = defaultdict(int)
        for instance in type_sum:
            for obj in instance.keys():
                object_frequencies[obj] += 1
        
        # Sort object frequencies
        sorted_frequencies = dict(sorted(object_frequencies.items(), key=lambda x: x[1], reverse=True))
        
        # Prepare the response data
        response_data = {
            "intensity_results": results['intensity'][3],
            "type_results": average_percentages,
            "litter_results": results['litter'][3],
            "frequency": sorted_frequencies,
            "garbage_percentage": str(sum(float(value) for value in avg_garbage_percentage) / len(processed_frames) 
                                     if len(processed_frames) > 0 else 0),
            "processed_video_URL": f'http://127.0.0.1:8000/upload/processed_processed.mp4',
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the video: {str(e)}")

    return JSONResponse(content=response_data)

@app.get("/upload/{filename}")
async def get_processed_video(background_tasks: BackgroundTasks, filename: str):
    """
    Endpoint to retrieve a processed video by filename.
    The video is sent to the client and cleanup of temporary files is scheduled.
    """
    video_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(video_path):
        background_tasks.add_task(cleanup_temp_files, [video_path])
        return FileResponse(video_path, media_type="video/mp4", filename=filename)
    raise HTTPException(status_code=404, detail="Processed video not found")

def cleanup_temp_files(paths: list):
    """
    Cleanup temporary files after some delay to ensure they are not needed.
    """
    time.sleep(600)
    try:
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted File {path} successfully")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error deleting the files: {str(e)}")

@app.websocket("/ws")
async def stream_cctv(websocket: WebSocket):
    """
    WebSocket stream for real-time CCTV video processing. Frames are processed
    and sent to the client in real-time.
    """
    await websocket.accept()

    # Open CCTV video stream
    video_capture = cv2.VideoCapture(CCTV_STREAM_URL)
    if not video_capture.isOpened():
        await websocket.close()
        print("Unable to access CCTV stream.")
        return

    frame_skip = 10  # Skip frames for performance
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
            await asyncio.sleep(0.01)  # Delay to achieve approx. 20 FPS
    except Exception as e:
        print(f"Error in WebSocket stream: {e}")
    finally:
        video_capture.release()
        await websocket.close()

def write_video(file_path, processed_frame_list, output_dir, file_name):
    """
    Writes processed frames to a new video file.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return

    output_path = f'./{output_dir}/processed_{file_name}.mp4'
    print(f"Output Path: {output_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = 640
    frame_height = 640

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Write each processed frame to the output video file
    for frame in processed_frame_list:
        out.write(frame)

    out.release()
    cap.release()

if __name__ == '__main__':
    uvicorn.run('API:app', host='localhost', port=8000, reload=True)
