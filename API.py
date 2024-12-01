from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse, FileResponse
import uuid
import imghdr
import uvicorn
import os
import cv2
import base64
from garbage import Predictions
from fastapi import BackgroundTasks
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

# FastAPI app initialization
app = FastAPI()

# Directory for uploaded files
UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# CCTV stream URL
CCTV_STREAM_URL = '/home/chinu_tensor/Downloads/action3.mp4'

# Thread pool for concurrent frame processing
executor = ThreadPoolExecutor(max_workers=2)

#### load florence model ########
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
if device =="cuda:0":
    desc_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True, cache_dir='./').to(device)
    desc_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True,cache_dir='./')
else:
    #workaround for unnecessary flash_attn requirement
    from unittest.mock import patch
    from transformers.dynamic_module_utils import get_imports
    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
        desc_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base",attn_implementation="sdpa" ,torch_dtype=torch_dtype, trust_remote_code=True, cache_dir='./').to(device)
        desc_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", attn_implementation="sdpa",trust_remote_code=True,cache_dir='./')


def run_example(task_prompt,image, text_input=None):
    """Visual LM to generate detailed description about image which on later can be used for to generate report or any other causes"""
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = desc_processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = desc_model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = desc_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = desc_processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer[task_prompt]

def process_predictions(frame):
    """
    Process a single frame for predictions. This function runs predictions on the given frame 
    and returns prediction results including intensity, type, litter, and a processed image.
    """
    prediction = Predictions(file=None, stream=True)
    prediction.image = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
    results = prediction.predict_all()
    deep_obj = DeepSort()
    # Smoothing settings
    frame_window = 5  # Number of frames to track for smoothing
    frame_history = defaultdict(list)  # Stores predictions across frames for smoothing
    smmoth_result= process_predictions_with_tracking(frame,results,deep_obj,frame_window,frame_history)

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
        image = Image.open(file_path) #opened image in PIL for VisualLM model florence-2-base
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
        description = run_example(task_prompt='<MORE_DETAILED_CAPTION>',text_input='',image=image) # generated Detailed description about image

        # Prepare the response data
        response_data = {
            "type_percentages": type_percentages,
            "intensity_percentages": intensity_percentages,
            "garbage_percentage": garbage_percentage,
            "processed_image": processed_image_bytes,
            "Image_Description":description,
            "processed_image_download_URL":"http://127.0.0.1:8000/upload/processed_image.jpg"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
    background_tasks.add_task(cleanup_temp_files,[file_path,"./upload/processed_image.jpg"])
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
        processed_frames, results, avg_garbage_percentage, type_sum,smooth_results = model.predict_over_video()
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
    video_capture = cv2.VideoCapture(0)
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
def process_predictions_with_tracking(frame,results,deep_obj,frame_window,frame_history):
    # Use tracking to maintain object identity across frames
    result_obj_model = results['litter'][1]
    # Extract bounding box, score, and class info
    maping = result_obj_model[0].names
    tracking_obj = [(r.xywh.to('cpu').numpy().tolist()[0],r.conf.to('cpu').item(),maping[int(r.cls.to('cpu').item())]) for r in result_obj_model[0].boxes]
    trackers = deep_obj.update_tracks(tracking_obj,frame)  # Update tracker with current frame data
    smoothed_results=[]
    # Process each tracked object (littering detection)
    for track_ in trackers:
        track_id = track_.track_id # tracking id of each object in image
        x1, y1, x2, y2 = track_.to_ltrb()  # Bounding box coordinates
        class_id = track_.get_det_class() # class name
        confidence = track_.get_det_conf() # confidence score
        if class_id == 'littering' and confidence > 0.7:
            # Store the predicted class for smoothing
            frame_history[track_id].append(class_id)
            # Only keep the most recent `history_window` frames in memory
            if len(frame_history[track_id]) > frame_window:
                frame_history[track_id].pop(0)
            # Majority voting to determine the final class
            predicted_class = majority_vote(frame_history[track_id])
            if predicted_class=="littering":
                # Save the image of the person who committed the littering action
                save_person_face(frame, x1, y1, x2, y2,track_id)
            smoothed_results.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "class": predicted_class,
                "confidence": confidence
            })
    deep_obj.delete_all_tracks() # clearing cache if not causes issue in overloading

    return smoothed_results

def majority_vote(class_history):
    """
    Perform majority voting over the last few frames to smooth the classification.
    """
    vote_counts = defaultdict(int)
    for cls in class_history:
        vote_counts[cls] += 1

    # Return the class with the highest vote count
    return max(vote_counts, key=vote_counts.get)

def save_person_face(frame, x1, y1, x2, y2,track_id):
    # Crop the face region from the bounding box of the person
    person_face = frame[int(y1): int(y1)+int(y2) ,int(x1):int(x1)+int(x2)]
    # Use a face detection model (e.g., OpenCV Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(person_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (fx, fy, fw, fh) in faces:
        face_image = person_face[int(fy):int(fy)+int(fh), int(fx):int(fx)+int(fw)]
        # Save face image
        face_filename = f"face_{track_id}.jpg"
        cv2.imwrite(os.path.join(UPLOAD_DIR, face_filename), face_image)
        print(f"Saved face image as {face_filename}")


if __name__ == '__main__':
    uvicorn.run('API:app', host='localhost', port=8000, reload=True)
