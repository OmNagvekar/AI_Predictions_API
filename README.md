# FastAPI Garbage Prediction API

This FastAPI project provides endpoints for analyzing images and videos to predict garbage intensity, types, and other characteristics using advanced deep learning models like Florence-2-base and DeepSort tracker.

## Features

- Upload images to predict garbage-related characteristics and generate detailed image descriptions.
- Upload videos to process frame-by-frame predictions and analyze garbage patterns.
- Utilizes advanced machine learning models for robust predictions.
- Multi-Model Garbage Detection
  - Garbage Intensity Detection: Estimates the percentage of garbage in an  area
  * Garbage Type Classification: Categorizes different types of garbage.
  - Littering Detection: Identifies instances of littering and tracks individuals responsible.
- DeepSort Tracker Integration
  - &#x20;Real-time object tracking for seamless monitoring.
  - &#x20;Face detection for identifying individuals involved in littering.
  - &#x20;Smoothed results using majority voting for robust classification.
- Multi-Model Predictions

  &#x20;Concurrent processing of multiple models for efficiency.

  &#x20;Parallel execution using Python’s \`ThreadPoolExecutor\`.
- Model Details
  - Garbage Intensity Model (garbage\_intensity.pt): Detects and quantifies the garbage percentage in an image or frame.
    Confidence threshold: 0.6

  - Garbage Type Classification Model (garbage\_type\_detect.pt): Identifies and classifies different types of garbage.
    Confidence threshold: 0.4

  - Littering Detection Model (littering2.pt): Tracks and identifies instances of littering. Confidence threshold: 0.7

---

## Prerequisites

- Python 3.11.0+
- CUDA-compatible GPU (optional but recommended for faster inference)
- Git (to clone the repository)

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/OmNagvekar/AI_Predictions_API.git
   cd AI_Predictions_API
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Requirements**:


   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Directories**:
   Ensure the required directories are in place:

   ```bash
   mkdir -p upload
   ```

5. **Model Caching**:
   Download and cache the Florence-2-base model by running the app once or manually setting up the cache directory (`./` by default).

---

## Running the Project

1. **Start the API Server**:

   ```bash
   uvicorn API:app --host 0.0.0.0 --port 8000  OR
   python API.py
   ```

2. **Access API Endpoints**:

   - Open your browser and navigate to `http://127.0.0.1:8000/docs` to view the automatically generated Swagger UI.

3. **Endpoints Overview**:

   - **`POST /process-image`**: Upload an image for analysis.
   - **`POST /process-video`**: Upload a video for frame-by-frame garbage analysis.

---

## Usage

### Image Processing

1. Send a `POST` request to `/process-image` with an image file.
2. Receive predictions including intensity, type, and litter information.
3. Optionally download the processed image from the provided URL.

### Video Processing

1. Send a `POST` request to `/process-video` with a video file.
2. Receive frame-by-frame analysis results, including average garbage percentages and type statistics.

---

## Directory Structure

```plaintext
.
├── API.py                 # Main application file
├── garbage.py             # Garbage prediction logic
├── upload/                # Directory for storing uploaded files
├── requirements.txt       # Dependencies (to be added later)
├── garbage_intensity.pt   # Detects and quantifies the garbage percentage in an image or frame.
├── garbage_type_detect.pt # Identifies and classifies different types of garbage.
├── littering2.pt          # Tracks and identifies instances of littering.
└── README.md              # Project documentation
```

---

## Example Requests

### Image Upload
```bash
curl -X POST "http://127.0.0.1:8000/process-image" \
-H "Content-Type: multipart/form-data" \
-F "file=@path/to/image.jpg"
```

### Video Upload
```bash
curl -X POST "http://127.0.0.1:8000/process-video" \
-H "Content-Type: multipart/form-data" \
-F "file=@path/to/video.mp4"
```

---

## Notes

- Ensure `torch` is installed with the appropriate CUDA version if running on a GPU.
- Verify that the Florence-2-base model is correctly cached in the specified directory.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

---

## Authors
- Om Nagvekar
- Chinmay Bhosale

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.


---

## Contact
For any questions or suggestions, feel free to contact on below Contact details:

- Om Nagvekar Portfolio Website, Email: https://omnagvekar.github.io/ , omnagvekar29@gmail.com
- Chinmay Bhosale : chinmayhbhosale02@gmail.com
- GitHub Profile:
   - Om Nagvekar: https://github.com/OmNagvekar
   - Chinmay Bhosale: http://github.com/chinu0609
