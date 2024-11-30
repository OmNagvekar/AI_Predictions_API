import ultralytics
from ultralytics import YOLO
import cv2
from shapely.geometry import box
from shapely.ops import unary_union

# Class to handle predictions from different YOLO models
class Predictions:
    def __init__(self, file, stream: bool = False) -> None:
        # Load the pre-trained YOLO models for different tasks
        self.model_intensity = YOLO('./garbage_intensity.pt')
        self.model_type = YOLO('./garbage_type_detect.pt')
        self.model_littering = YOLO('./littering2.pt')
        
        # Confidence thresholds for different models
        self.model_intensity_conf = 0.6
        self.model_type_conf = 0.4
        self.model_littering_conf = 0.7
        
        # Set the input file (image/video)
        self.file = file 
        
        # If streaming, don't process image yet, else process image immediately
        if stream:
            self.image = None 
        else:
            self.image = self.process_image(file)

    # Function to process and resize the image from the file for prediction
    def process_image(self, file):
        image = cv2.imread(file)
        image = cv2.resize(image, (640, 640))  # Resize to 640x640 for model input
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for model processing
        return image

    # Function to process individual frames during video prediction
    def process_frame(self, img):
        image = cv2.resize(img, (640, 640))  # Resize the frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        return image

    # Function to run predictions using a chosen model
    def predict(self, chosen_model, img, classes=[], conf=0.25):
        if classes:
            # Predict using specified classes
            results = chosen_model.predict(img, classes=classes, conf=conf)
        else:
            # Predict without filtering by classes
            results = chosen_model.predict(img, conf=conf)
        return results

    # Function to predict and detect objects with additional processing (bounding boxes, percentages)
    def predict_and_detect(self, chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1, intensity=False):
        results = self.predict(chosen_model, img, classes, conf=conf)
        
        # Initialize variables for garbage percentage and object counts
        garbage_percentage = None
        object_counts = {}  # Dictionary to track object counts
        total_objects = 0   # Total number of objects detected
        
        # If no objects detected, return the image as is
        if not results:
            return img, results, garbage_percentage, object_counts  # No objects detected
        
        # If intensity processing is enabled, calculate garbage percentage
        if intensity:
            bounding_boxes = [coordinates.xyxy[0] for coordinates in results[0].boxes]
            rectangles = [box(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in bounding_boxes]
            merged_area = unary_union(rectangles).area  # Combine bounding boxes to calculate area
            image_area = 640 * 640  # Area of the image
            garbage_percentage = (merged_area / image_area) * 100  # Calculate the percentage of garbage

        # Iterate over the detected objects and draw bounding boxes with labels
        for result in results:
            for boxs in result.boxes:
                cls_name = result.names[int(boxs.cls[0])]  # Get the object class name
                object_counts[cls_name] = object_counts.get(cls_name, 0) + 1  # Update object count
                total_objects += 1  # Increment total object count
                
                # Draw bounding box and label on the image
                cv2.rectangle(img, (int(boxs.xyxy[0][0]), int(boxs.xyxy[0][1])),
                            (int(boxs.xyxy[0][2]), int(boxs.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{cls_name}",
                            (int(boxs.xyxy[0][0]), int(boxs.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)

        # Calculate the percentage of each object class in the total count
        object_percentages = {
            obj: (count / total_objects) * 100 for obj, count in object_counts.items()
        } if total_objects > 0 else {}

        return img, results, garbage_percentage, object_percentages

    # Function to process predictions over a video file
    def predict_over_video(self):
        cap_ = cv2.VideoCapture(self.file)  # Open the video file
        pred_frames = []  # List to store processed frames
        type_sums = []  # List to store type sums
        average_garbage = []  # List to store average garbage percentages

        # Read and process each frame from the video
        while cap_.isOpened():
            ret, frame = cap_.read()
            if not ret:
                break

            # Process the frame
            self.image = self.process_frame(frame) 
            results = self.predict_all()  # Get predictions for all models
            pred_frames.append(results['intensity'][0])  # Append processed frame
            average_garbage.append(results['intensity'][2])  # Append garbage percentage
            type_sums.append(results['type'][3])  # Append type sums
            
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
        
        # Release the video capture object
        cap_.release()

        # Return the results of predictions over the video
        return pred_frames, results, average_garbage, type_sums

    # Function to run predictions for all models concurrently
    def predict_all(self):
        from concurrent.futures import ThreadPoolExecutor
        results = {}

        # Function for intensity model prediction task
        def intensity_task():
            return self.predict_and_detect(self.model_intensity, self.image, conf=self.model_intensity_conf, intensity=True)

        # Function for type model prediction task
        def type_task():
            return self.predict_and_detect(self.model_type, self.image, conf=self.model_type_conf)

        # Function for littering model prediction task
        def litter_task():
            return self.predict_and_detect(self.model_littering, self.image, conf=self.model_littering_conf)

        # Use ThreadPoolExecutor for concurrent execution of prediction tasks
        with ThreadPoolExecutor() as executor:
            # Submit tasks for parallel execution
            future_intensity = executor.submit(intensity_task)
            future_type = executor.submit(type_task)
            future_litter = executor.submit(litter_task)

            # Collect the results
            results['intensity'] = future_intensity.result()
            results['type'] = future_type.result()
            results['litter'] = future_litter.result()

        return results
