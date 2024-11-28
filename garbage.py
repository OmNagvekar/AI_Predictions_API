import ultralytics
from ultralytics import YOLO
import cv2
from shapely.geometry import box
from shapely.ops import unary_union


class Predictions:
    def __init__(self, file) -> None:
        ultralytics.checks()
        self.model_intensity = YOLO('./garbage_intensity.pt')
        self.model_type = YOLO('./garbage_type_detect.pt')
        self.model_littering = YOLO('./littering2.pt')
        self.model_intensity_conf = 0.6
        self.model_type_conf = 0.4
        self.model_littering_conf = 0.6
        self.image = self.process_image(file)

    def process_image(self, file):
        image = cv2.imread(file)
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def predict(self, chosen_model, img, classes=[], conf=0.25):
        if classes:
            results = chosen_model.predict(img, classes=classes, conf=conf)
        else:
            results = chosen_model.predict(img, conf=conf)
        return results

    def predict_and_detect(self, chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1, intensity=False):
        results = self.predict(chosen_model, img, classes, conf=conf)
        
        # Initialize variables for garbage percentage and object counts
        garbage_percentage = None
        object_counts = {}  # Dictionary to count object occurrences
        total_objects = 0   # Total detected objects
        
        # Check if no objects were detected
        if not results:
            return img, results, garbage_percentage, object_counts  # Return empty counts if no objects detected
        
        if intensity:
            # Handle intensity processing when objects are detected
            bounding_boxes = [coordinates.xyxy[0] for coordinates in results[0].boxes]
            rectangles = [box(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in bounding_boxes]
            merged_area = unary_union(rectangles).area
            image_area = 640 * 640
            garbage_percentage = (merged_area / image_area) * 100

        # Iterate over the detected objects and draw bounding boxes
        for result in results:
            for boxs in result.boxes:
                cls_name = result.names[int(boxs.cls[0])]
                object_counts[cls_name] = object_counts.get(cls_name, 0) + 1
                total_objects += 1
                # Draw rectangles and labels on the image
                cv2.rectangle(img, (int(boxs.xyxy[0][0]), int(boxs.xyxy[0][1])),
                            (int(boxs.xyxy[0][2]), int(boxs.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{cls_name}",
                            (int(boxs.xyxy[0][0]), int(boxs.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)

        # Calculate percentage for each object type
        object_percentages = {
            obj: (count / total_objects) * 100 for obj, count in object_counts.items()
        } if total_objects > 0 else {}

        return img, results, garbage_percentage, object_percentages



    def predict_all(self):
        from concurrent.futures import ThreadPoolExecutor
        results = {}

        def intensity_task():
            return self.predict_and_detect(self.model_intensity, self.image, conf=self.model_intensity_conf, intensity=True)

        def type_task():
            return self.predict_and_detect(self.model_type, self.image, conf=self.model_type_conf)

        def litter_task():
            return self.predict_and_detect(self.model_littering, self.image, conf=self.model_littering_conf)

        with ThreadPoolExecutor() as executor:
            future_intensity = executor.submit(intensity_task)
            future_type = executor.submit(type_task)
            future_litter = executor.submit(litter_task)

            results['intensity'] = future_intensity.result()
            results['type'] = future_type.result()
            results['litter'] = future_litter.result()

        return results
