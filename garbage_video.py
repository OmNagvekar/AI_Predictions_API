import ultralytics
from ultralytics import YOLO
import cv2
from shapely.geometry import box
from shapely.ops import unary_union


class PredictionsVideo:
    def __init__(self, video_file) -> None:
        ultralytics.checks()
        self.model_intensity = YOLO('./garbage_intensity.pt')
        self.model_type = YOLO('./garbage_type_detect.pt')
        self.model_littering = YOLO('./littering2.pt')
        self.model_intensity_conf = 0.6
        self.model_type_conf = 0.4
        self.model_littering_conf = 0.6
        self.video_file = video_file

    def predict(self, chosen_model, img, classes=[], conf=0.25):
        if classes:
            results = chosen_model.predict(img, classes=classes, conf=conf)
        else:
            results = chosen_model.predict(img, conf=conf)
        return results

    def process_frame(self, frame):
        """ Resize and prepare the frame for predictions """
        frame_resized = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def predict_and_detect(self, chosen_model, frame, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1, intensity=False):
        frame_rgb = self.process_frame(frame)
        results = self.predict(chosen_model, frame_rgb, classes, conf=conf)

        garbage_percentage = None
        object_counts = {}
        total_objects = 0
        
        if not results:
            return frame, results, garbage_percentage, object_counts

        if intensity:
            bounding_boxes = [coordinates.xyxy[0] for coordinates in results[0].boxes]
            rectangles = [box(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in bounding_boxes]
            merged_area = unary_union(rectangles).area
            image_area = 640 * 640
            garbage_percentage = (merged_area / image_area) * 100

        for result in results:
            for boxs in result.boxes:
                cls_name = result.names[int(boxs.cls[0])]
                object_counts[cls_name] = object_counts.get(cls_name, 0) + 1
                total_objects += 1
                cv2.rectangle(frame, (int(boxs.xyxy[0][0]), int(boxs.xyxy[0][1])),
                              (int(boxs.xyxy[0][2]), int(boxs.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(frame, f"{cls_name}",
                            (int(boxs.xyxy[0][0]), int(boxs.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)

        object_percentages = {
            obj: (count / total_objects) * 100 for obj, count in object_counts.items()
        } if total_objects > 0 else {}

        return frame, results, garbage_percentage, object_percentages

    def process_video(self):
        """ Process video frame by frame """
        cap = cv2.VideoCapture(self.video_file)
        frame_count = 0
        processed_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            results = {}

            processed_frame_intensity, _, garbage_percentage, _ = self.predict_and_detect(self.model_intensity, frame, intensity=True)
            processed_frame_type, _, _, object_percentages = self.predict_and_detect(self.model_type, frame)
            processed_frame_litter, _, _, _ = self.predict_and_detect(self.model_littering, frame)

            processed_frames.append(processed_frame_intensity)

            results['intensity'] = (processed_frame_intensity, garbage_percentage)
            results['type'] = (processed_frame_type, object_percentages)
            results['litter'] = processed_frame_litter


        cap.release()
        return processed_frames, results
