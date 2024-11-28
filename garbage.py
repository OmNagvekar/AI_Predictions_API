import ultralytics
from ultralytics import YOLO
import cv2

class Predictions:
    def __init__(self,file) -> None:
        ultralytics.checks()
        self.model_intensity = YOLO('garabge_intensity.pt')
        self.model_type = YOLO('garbage_type_detect.pt')
        self.model_littering = YOLO('littering2.pt')
        self.model_intensity_conf = 0.6
        self.model_type_conf = 0.4
        self.model_littering_conf = 0.6
        self.image = self.process_image(file)
    
    def process_image(self,file):
        image = cv2.imread(file)
        image = cv2.resize(image, (640,640))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image
    def pred_intensity(self):
        image,result = self.predict_and_detect(self.model_intensity,self.image,conf=self.model_intensity_conf)
        return image,result
    def predict(self,chosen_model, img, classes=[], conf=0.25):
            if classes:
                results = chosen_model.predict(img, classes=classes, conf=conf)
            else:
                results = chosen_model.predict(img, conf=conf)

            return results
    def predict_and_detect(self,chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(chosen_model, img, classes, conf=conf)
        for result in results:
            for box in result.boxes:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
        return img, results