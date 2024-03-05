import cv2
from ultralytics import YOLO
import supervision as sv

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CONFIDENCE_THRESHOLD = 0.7
THICKNESS = 2

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=THICKNESS,
        text_thickness=THICKNESS,
        text_scale=1
    )

    while True:
        try:
            ret, frame = cap.read()

            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov5(result)
            
            high_conf_detections = [
                (x, confidence, class_id, class_name) 
                for x, confidence, class_id, class_name 
                in detections if confidence > CONFIDENCE_THRESHOLD and model.model.names[class_id] == "pothole"
            ]

            labels = [
                f"{class_name} {confidence:0.2f}"
                for _, confidence, _, class_name 
                in high_conf_detections
            ]

            print("High confidence detections:", high_conf_detections)

            frame_annotated = box_annotator.annotate(
                scene=frame.copy(), 
                detections=high_conf_detections
            )

            # Display the annotated frame
            cv2.imshow("Live Feed with Annotations", frame_annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
