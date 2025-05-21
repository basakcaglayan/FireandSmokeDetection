
import firebase_admin
import firebase_admin
from firebase_admin import credentials, storage, db
import cv2
import numpy as np
from PIL import Image
import io
import time
from ultralytics import YOLO
import supervision as sv

# Firebase Initialization
try:
    # Replace with the actual path to your Firebase Admin SDK JSON file
    cred = credentials.Certificate("fireandsmokedetection-8631e-firebase-adminsdk-fbsvc-4b4740fd67.json") 
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'fireandsmokedetection-8631e.firebasestorage.app',  
        'databaseURL': 'https://fireandsmokedetection-8631e-default-rtdb.firebaseio.com/' 
    })

except Exception as e:
    print(f"Firebase initialization error: {e}")
    exit()

bucket = storage.bucket()
db_ref_status = db.reference('/status')

# YOLO Model Initialization
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"[Model Load Error] {e}")
    exit()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def get_image_from_firebase():
    """Retrieve and preprocess the image from Firebase Storage."""
    try:
        # image_name_firebase = "3.jpg"
        image_name_firebase = "data/photo.jpg"
        blob = bucket.blob(image_name_firebase)

        if not blob.exists():
            print(f"[Storage Error] Image '{image_name_firebase}' not found.")
            return None

        image_data = blob.download_as_bytes()
        image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')  # Ensures RGB

        image_np = np.array(image_pil)
        return image_np

    except Exception as e:
        print(f"[Image Retrieval Error] {e}")
        return None

def main():
    """Main loop for detection and Firebase update."""
    while True:
        frame_rgb = cv2.flip(get_image_from_firebase(),0)

        if frame_rgb is not None:
            # Resize to model input size
            resized_rgb = cv2.resize(frame_rgb, (640, 640))

            # Convert RGB to BGR (YOLO usually trained on BGR)
            resized_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)

            results = model(resized_bgr)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Debug output
            print("=== Detection Results ===")
            print("Boxes:", results.boxes)
            print("Class IDs:", detections.class_id)
            print("Confidences:", detections.confidence)

            LED_STATE = False
            detected_labels_list = []

            if detections.class_id is not None and len(detections.class_id) > 0:
                for i in range(len(detections.class_id)):
                    label_index = int(detections.class_id[i])
                    if 0 <= label_index < len(model.names):
                        label = model.names[label_index]
                        detected_labels_list.append(label)
                        if "fire" in label.lower():
                            LED_STATE = True

            # Update Firebase Realtime Database
            try:
                db_ref_status.set({
                    'LED_STATE': LED_STATE,
                    'detected_objects': detected_labels_list,
                    'timestamp': int(time.time())
                })
                print("🔥 Fire DETECTED!" if LED_STATE else "✅ No fire detected.")
                print("Objects:", detected_labels_list)
            except Exception as e:
                print(f"[Database Update Error] {e}")

            # Annotate frame
            annotated = resized_bgr.copy()
            custom_labels = []
            for i in range(len(detections.xyxy)):
                class_id = int(detections.class_id[i])
                confidence = detections.confidence[i]
                name = model.names[class_id] if 0 <= class_id < len(model.names) else "Unknown"
                custom_labels.append(f"{name} {confidence:.2f}")

            annotated = bounding_box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=custom_labels)

            # Show image
            cv2.imshow("Firebase Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Program exited by user.")
                break
        else:
            print("⚠️ No image received from Firebase. Retrying in 5s...")
            time.sleep(5)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
