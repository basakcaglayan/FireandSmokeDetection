import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db
import cv2
import numpy as np
from PIL import Image
import io
import time
from ultralytics import YOLO
import supervision as sv

# Firebase initialization (ensure your credential file is in the correct path)
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
# Reference for images (if you were to list them or use a dynamic name, not used in current get_image)
# db_ref_images = db.reference('/images')
db_ref_status = db.reference('/status') # Reference for the detection status

# Load the YOLO model (ensure 'best-1.pt' is in your working directory or provide the full path)
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def get_image_from_firebase():
    """Retrieves the image from Firebase Storage."""
    try:
        image_name_firebase = "toplu.jpg" # For testing with a specific file
        # image_name_firebase = "data/photo.jpg" # Actual file path in Firebase Storage
        blob = bucket.blob(image_name_firebase)

        if not blob.exists():
            print(f"Error: Image '{image_name_firebase}' not found in Firebase Storage.")
            return None

        image_data = blob.download_as_bytes()
        image_pil = Image.open(io.BytesIO(image_data))
        
        image_np_rgb = np.array(image_pil)
        
        # If the image is RGBA (with alpha channel), convert it to RGB
        if image_np_rgb.ndim == 3 and image_np_rgb.shape[2] == 4:
            image_np_rgb = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGBA2RGB)
        # If the image is Grayscale, convert it to RGB for consistency with model and display
        elif image_np_rgb.ndim == 2:
            image_np_rgb = cv2.cvtColor(image_np_rgb, cv2.COLOR_GRAY2RGB)
        # Ensure it's a 3-channel image if it wasn't RGBA or Grayscale
        elif image_np_rgb.ndim == 3 and image_np_rgb.shape[2] != 3:
            print(f"Warning: Image has an unexpected number of channels: {image_np_rgb.shape[2]}. Attempting to use as is.")


        return image_np_rgb # Return as RGB

    except firebase_admin.exceptions.NotFoundError:
        print(f"Error: Image '{image_name_firebase}' not found in Firebase Storage (SDK error).")
        return None
    except Exception as e:
        print(f"Error getting image from Firebase: {e}")
        return None


def main():
    """Main loop to fetch images, process them, and display results."""
    while True:
        frame_rgb = get_image_from_firebase()

        if frame_rgb is not None:
            results = model(frame_rgb)[0]
            detections = sv.Detections.from_ultralytics(results)

            LED_STATE = False
            detected_labels_list = []

            if detections.class_id is not None and len(detections.class_id) > 0:
                for i in range(len(detections.class_id)):
                    class_id_tensor = detections.class_id[i]
                    label_index = int(class_id_tensor.item()) if hasattr(class_id_tensor, 'item') else int(class_id_tensor)
                    
                    if 0 <= label_index < len(model.names):
                        class_name = model.names[label_index]
                        detected_labels_list.append(class_name) # Add class name to the list
                        print(f"Detected: {class_name} (ID: {label_index})")
                        if "fire" in class_name.lower(): # Case-insensitive check for "fire"
                            LED_STATE = True
                    else:
                        print(f"Warning: Invalid label index {label_index} encountered.")

            # Update fire status in Firebase Realtime Database
            try:
                db_ref_status.set({
                    'LED_STATE': LED_STATE,
                    'detected_objects': detected_labels_list, # Send the list of detected objects
                    'timestamp': int(time.time())
                })
                if LED_STATE:
                    print(f"Fire DETECTED! Objects: {detected_labels_list}")
                else:
                    print(f"No fire detected. Objects: {detected_labels_list}")
            except Exception as e:
                print(f"Error updating Firebase Realtime Database: {e}")


            # For visual annotation, use a copy of the RGB frame
            # as annotators might modify the image in-place.
            annotated_image_rgb = frame_rgb.copy()
            
            custom_labels = None
            if detections.class_id is not None and len(detections.class_id) > 0:
                custom_labels = []
                for i in range(len(detections.xyxy)): # Iterate through all detections
                    class_id_tensor = detections.class_id[i]
                    label_idx = int(class_id_tensor.item()) if hasattr(class_id_tensor, 'item') else int(class_id_tensor)
                    confidence = detections.confidence[i] if detections.confidence is not None else 0.0
                    
                    if 0 <= label_idx < len(model.names):
                        label_name = model.names[label_idx]
                        custom_labels.append(f"{label_name} {confidence:.2f}")
                    else:
                        custom_labels.append(f"Unknown {confidence:.2f}")


            annotated_image_rgb = bounding_box_annotator.annotate(
                scene=annotated_image_rgb, detections=detections)
            annotated_image_rgb = label_annotator.annotate(
                scene=annotated_image_rgb, detections=detections,
                labels=custom_labels # Use custom labels with confidence scores
            )

            # Convert RGB to BGR for display with OpenCV
            annotated_image_bgr = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Firebase Stream", annotated_image_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting program...")
                break
        else:
            print("No image received from Firebase. Waiting...")
            time.sleep(5) # Wait a bit longer if an image wasn't received

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()