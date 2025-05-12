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

cred = credentials.Certificate("fireandsmokedetection-8631e-firebase-adminsdk-fbsvc-4b4740fd67.json") 
firebase_admin.initialize_app(cred, {
    'storageBucket': 'fireandsmokedetection-8631e.firebasestorage.app',  
    'databaseURL': 'https://fireandsmokedetection-8631e-default-rtdb.firebaseio.com/' 
})

bucket = storage.bucket()
db_ref = db.reference('/images')

model = YOLO('best.pt')

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


# # 3. Function to Fetch Image from Firebase
# def get_image_from_firebase():
#     """Retrieves the latest PNG image from Firebase Storage and converts it to a NumPy array."""
#     try:
#         snapshot = db_ref.order_by_key().limit_to_last(1).get()
#         if snapshot:
#             for key, value in snapshot.items():
#                 image_name = value['filename']
#                 print(f"Image name: {image_name}")

#                 image_path = f'images/{image_name}'

#                 blob = bucket.blob(image_path)
#                 image_data = blob.download_as_bytes()

#                 image = Image.open(io.BytesIO(image_data))
#                 image_np = np.array(image)

#                 return image_np

#         else:
#             print("No images found in Firebase Realtime Database.")
#             return None


#     except Exception as e:
#         print(f"Error getting image from Firebase: {e}")
#         return None


# # 4. Main Loop
# def main():
#     while True:
#         frame = get_image_from_firebase() 

#         if frame is not None:
#             results = model(frame)[0]
#             detections = sv.Detections.from_ultralytics(results)

#             annotated_image = bounding_box_annotator.annotate(
#                 scene=frame, detections=detections)
#             annotated_image = label_annotator.annotate(
#                 scene=annotated_image, detections=detections)

#             cv2.imshow("Firebase Stream", annotated_image)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             print("No image received. Waiting...")
#             time.sleep(1)

#     cv2.destroyAllWindows()

# Manual testing of image retrieval
def get_image_from_firebase():
    """Retrieves the image from Firebase Storage.  Filename is now hardcoded for testing."""
    try:
        # image_name = "firepng.png"
        image_name = "data/photo.jpg"

        image_path = f'{image_name}'


        blob = bucket.blob(image_path)
        image_data = blob.download_as_bytes()

        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        return image_np

    except FileNotFoundError:
        print("Error: Service account key file not found.")
        return None
    except Exception as e:
        print(f"Error getting image from Firebase: {e}")
        return None


def main():
    while True:
        frame = get_image_from_firebase()

        if frame is not None:
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)

            annotated_image = bounding_box_annotator.annotate(
                scene=frame, detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections)

            cv2.imshow("Firebase Stream", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No image received. Waiting...")
            time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()