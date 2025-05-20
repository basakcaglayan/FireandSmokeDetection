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

model = YOLO('best-1.pt')

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def get_image_from_firebase():
    """Retrieves the image from Firebase Storage."""
    try:
        # image_name = "firepng.png" # Test için
        image_name = "data/photo.jpg" # Gerçek dosya yolu
        image_path = f'{image_name}' # Firebase Storage'daki tam yol

        blob = bucket.blob(image_path)
        if not blob.exists():
            print(f"Hata: Firebase Storage'da '{image_path}' bulunamadı.")
            return None

        image_data = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_data))
        
        # Pillow görüntüsünü NumPy dizisine dönüştür (Bu aşamada RGB formatındadır)
        image_np_rgb = np.array(image)
        
        # Eğer görüntü RGBA (alpha kanalı ile) ise RGB'ye çevir
        if image_np_rgb.shape[2] == 4:
            image_np_rgb = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGBA2RGB)
        
        return image_np_rgb # RGB olarak döndür

    except firebase_admin.exceptions.NotFoundError:
        print(f"Hata: Firebase Storage'da '{image_path}' bulunamadı (SDK hatası).")
        return None
    except Exception as e:
        print(f"Firebase'den görüntü alınırken hata oluştu: {e}")
        return None
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

def main():
    while True:
        frame = get_image_from_firebase()

        if frame is not None:
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)

            fire_detected = False
            for label in detections.class_id:
                # Bu kısmı kendi modelinin sınıf adlarına göre düzenle
                class_name = model.names[label]
                if "fire" in class_name.lower():  # veya class_name == 'fire' gibi sabit kontrol
                    fire_detected = True
                    break

            # Firebase'e yangın durumu güncellemesi
            db.reference('/status').set({
                'fire_detected': fire_detected,
                'timestamp': int(time.time())
            })

            # Görsel anotasyon ve gösterim
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