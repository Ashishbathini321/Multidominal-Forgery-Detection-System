import cv2

def safe_face_crop(img):
    """
    Try face detection.
    If fails → fallback to center crop.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            return img[y:y+h, x:x+w]

        # fallback center crop
        h, w = img.shape[:2]
        s = min(h, w)
        return img[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]

    except Exception:
        return img