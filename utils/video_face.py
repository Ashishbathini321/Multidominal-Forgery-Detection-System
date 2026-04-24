# utils/video_face.py

import cv2
import numpy as np

def read_frames(path, max_frames=90, step=3, resize_to=640):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break

        if idx % step == 0:
            h, w = f.shape[:2]
            scale = resize_to / max(h, w)
            if scale < 1.0:
                f = cv2.resize(f, (int(w*scale), int(h*scale)))

            frames.append(f)

            if len(frames) >= max_frames:
                break

        idx += 1

    cap.release()
    return frames


# ✅ REPLACEMENT (NO face_recognition)
def face_embed_bgr(frame_bgr):
    """
    Lightweight embedding using color histogram (NO dlib needed)
    """
    if frame_bgr is None:
        return None

    img = cv2.resize(frame_bgr, (64, 64))

    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8],
                        [0,256,0,256,0,256]).flatten()

    hist = hist / (np.linalg.norm(hist) + 1e-8)
    return hist


def video_face_embeddings(path, max_frames=90, step=3):
    frames = read_frames(path, max_frames=max_frames, step=step)

    embeds, sharp = [], []

    for f in frames:
        e = face_embed_bgr(f)

        if e is not None:
            embeds.append(e)

            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    return np.array(embeds), np.array(sharp)


def cos_sim(a, b, eps=1e-9):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))


def temporal_flicker_score(embeds, sharp):
    if len(embeds) < 3:
        return 50.0

    dists = []
    for i in range(1, len(embeds)):
        d = 1.0 - cos_sim(embeds[i-1], embeds[i])
        dists.append(d)

    id_flicker = float(np.mean(dists))

    if len(sharp) >= 3:
        sdiff = np.abs(np.diff(sharp))
        s_norm = float(np.tanh(np.mean(sdiff) / (np.mean(sharp) + 1e-6)))
    else:
        s_norm = 0.2

    Susp = np.clip(70*id_flicker + 30*s_norm, 0, 1) * 100.0
    return float(Susp)