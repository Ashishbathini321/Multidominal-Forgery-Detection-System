# utils/image_forensics.py
# Minimal changes: embeddings are available but identity_similarity_pct now
# ignores embeddings and relies on ORB+RANSAC geometry to avoid false positives.

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple, Dict

# Optional embedding backend (face_recognition)
try:
    import face_recognition
    _HAS_FR = True
except Exception:
    face_recognition = None
    _HAS_FR = False


# ---------------- small math helpers ----------------

def _l2norm(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return None
    return v / n

def _cos01(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity mapped from [-1,1] -> [0,1]."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    raw = float(np.dot(a, b) / (na * nb))
    return float(np.clip((raw + 1.0) * 0.5, 0.0, 1.0))


# ---------------- face detection / crops ----------------

def _detect_face_rects(bgr: np.ndarray) -> list:
    """Return face boxes [(x,y,w,h), ...] using Haar cascade."""
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return []
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def _crop_face_strict(bgr: np.ndarray, margin: float = 0.25) -> Optional[np.ndarray]:
    """Crop largest face with a tight square margin; return BGR crop or None."""
    boxes = _detect_face_rects(bgr)
    if not boxes:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    cx, cy = x + w // 2, y + h // 2
    s = int(max(w, h) * (1.0 + margin))
    x1 = max(0, cx - s // 2)
    y1 = max(0, cy - s // 2)
    x2 = min(bgr.shape[1], x1 + s)
    y2 = min(bgr.shape[0], y1 + s)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop

def _jitter_crops(bgr: np.ndarray, n: int = 5, resize=(160, 160)) -> list:
    """Small translations to reduce alignment sensitivity."""
    H, W = bgr.shape[:2]
    crops = []
    for dx, dy in [(0, 0), (8, 0), (-8, 0), (0, 8), (0, -8)][:n]:
        x1 = max(0, 0 + dx)
        y1 = max(0, 0 + dy)
        x2 = min(W, W + dx)
        y2 = min(H, H + dy)
        c = bgr[y1:y2, x1:x2]
        if c.size:
            crops.append(cv2.resize(c, resize, interpolation=cv2.INTER_LINEAR))
    if not crops:
        crops = [cv2.resize(bgr, resize, interpolation=cv2.INTER_LINEAR)]
    return crops

def _face_quality(bgr: np.ndarray) -> float:
    """Quick quality estimate 0..1 (blur/size)."""
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    var_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = np.clip(var_lap / 200.0, 0.0, 1.0)
    size_ok = np.clip(min(bgr.shape[:2]) / 120.0, 0.0, 1.0)
    return 0.6 * blur + 0.4 * size_ok


# ---------------- embedding (face_recognition OR fallback) ----------------

def _fallback_embedding(bgr: np.ndarray, out_dim: int = 128) -> np.ndarray:
    """
    Lightweight fallback embedding: normalized 3D color histogram expanded/padded to out_dim.
    """
    if bgr is None or bgr.size == 0:
        return np.zeros(out_dim, dtype=np.float32)
    hist = cv2.calcHist([bgr], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256]).flatten()
    norm = np.linalg.norm(hist) + 1e-9
    hist = (hist / norm).astype(np.float32)
    if hist.size < out_dim:
        hist = np.pad(hist, (0, out_dim - hist.size), mode='constant')
    return hist[:out_dim].astype(np.float32)

def face_embedding(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Return 128-d embedding:
      - If face_recognition is available, use its encodings (128-d).
      - Otherwise produce a deterministic fallback embedding (128-d histogram).
    """
    if bgr is None or bgr.size == 0:
        return None

    if _HAS_FR:
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            if not boxes:
                return None
            enc = face_recognition.face_encodings(rgb, known_face_locations=[boxes[0]])
            if not enc:
                return None
            return np.asarray(enc[0], dtype=np.float32)
        except Exception:
            return _fallback_embedding(bgr, out_dim=128)

    return _fallback_embedding(bgr, out_dim=128)


# ---------------- ORB + RANSAC geometry ----------------

def _orb_match_score(faceA: np.ndarray, faceB: np.ndarray) -> Tuple[float, int, int, int]:
    """Return (score01, inliers, good_matches, total_matches)."""
    try:
        grayA = cv2.cvtColor(faceA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(faceB, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0.0, 0, 0, 0

    orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, edgeThreshold=15, patchSize=31)
    k1, d1 = orb.detectAndCompute(grayA, None)
    k2, d2 = orb.detectAndCompute(grayB, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return 0.0, 0, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < 12:
        return 0.0, 0, len(good), len(knn)

    ptsA = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return 0.0, 0, len(good), len(knn)

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / max(1, len(good))
    score01 = float(np.clip((inlier_ratio - 0.35) / 0.65, 0.0, 1.0))
    return score01, inliers, len(good), len(knn)


# ---------------- AI probability (improved) ----------------

def _highfreq_ratio(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0,0), sigmaX=3.0)
    hf = gray - low
    hf_energy = float(np.mean(np.abs(hf)))
    base = float(np.mean(np.abs(gray))) + 1e-6
    ratio = np.clip(hf_energy / base, 0.0, 5.0)
    return float(np.tanh(ratio * 0.6))

def _ela_inconsistency(bgr: np.ndarray, q: int = 90) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return 0.0
    rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(bgr, rec).astype(np.float32)
    v = float(np.mean(diff))
    return float(np.clip(v / 30.0, 0.0, 1.0))

def _noise_irregularity(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    noise = cv2.subtract(gray, blur).astype(np.float32)
    overall = float(np.std(noise)) + 1e-6
    h, w = noise.shape
    bs = 32
    blocks = []
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            blk = noise[y:y+bs, x:x+bs]
            if blk.size:
                blocks.append(float(np.std(blk)))
    if not blocks:
        return 0.0
    block_std = float(np.std(blocks))
    val = np.clip((block_std / overall), 0.0, 5.0)
    return float(np.tanh(val * 0.9))

def ai_probability(bgr: np.ndarray, debug: bool = False) -> Tuple[float, Dict]:
    """
    Returns (prob_in_0_1, details). Higher -> more likely AI/synthetic.
    Call with debug=True to see raw feature values.
    """
    if bgr is None or bgr.size == 0:
        return 0.0, {"err": "empty"}

    hf = _highfreq_ratio(bgr)
    ela90 = _ela_inconsistency(bgr, q=90)
    ela75 = _ela_inconsistency(bgr, q=75)
    noise_irreg = _noise_irregularity(bgr)
    lap = float(np.var(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F)))
    lap_norm = float(np.clip(lap / 400.0, 0.0, 1.0))

    ela = max(ela90, ela75)
    comp_hf = (1.0 - hf) * 0.25
    comp_ela = np.clip(ela * 0.30 - 0.05, 0.0, 0.45)
    comp_noise = np.clip(noise_irreg * 0.45, 0.0, 0.5)
    comp_lap = np.clip((1.0 - lap_norm) * 0.20, 0.0, 0.2)

    raw = comp_hf + comp_ela + comp_noise + comp_lap
    prob = float(np.clip(raw, 0.0, 1.0))

    details = {
        "hf": hf,
        "ela90": ela90,
        "ela75": ela75,
        "ela": ela,
        "noise_irreg": noise_irreg,
        "lap_var": lap,
        "lap_norm": lap_norm,
        "raw": raw
    }
    if debug:
        return prob, details
    return prob, details


# ---------------- identity similarity (ORB-only fusion) ----------------

def identity_similarity_pct(bgr_a: np.ndarray, bgr_b: np.ndarray) -> Tuple[float, Dict]:
    """
    Robust geometry-first identity score:
      - Crop largest faces
      - ORB+RANSAC geometry score (primary)
      - Embeddings are intentionally ignored in this quick-patch version to avoid
        false positives produced by simple fallback embeddings.
    Returns (pct_0_100, meta_dict)
    """
    meta = {"qa": 0.0, "qb": 0.0, "pairs": 0, "mode": "", "inliers": 0, "good": 0}

    faceA = _crop_face_strict(bgr_a)
    faceB = _crop_face_strict(bgr_b)
    if faceA is None or faceB is None:
        meta["reason"] = "no_face"
        return 0.0, meta

    qa = _face_quality(faceA)
    qb = _face_quality(faceB)
    meta["qa"], meta["qb"] = qa, qb

    # Normalize to fixed size for ORB
    faceA = cv2.resize(faceA, (224, 224), interpolation=cv2.INTER_LINEAR)
    faceB = cv2.resize(faceB, (224, 224), interpolation=cv2.INTER_LINEAR)

    orb_s, inliers, good, total = _orb_match_score(faceA, faceB)
    meta["inliers"], meta["good"], meta["pairs"] = int(inliers), int(good), int(inliers)
    meta["mode"] = "orb_only"  # forced mode

    # Force ignore any embeddings (quick patch). If you want embeddings back later,
    # replace this logic with embedding fusion and tune weights.
    emb_s = None

    # Fusion and guard rails (geometry-first)
    if inliers < 18 or orb_s < 0.25:
        orb_s = min(orb_s, 0.25)
        final01 = orb_s
    else:
        final01 = orb_s

    # Quality penalty
    if min(qa, qb) < 0.35:
        final01 = max(0.0, final01 - 0.15)

    pct = float(np.clip(final01, 0.0, 1.0) * 100.0)
    return pct, meta


# ---------------- exported helpers (public API) ----------------

def crop_face(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Public wrapper expected by other modules.
    Attempts a strict crop first; returns None if no face found.
    """
    return _crop_face_strict(bgr)


__all__ = [
    "ai_probability",
    "identity_similarity_pct",
    "face_embedding",
    "crop_face",
]
