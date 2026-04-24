# app_compare_videos.py
import os
import cv2
import numpy as np
import tempfile
import streamlit as st
from typing import List, Tuple, Optional

# Toggle debug to True to save crops and print embedding diagnostics
DEBUG = True

# Try to import the project's face embedding / crop helpers.
try:
    from utils.image_forensics import face_embedding, crop_face
except Exception:
    try:
        from .image_forensics import face_embedding, crop_face
    except Exception:
        face_embedding = None
        crop_face = None


# ---------- low-level helpers -----------------------------------------------

def _read_video_frames(path: str, max_frames: int = 64) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if total <= 0:
        read = 0
        while read < max_frames:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            frames.append(bgr)
            read += 1
        cap.release()
        return frames

    idxs = np.linspace(0, max(0, total - 1), num=min(max_frames, max(1, total))).astype(int)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(bgr)
    cap.release()
    return frames


def _face_quality(bgr: Optional[np.ndarray]) -> float:
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = np.clip(var / 200.0, 0.0, 1.0)
    size_ok = np.clip(min(bgr.shape[:2]) / 120.0, 0.0, 1.0)
    return 0.6 * blur + 0.4 * size_ok


def _l2(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    v = v.astype(np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return None
    return v / n


def _align_length(A: np.ndarray, B: np.ndarray, L: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    def resample(X, L):
        if len(X) == 0:
            return X
        idx = np.linspace(0, len(X) - 1, num=L)
        i0 = np.floor(idx).astype(int)
        i1 = np.clip(i0 + 1, 0, len(X) - 1)
        w = (idx - i0)[:, None]
        return (1 - w) * X[i0] + w * X[i1]
    return resample(A, L), resample(B, L)


# ---------- debug helpers ---------------------------------------------------

def _debug_save_crops_and_embs(tag: str, crops: List[np.ndarray], embs: List[np.ndarray]):
    try:
        d = os.path.join(tempfile.gettempdir(), "face_debug")
        os.makedirs(d, exist_ok=True)
        prefix = tag.replace(os.sep, "_")[:16]
        for i, c in enumerate(crops[:8]):
            try:
                fn = os.path.join(d, f"{prefix}_crop_{i}.jpg")
                cv2.imwrite(fn, c)
            except Exception:
                pass
        if len(embs):
            E = np.stack(embs, 0)
            print(f"[DEBUG] {prefix} emb shape: {E.shape}, mean:{E.mean():.8f}, std:{E.std():.8f}")
            up_to = min(6, len(E))
            print("[DEBUG] sample pairwise dot (first rows):")
            for i in range(up_to):
                for j in range(i+1, up_to):
                    s = float(np.dot(E[i], E[j]))
                    print(f"  {i}-{j}: dot={s:.6f}")
            try:
                np.savez(os.path.join(d, f"{prefix}_embs.npz"), E=E)
            except Exception:
                pass
        else:
            print(f"[DEBUG] {prefix} no embeddings extracted.")
        if DEBUG:
            print(f"[DEBUG] saved crops/embeddings to: {d}")
    except Exception as ex:
        print("[DEBUG] debug save failed:", ex)


# ---------- ORB geometry score (wrapped) -----------------------------------

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


# ---------- embedding extraction for video (now returns crops too) ----------

def _embed_video(path: str, max_frames: int = 64) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Extract per-face embeddings, quality scores, and face crops from a video.
    Returns (E, Q, crops):
      E: [T, D] embeddings (float32) or empty (0,D)
      Q: [T] quality floats (float32) or empty
      crops: list of face crops (BGR) aligned with Q (may have fewer entries than frames if crop failed)
    """
    if face_embedding is None or crop_face is None:
        raise RuntimeError("face_embedding and/or crop_face are not available. Check utils.image_forensics imports.")

    frames = _read_video_frames(path, max_frames=max_frames)
    embs: List[np.ndarray] = []
    quals: List[float] = []
    crops: List[np.ndarray] = []

    for bgr in frames:
        face = crop_face(bgr)
        if face is None:
            continue
        e = face_embedding(face)
        e = _l2(e) if e is not None else None
        if e is not None:
            embs.append(e)
        quals.append(_face_quality(face))
        crops.append(face.copy())

    if DEBUG:
        try:
            _debug_save_crops_and_embs(os.path.basename(path), crops, embs)
        except Exception as ex:
            print("[DEBUG] debug helper failed:", ex)

    if not embs:
        E = np.empty((0, 128), np.float32)
    else:
        E = np.stack(embs, 0).astype(np.float32)

    Q = np.asarray(quals, np.float32)
    return E, Q, crops


# ---------- representative center (medoid) ---------------------------------

def _representative_center(E: np.ndarray) -> Optional[np.ndarray]:
    if E is None or len(E) == 0:
        return None
    T = len(E)
    if T == 1:
        return _l2(E[0])
    D = np.dot(E, E.T)
    mean_d = D.mean(axis=1)
    idx = int(np.argmax(mean_d))
    return _l2(E[idx])


# ---------- strict video face-match scoring (fusion with ORB frames) -------

def video_face_match_strict(pathA: str, pathB: str, max_frames: int = 64):
    EA, QA, cropsA = _embed_video(pathA, max_frames=max_frames)
    EB, QB, cropsB = _embed_video(pathB, max_frames=max_frames)

    diag = {
        "framesA": int(len(cropsA)),
        "framesB": int(len(cropsB)),
        "meanQ_A": float(QA.mean() if len(QA) else 0.0),
        "meanQ_B": float(QB.mean() if len(QB) else 0.0),
    }

    # conservative requirements based on face crops (not embeddings)
    min_frames_required = 8
    min_quality_required = 0.30

    if len(cropsA) < min_frames_required or len(cropsB) < min_frames_required:
        return 0.0, f"Uncertain (not enough face frames; need >= {min_frames_required})", diag
    if diag["meanQ_A"] < min_quality_required or diag["meanQ_B"] < min_quality_required:
        return 0.0, "Uncertain (faces too blurry/small)", diag

    # For fusing, resample embeddings (if present) and crops to same length L
    L = min(64, max(len(cropsA), len(cropsB)))
    # if embeddings exist, align them; otherwise produce empty arrays
    if EA.shape[0] > 0 and EB.shape[0] > 0:
        A_emb, B_emb = _align_length(EA, EB, L=L)
        emb_exists = True
    else:
        A_emb = np.zeros((L, 128), np.float32)
        B_emb = np.zeros((L, 128), np.float32)
        emb_exists = False

    # align crops by sampling indices
    def resample_list(lst, L):
        if not lst:
            return []
        idx = np.linspace(0, len(lst) - 1, num=L).astype(int)
        return [lst[i] for i in idx]

    A_crops = resample_list(cropsA, L)
    B_crops = resample_list(cropsB, L)

    # per-time-step similarities
    emb_sims = np.zeros((L,), np.float32)
    orb_sims = np.zeros((L,), np.float32)

    # compute embedding sims if embeddings exist
    if emb_exists:
        # embeddings are LxD
        emb_sims = np.einsum("ld,ld->l", A_emb, B_emb).astype(np.float32)
        emb_sims = (emb_sims + 1.0) / 2.0  # map to [0,1]

    # compute ORB geometry per aligned crop pair
    for i in range(L):
        a_crop = A_crops[i] if i < len(A_crops) else None
        b_crop = B_crops[i] if i < len(B_crops) else None
        if a_crop is None or b_crop is None:
            orb_sims[i] = 0.0
            continue
        # resize to canonical size for ORB matching
        try:
            ac = cv2.resize(a_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            bc = cv2.resize(b_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            s, inl, good, tot = _orb_match_score(ac, bc)
            orb_sims[i] = s  # already in [0,1]
        except Exception:
            orb_sims[i] = 0.0

    # fuse per-frame: if embeddings exist, weight them 60/40; else use ORB only
    if emb_exists:
        diag01 = 0.6 * emb_sims + 0.4 * orb_sims
    else:
        diag01 = orb_sims

    # center similarity (embedding medoid) only if embeddings exist
    if emb_exists:
        cenA = _representative_center(EA)
        cenB = _representative_center(EB)
        center = (float(np.dot(cenA, cenB)) + 1.0) / 2.0 if (cenA is not None and cenB is not None) else 0.0
    else:
        center = float(np.mean(orb_sims))  # use average geometry as a proxy

    p75 = float(np.percentile(diag01, 75)) if len(diag01) else 0.0
    p50 = float(np.percentile(diag01, 50)) if len(diag01) else 0.0
    std = float(diag01.std()) if len(diag01) else 0.0
    prop = float((diag01 >= 0.65).mean()) if len(diag01) else 0.0

    # weight p75 more heavily since time-aligned consistency matters
    score01 = 0.35 * center + 0.65 * p75
    score_pct = 100.0 * score01

    diag.update({
        "center": center, "p75": p75, "p50": p50, "std": std,
        "prop>=0.65": prop, "score01": score01,
        "diag_sims_sample": diag01.tolist()[:64],
        "orb_sims_sample": orb_sims.tolist()[:64],
        "emb_sims_sample": emb_sims.tolist()[:64] if emb_exists else []
    })

    # Stricter decision rules — require strong agreement across frames & geometry
    if score01 >= 0.92 and prop >= 0.75 and std <= 0.12:
        verdict = "Likely same person"
    elif score01 >= 0.78 and prop >= 0.55:
        verdict = "Uncertain"
    else:
        verdict = "Likely different"

    return float(score_pct), verdict, diag


# ---------- Streamlit page --------------------------------------------------

def run_compare_videos():
    # NOTE: Do NOT call st.set_page_config here; main.py should set it once.
    st.title("Video vs Video — Face Match (Strict)")

    c1, c2 = st.columns(2)
    with c1:
        upA = st.file_uploader("Upload ORIGINAL video", type=["mp4", "mov", "avi", "mpeg", "mkv"], key="vidA")
    with c2:
        upB = st.file_uploader("Upload SUSPECTED video", type=["mp4", "mov", "avi", "mpeg", "mkv"], key="vidB")

    if not upA or not upB:
        st.info("Upload both videos and click Compare.")
        return

    if st.button("Compare"):
        tmpA = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmpA.write(upA.read()); tmpA.flush(); pathA = tmpA.name
        tmpA.close()

        tmpB = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmpB.write(upB.read()); tmpB.flush(); pathB = tmpB.name
        tmpB.close()

        try:
            with st.spinner("Extracting faces & comparing… (this may take a while)"):
                try:
                    score, verdict, diag = video_face_match_strict(pathA, pathB, max_frames=64)
                except RuntimeError as e:
                    st.error(f"Processing error: {e}")
                    if DEBUG:
                        st.info(f"Debug artifacts (if any) saved to: {os.path.join(tempfile.gettempdir(), 'face_debug')}")
                    return

            st.subheader(f"Face Match: {score:.2f}%")
            if verdict.startswith("Likely same"):
                st.success(verdict)
            elif verdict.startswith("Uncertain"):
                st.warning(verdict)
            else:
                st.error(verdict)

            with st.expander("Diagnostics (use when tuning)"):
                st.write(
                    "framesA={fa}, framesB={fb}, meanQ_A={qa:.2f}, meanQ_B={qb:.2f}\n"
                    "center={c:.2f}, p75={p75:.2f}, p50={p50:.2f}, std={sd:.2f}, "
                    "prop>=0.65={pp:.2f}, final score01={sc:.2f}".format(
                        fa=diag["framesA"], fb=diag["framesB"],
                        qa=diag["meanQ_A"], qb=diag["meanQ_B"],
                        c=diag["center"], p75=diag["p75"], p50=diag["p50"],
                        sd=diag["std"], pp=diag["prop>=0.65"], sc=diag["score01"]
                    )
                )
                st.write("Sample per-frame fused similarity (first 64 entries):")
                st.write(np.array(diag.get("diag_sims_sample", [])).tolist())

                if DEBUG:
                    d = os.path.join(tempfile.gettempdir(), "face_debug")
                    st.write(f"Debug crops/embs saved (if any) at `{d}`. Inspect saved images like `*_crop_0.jpg`.")
                    try:
                        files = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg")])
                        if files:
                            st.write("Sample debug crop images:")
                            for p in files[:6]:
                                st.image(p, width=160)
                    except Exception:
                        pass

        finally:
            try: os.unlink(pathA)
            except Exception: pass
            try: os.unlink(pathB)
            except Exception: pass


if __name__ == "__main__":
    run_compare_videos()
