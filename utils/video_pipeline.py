import numpy as np
from utils.video_face import video_face_embeddings, temporal_flicker_score

def analyze_video(path):
    """
    Returns:
    - final_score (0..1)
    - details dict
    """

    embeds, sharp = video_face_embeddings(path, max_frames=120, step=3)

    if len(embeds) == 0:
        return 0.0, {"error": "no_face"}

    # Flicker score (0–100 → normalize)
    flicker = temporal_flicker_score(embeds, sharp)
    flicker_norm = np.clip(flicker / 100.0, 0, 1)

    # Stability (important)
    if len(embeds) > 1:
        diffs = []
        for i in range(1, len(embeds)):
            d = np.linalg.norm(embeds[i] - embeds[i-1])
            diffs.append(d)
        stability = np.mean(diffs)
    else:
        stability = 0.0

    stability_norm = np.clip(stability, 0, 1)

    # Final fusion (balanced)
    final_score = 0.7 * flicker_norm + 0.3 * stability_norm

    return float(final_score), {
        "flicker": float(flicker),
        "stability": float(stability),
        "frames": int(len(embeds))
    }