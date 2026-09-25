"""
Post-processing for fault predictions to boost Final Round discovery.
- Low threshold for high recall (beta=0.8)
- Morphological closing to connect segments
- Frangi filter for line enhancement
- Skeletonization optional
- Length filtering

Sources:
- Mattéo et al 2021 automatic fault mapping: https://doi.org/10.1029/2020JB021269
- Hermant et al 2025 deep learning for Quaternary faults: https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf
"""

import numpy as np
from skimage.morphology import skeletonize, closing, remove_small_objects
try:                                    # skimage >= 0.25 (square() deprecated there, removed in 0.27)
    from skimage.morphology import footprint_rectangle as _rect
except ImportError:                     # pragma: no cover - older skimage
    from skimage.morphology import square as _square_impl
    _rect = None
from skimage.filters import frangi
from scipy.ndimage import binary_dilation

def apply_threshold(pred, thresh=0.15):
    return (pred > thresh).astype(np.uint8)

def morphological_close(binary, kernel_size=3):
    """Square structuring element, version-tolerant (footprint_rectangle on >=0.25)."""
    k = int(kernel_size)
    fp = _rect((k, k)) if _rect is not None else _square_impl(k)
    return closing(binary, fp)

def frangi_enhance(pred, scale_range=(1,10), scale_step=2, weight=0.4):
    """Vesselness via Frangi. Version-tolerant: scikit-image >=0.19 renamed
    scale_range->sigmas and beta1/beta2->beta/gamma (old kwargs REMOVED in 0.21+
    and previously failed silently here, disabling the filter)."""
    import inspect
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    sigmas = np.arange(scale_range[0], scale_range[1] + 1, scale_step)
    try:
        if "sigmas" in inspect.signature(frangi).parameters:
            vessel = frangi(pred, sigmas=sigmas, beta=0.5, gamma=15)
        else:  # very old scikit-image
            vessel = frangi(pred, scale_range=scale_range, scale_step=scale_step, beta1=0.5, beta2=15)
        v = vessel
        if v.max() > v.min():
            v = (v - v.min()) / (v.max() - v.min())
        enhanced = (1.0 - weight) * pred + weight * v
        return np.clip(enhanced, 0, 1)
    except Exception as e:
        print(f"Frangi failed: {e}, returning original")
        return pred

def connect_faults(binary, iterations=1):
    # Dilate then skeletonize to connect
    dilated = binary_dilation(binary, iterations=iterations)
    # Could skeletonize
    # skel = skeletonize(dilated)
    return dilated.astype(np.uint8)

def filter_small_faults(binary, min_length=5):
    """Drop connected components with fewer than `min_length` pixels.

    skimage 0.26 renamed `min_size` (strictly smaller) to `max_size` (smaller *or equal*);
    passing the old name emits a FutureWarning, so try the new keyword first.  A bare
    `except:` here used to swallow real errors and return the unfiltered mask silently -
    now only a genuine signature TypeError falls back.
    """
    mask = binary.astype(bool)
    try:
        cleaned = remove_small_objects(mask, max_size=min_length - 1)
    except TypeError:
        cleaned = remove_small_objects(mask, min_size=min_length)
    return cleaned.astype(np.uint8)

def postprocess_pipeline(pred, config):
    """
    pred: (H,W) float [0,1]
    config: dict from config.yaml postprocess
    Returns processed float map and binary map
    """
    # Frangi
    if config.get("frangi_filter", False):
        scale_range = tuple(config.get("frangi_scale_range", [1,10]))
        pred_enh = frangi_enhance(pred, scale_range=scale_range,
                                  weight=float(config.get("frangi_weight", 0.4)))
    else:
        pred_enh = pred

    # Threshold (null in config = keep a pure probability field, which is what the
    # submission format asks for; the binary map is then only used for diagnostics)
    thresh = config.get("threshold", 0.15)
    if thresh is None:
        thresh = 0.5
    binary = apply_threshold(pred_enh, thresh=thresh)

    # Closing
    if config.get("morphological_closing", True):
        k = config.get("closing_kernel", 3)
        binary = morphological_close(binary, kernel_size=k)

    # Connect
    binary = connect_faults(binary, iterations=1)

    # Filter small
    min_len = config.get("min_fault_length", 5)
    binary = filter_small_faults(binary, min_length=min_len)

    # Optionally skeletonize for final? But submission expects probabilities, not binary.
    # So we keep float enhanced for submission, but binary for analysis.

    # For final submission, we want probabilities: use enhanced float, but boost where binary is 1
    # e.g., keep enhanced but ensure binary regions have at least thresh
    final_prob = pred_enh.copy()
    if config.get("boost_to_threshold", True) and thresh < 0.5:
        final_prob[binary > 0] = np.maximum(final_prob[binary > 0], thresh)

    return final_prob, binary
