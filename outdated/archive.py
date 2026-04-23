"""
Old code that may become useful at some point in the future...
"""


"""
def load_image(path: str) -> np.ndarray:
    \"""Load any image (including AVIF) as a BGR numpy array.\"""
    pil_img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

SOBEL_KSIZE       = 3        # Sobel kernel size
def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
"""