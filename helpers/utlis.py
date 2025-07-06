from io import BytesIO
from PIL import Image
import numpy as np

def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))
    return np.expand_dims(image, axis=0).astype(np.float32)