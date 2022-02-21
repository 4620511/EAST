from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from streamlit.uploaded_file_manager import UploadedFile

from east.detector import Detector
from east.drawer import Box, Drawer

CHECKPOINT_PATH = Path("./weights/east_vgg16.pth")


def get_uploaded_file() -> Optional[UploadedFile]:
    return st.file_uploader("Upload image", type=["jpg", "png"], accept_multiple_files=False)


def detect(detector: Detector, image: Image.Image):
    boxes = detector.detect(image)

    drawer = Drawer(image)
    drawer.draw(boxes=[Box.from_ndarray(box) for box in boxes])

    st.image(image)

    table = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "score"])
    st.table(table)


def main():
    st.balloons()
    st.title("EAST Text Detection")

    detector = Detector(CHECKPOINT_PATH, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    file = get_uploaded_file()
    if file is None:
        return

    image = Image.open(BytesIO(file.getvalue()))
    st.image(image)

    detect(detector, image)


if __name__ == "__main__":
    main()
