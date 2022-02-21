from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Box:
    p1: tuple[int, int]
    p2: tuple[int, int]
    p3: tuple[int, int]
    p4: tuple[int, int]

    def as_xy(self) -> list[tuple[int, int]]:
        return [self.p1, self.p2, self.p3, self.p4]

    @staticmethod
    def from_ndarray(box: np.ndarray) -> "Box":
        return Box(
            p1=(int(box[0]), int(box[1])),
            p2=(int(box[2]), int(box[3])),
            p3=(int(box[4]), int(box[5])),
            p4=(int(box[6]), int(box[7])),
        )


class Drawer:
    def __init__(self, image: Image.Image):
        self._image = image
        self._drawer = ImageDraw.Draw(image)

    def _draw_box(self, box: Box):
        self._drawer.polygon(box.as_xy(), width=3)

    def draw(self, boxes: list[Box]):
        for box in boxes:
            self._draw_box(box)

    def save(self, path: Path):
        self._image.save(path)
