from pathlib import Path

import torch
from fire import Fire
from loguru import logger
from PIL import Image

from east.detector import Detector
from east.drawer import Box, Drawer

CHECKPOINT_PATH = Path("./weights/east_vgg16.pth")


def main(input: str, output: str = "./outputs", pattern: str = "*"):
    input_dir = Path(input)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"output directory was created: {output_dir.absolute()}")

    detector = Detector(
        checkpoint_path=CHECKPOINT_PATH,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        logger=logger,
    )

    for p in sorted(input_dir.glob(pattern)):
        logger.debug(f"input: {p}")

        image = Image.open(p)
        boxes = detector.detect(image)

        drawer = Drawer(image)
        drawer.draw(boxes=[Box.from_ndarray(box) for box in boxes])

        output_path = output_dir.joinpath(p.name)
        drawer.save(output_path)
        logger.debug(f"output image was saved to {output_path.absolute()}")


if __name__ == "__main__":
    Fire(main)
