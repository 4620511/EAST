import math
from pathlib import Path

import lanms
import loguru
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from east.model import EAST


class Detector:
    def __init__(self, checkpoint_path: Path, device: torch.device, logger: loguru._Logger = None):  # type: ignore
        self._device = device
        self._logger = logger if logger is not None else loguru.logger

        self._model = EAST(pretrained=False).to(device)
        self._model.load_state_dict(torch.load(checkpoint_path))
        self._logger.debug("state dict was loaded into EAST model")

    @staticmethod
    def _make_divisible(image: Image.Image, div: int = 32) -> tuple[Image.Image, float, float]:
        w, h = image.size
        resize_w, resize_h = w, h

        resize_h = resize_h if resize_h % div == 0 else int(resize_h / div) * div
        resize_w = resize_w if resize_w % div == 0 else int(resize_w / div) * div

        image = image.resize((resize_w, resize_h), Image.BILINEAR)

        ratio_h = resize_h / h
        ratio_w = resize_w / w

        return image, ratio_h, ratio_w

    @staticmethod
    def _pil2tensor(image: Image.Image) -> torch.Tensor:
        t = T.Compose([T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        return t(image).unsqueeze(0)  # type: ignore

    @staticmethod
    def _get_rotate_mat(theta: float) -> np.ndarray:
        return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    @staticmethod
    def _is_valid_poly(res: np.ndarray, score_shape: tuple[int, int], scale: int) -> bool:
        """Check if the poly in image scope.

        Args:
            res (np.ndarray): restored poly in original image
            score_shape (tuple[int, int]) score map shape
            scale (int): image / feature map

        Returns:
            bool: whether the poly is valid
        """
        cnt = 0
        for i in range(res.shape[1]):
            if (
                res[0, i] < 0
                or res[0, i] >= score_shape[1] * scale
                or res[1, i] < 0
                or res[1, i] >= score_shape[0] * scale
            ):
                cnt += 1

        return cnt <= 1

    def _restore_polys(
        self, valid_pos: np.ndarray, valid_geo: np.ndarray, score_shape: tuple[int, int], scale=4
    ) -> tuple[np.ndarray, list[int]]:
        """Restore polys from feature map in given positions.

        Args:
            valid_pos (np.ndarray): potential text positions (n,2)
            valid_geo (np.ndarray): geometry in `valid_pos` (5,n)
            score_shape: shape of score map
            scale: image / feature map

        Returns:
            tuple[np.ndarray, list[int]]: restored polys (n,8) and list of index
        """
        polys = []
        indices = []
        valid_pos *= scale
        d = valid_geo[:4, :]  # 4 x n
        angle = valid_geo[4, :]  # n

        for i in range(valid_pos.shape[0]):
            x = valid_pos[i, 0]
            y = valid_pos[i, 1]
            y_min = y - d[0, i]
            y_max = y + d[1, i]
            x_min = x - d[2, i]
            x_max = x + d[3, i]
            rotate_mat = self._get_rotate_mat(-angle[i])

            temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
            temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
            coordidates = np.concatenate((temp_x, temp_y), axis=0)
            res = np.dot(rotate_mat, coordidates)
            res[0, :] += x
            res[1, :] += y

            if self._is_valid_poly(res, score_shape, scale):
                indices.append(i)
                polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])

        return np.array(polys), indices

    def _get_boxes(
        self, score: np.ndarray, geo: np.ndarray, score_thresh: float = 0.9, nms_thresh: float = 0.2
    ) -> np.ndarray:
        """
        Args:
            score (np.ndarray): score map (1,row,col)
            geo (np.ndarray): geo map (5,row,col)
            score_thresh (float): threshold to segment score map
            nms_thresh (float): threshold in nms

        Returns:
            np.ndarray: final polys (n,9)
        """
        score = score[0, :, :]
        xy_text = np.argwhere(score > score_thresh)  # n x 2, [r, c]
        if xy_text.size == 0:
            return np.array([])

        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
        valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
        polys_restored, indices = self._restore_polys(valid_pos, valid_geo, score.shape)

        if polys_restored.size == 0:
            return np.array([])

        boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = polys_restored
        boxes[:, 8] = score[xy_text[indices, 0], xy_text[indices, 1]]
        boxes = lanms.merge_quadrangle_n9(boxes.astype("float32"), nms_thresh)

        return boxes

    @staticmethod
    def _adjust_ratio(boxes: np.ndarray, ratio_w: float, ratio_h: float) -> np.ndarray:
        """Refine boxes.

        Args:
            boxes (np.ndarray): detected polys (n,9)
            ratio_w (float): ratio of width
            ratio_h (float): ratio of height

        Returns:
            np.ndarray: adjusted boxes (n,9)
        """
        boxes[:, [0, 2, 4, 6]] /= ratio_w  # type: ignore
        boxes[:, [1, 3, 5, 7]] /= ratio_h  # type: ignore

        return np.around(boxes)

    @torch.no_grad()
    def detect(self, image: Image.Image) -> np.ndarray:
        self._model.eval()

        image, ratio_h, ratio_w = self._make_divisible(image, 32)
        score, geo = self._model(self._pil2tensor(image).to(self._device))

        boxes = self._get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
        self._logger.debug(f"found {boxes.size // 9 } boxes")

        if boxes.size == 0:
            return np.array([])

        return self._adjust_ratio(boxes, ratio_w, ratio_h)
