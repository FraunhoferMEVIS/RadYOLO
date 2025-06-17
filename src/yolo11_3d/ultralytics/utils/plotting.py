# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import contextlib
import logging
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, ImageDraw, ImageFont

from . import ops

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class Colors:
    """Ultralytics color palette.

    https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.

    ## Ultralytics Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "Ultralytics Brand Colors"

        For Ultralytics brand colors see [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand). Please use the official Ultralytics colors for all marketing materials.
    """

    def __init__(self) -> None:
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette: list[tuple[int, int, int]] = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n: int = len(self.palette)
        self.pose_palette: npt.NDArray[np.uint8] = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i: int | str, bgr: bool = False) -> tuple[int, int, int]:
        """Converts hex color codes to RGB values."""
        c: tuple[int, int, int] = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple[int, int, int]:
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:
    """Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image or numpy array): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (List[List[int]]): Skeleton structure for keypoints.
        limb_color (List[int]): Color palette for limbs.
        kpt_color (List[int]): Color palette for keypoints.
    """

    def __init__(
        self,
        im: Image.Image | npt.NDArray[np.uint8 | np.float32],
        line_width: int | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        # example: str = "abc",
    ) -> None:
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        # non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        # input_is_pil = isinstance(im, Image.Image)
        # self.pil = pil or input_is_pil
        self.lw: int = line_width or max(
            round(sum(im.size if isinstance(im, Image.Image) else im.shape) / 2 * 0.003), 2
        )
        self.im: Image.Image | npt.NDArray[np.uint8 | np.float32]
        if isinstance(im, Image.Image) or pil:  # use PIL
            self.pil = True
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw: ImageDraw.ImageDraw = ImageDraw.Draw(self.im)
            try:
                # font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            # Deprecation fix for w, h = getsize(string) -> _, _, w, h = getbox(string)
            # if check_version(pil_version, "9.2.0"):
            #     self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # text width, height
            self.getsize: Callable[[str], tuple[int, int] | tuple[float, float]] = (
                lambda x: self.font.getbbox(x)[2:4]
            )
        else:  # use cv2
            self.pil = False
            assert im.data.contiguous, (
                "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            )
            self.im = im if im.flags.writeable else im.copy()
            self.tf: int = max(self.lw - 1, 1)  # font thickness
            self.sf: float = self.lw / 3  # font scale
        # Pose
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[
            [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
        ]
        self.kpt_color = colors.pose_palette[
            [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]
        ]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(
        self,
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> (
        tuple[Literal[104], Literal[31], Literal[17]]
        | tuple[Literal[255], Literal[255], Literal[255]]
        | tuple[int, int, int]
    ):
        """Assign text color based on background color.

        Args:
            color (tuple, optional): The background color of the rectangle for text (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).

        Returns:
            txt_color (tuple): Text color for label
        """
        if color in self.dark_colors:
            return 104, 31, 17
        if color in self.light_colors:
            return 255, 255, 255
        return txt_color

    def circle_label(
        self,
        box: tuple[float, float, float, float],
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
        margin: int = 2,
    ) -> None:
        """Draws a label with a background circle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
        # If label have more than 3 characters, skip other characters, due to circle size
        if len(label) > 3:
            print(
                f"Length of label is {len(label)}, initial 3 label characters will be considered for circle annotation!"
            )
            label = label[:3]

        # Calculate the center of the box
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # Get the text size
        text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[
            0
        ]
        # Calculate the required radius to fit the text with the margin
        required_radius = int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin
        # Draw the circle with the required radius
        cv2.circle(self.im, (x_center, y_center), required_radius, color, -1)
        # Calculate the position for the text
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # Draw the text
        cv2.putText(
            self.im,
            str(label),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def text_label(
        self,
        box: tuple[float, float, float, float],
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
        margin: int = 5,
    ) -> None:
        """Draws a label with a background rectangle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
        # Calculate the center of the bounding box
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # Get the size of the text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.1, self.tf)[0]
        # Calculate the top-left corner of the text (to center it)
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # Calculate the coordinates of the background rectangle
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        # Draw the background rectangle
        cv2.rectangle(self.im, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        # Draw the text on top of the rectangle
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.1,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def box_label(
        self,
        box: tuple[float, float, float, float] | torch.Tensor,
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
        rotated: bool = False,
    ) -> None:
        """Draws a bounding box to image with label.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            rotated (bool, optional): Variable used to check if task is OBB
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil:  # or not is_ascii(label):
            if rotated:
                p1 = box[0]
                self.draw.polygon(
                    [tuple(b) for b in box], width=self.lw, outline=color
                )  # PIL requires tuple box
            else:
                p1 = (box[0], box[1])
                self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.getsize(label)  # text width, height
                outside = p1[1] >= h  # label fits outside box
                if (
                    p1[0] > self.im.size[0] - w
                ):  # size is (w, h), check if label extend beyond right side of image
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (
                        p1[0],
                        p1[1] - h if outside else p1[1],
                        p1[0] + w + 1,
                        p1[1] + 1 if outside else p1[1] + h + 1,
                    ),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text(
                    (p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font
                )
        else:  # cv2
            if rotated:
                p1 = [int(b) for b in box[0]]
                cv2.polylines(
                    self.im, [np.asarray(box, dtype=int)], True, color, self.lw
                )  # cv2 requires nparray box
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[
                    0
                ]  # text width, height
                h += 3  # add pixels to pad text
                outside = p1[1] >= h  # label fits outside box
                if (
                    p1[0] > self.im.shape[1] - w
                ):  # shape is (h, w), check if label extend beyond right side of image
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )

    # def masks(self, masks: torch.Tensor, colors, im_gpu, alpha=0.5, retina_masks=False):
    #     """
    #     Plot masks on image.

    #     Args:
    #         masks (tensor): Predicted masks on cuda, shape: [n, h, w]
    #         colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
    #         im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
    #         alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
    #         retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
    #     """
    #     if self.pil:
    #         # Convert to numpy first
    #         self.im = np.asarray(self.im).copy()
    #     if len(masks) == 0:
    #         self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
    #     if im_gpu.device != masks.device:
    #         im_gpu = im_gpu.to(masks.device)
    #     colors = (
    #         torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0
    #     )  # shape(n,3)
    #     colors = colors[:, None, None]  # shape(n,1,1,3)
    #     masks = masks.unsqueeze(3)  # shape(n,h,w,1)
    #     masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    #     inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    #     mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

    #     im_gpu = im_gpu.flip(dims=[0])  # flip channel
    #     im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
    #     im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
    #     im_mask = im_gpu * 255
    #     im_mask_np = im_mask.byte().cpu().numpy()
    #     self.im[:] = im_mask_np if retina_masks else ops.scale_image(im_mask_np, self.im.shape)
    #     if self.pil:
    #         # Convert im back to PIL and update draw
    #         self.fromarray(self.im)

    # def kpts(
    #     self, kpts, shape=(640, 640), radius=None, kpt_line=True, conf_thres=0.25, kpt_color=None
    # ):
    #     """
    #     Plot keypoints on the image.

    #     Args:
    #         kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
    #         shape (tuple, optional): Image shape (h, w). Defaults to (640, 640).
    #         radius (int, optional): Keypoint radius. Defaults to 5.
    #         kpt_line (bool, optional): Draw lines between keypoints. Defaults to True.
    #         conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
    #         kpt_color (tuple, optional): Keypoint color (B, G, R). Defaults to None.

    #     Note:
    #         - `kpt_line=True` currently only supports human pose plotting.
    #         - Modifies self.im in-place.
    #         - If self.pil is True, converts image to numpy array and back to PIL.
    #     """
    #     radius = radius if radius is not None else self.lw
    #     if self.pil:
    #         # Convert to numpy first
    #         self.im = np.asarray(self.im).copy()
    #     nkpt, ndim = kpts.shape
    #     is_pose = nkpt == 17 and ndim in {2, 3}
    #     kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
    #     for i, k in enumerate(kpts):
    #         color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
    #         x_coord, y_coord = k[0], k[1]
    #         if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
    #             if len(k) == 3:
    #                 conf = k[2]
    #                 if conf < conf_thres:
    #                     continue
    #             cv2.circle(
    #                 self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA
    #             )

    #     if kpt_line:
    #         ndim = kpts.shape[-1]
    #         for i, sk in enumerate(self.skeleton):
    #             pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
    #             pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
    #             if ndim == 3:
    #                 conf1 = kpts[(sk[0] - 1), 2]
    #                 conf2 = kpts[(sk[1] - 1), 2]
    #                 if conf1 < conf_thres or conf2 < conf_thres:
    #                     continue
    #             if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
    #                 continue
    #             if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
    #                 continue
    #             cv2.line(
    #                 self.im,
    #                 pos1,
    #                 pos2,
    #                 kpt_color or self.limb_color[i].tolist(),
    #                 thickness=int(np.ceil(self.lw / 2)),
    #                 lineType=cv2.LINE_AA,
    #             )
    #     if self.pil:
    #         # Convert im back to PIL and update draw
    #         self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width: int = 1) -> None:
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(
        self,
        xy,
        text: str,
        txt_color: tuple[int, int, int] = (255, 255, 255),
        anchor: Literal["top", "bottom"] = "top",
        box_style: bool = False,
    ) -> None:
        """Adds text to an image using PIL or cv2."""
        if anchor == "bottom":  # start y from font bottom
            w, h = self.getsize(text)  # text width, height
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            if "\n" in text:
                lines = text.split("\n")
                _, h = self.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[
                    0
                ]  # text width, height
                h += 3  # add pixels to pad text
                outside = xy[1] >= h  # label fits outside box
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            cv2.putText(
                self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA
            )

    def fromarray(self, im: npt.NDArray[np.uint8 | np.float32]) -> None:
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self) -> npt.NDArray:
        """Return annotated image as array."""
        return np.asarray(self.im)

    def show(self, title: str | None = None) -> None:
        """Show the annotated image."""
        im = Image.fromarray(
            np.asarray(self.im)[..., ::-1]
        )  # Convert numpy array to PIL Image with RGB to BGR
        # if IS_COLAB or IS_KAGGLE:  # can not use IS_JUPYTER as will run for all ipython environments
        #     try:
        #         display(im)  # noqa - display() function only available in ipython environments
        #     except ImportError as e:
        #         LOGGER.warning(f"Unable to display image in Jupyter notebooks: {e}")
        # else:
        im.show(title=title)

    def save(self, filename: str = "image.jpg") -> None:
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))

    def get_bbox_dimension(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float]:
        """Calculate the area of a bounding box.

        Args:
            bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

        Returns:
            width (float): Width of the bounding box.
            height (float): Height of the bounding box.
            area (float): Area enclosed by the bounding box.
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height

    # def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
    #     """Draw region line.

    #     Args:
    #         reg_pts (list): Region Points (for line 2 points, for region 4 points)
    #         color (tuple): Region Color value
    #         thickness (int): Region area thickness value
    #     """
    #     cv2.polylines(
    #         self.im,
    #         [np.array(reg_pts, dtype=np.int32)],
    #         isClosed=True,
    #         color=color,
    #         thickness=thickness,
    #     )

    #     # Draw small circles at the corner points
    #     for point in reg_pts:
    #         cv2.circle(
    #             self.im, (point[0], point[1]), thickness * 2, color, -1
    #         )  # -1 fills the circle


def plot_labels(
    boxes: npt.NDArray,
    cls: npt.NDArray,
    names: dict[int, str],
    save_dir: Path = Path(),
    # on_plot=None,
):
    """Plot training labels including class histograms and box statistics."""
    import pandas  # scope for faster 'import ultralytics'
    import seaborn  # scope for faster 'import ultralytics'

    # Filter matplotlib>=3.7.2 warning and Seaborn use_inf and is_categorical FutureWarnings
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="The figure layout has changed to tight"
    )
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Plot dataset labels
    LOGGER.info("Plotting labels to %s... ", save_dir / "labels.jpg")
    nc = int(cls.max() + 1)  # number of classes
    boxes = boxes[:1000000]  # limit to 1M boxes
    x = pandas.DataFrame(boxes, columns=["x", "y", "width", "height"])

    # Seaborn correlogram
    seaborn.pairplot(
        x,
        corner=True,
        diag_kind="auto",
        kind="hist",
        diag_kws={"bins": 50},
        plot_kws={"pmax": 0.9},
    )
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # Matplotlib labels
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("classes")
    seaborn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    seaborn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # Rectangles
    boxes[:, 0:2] = 0.5  # center
    boxes = ops.xywh2xyxy(boxes) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for cls, box in zip(cls[:500], boxes[:500], strict=True):
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    # if on_plot:
    #     on_plot(fname)


def plot_images(
    images: torch.Tensor | np.ndarray,
    batch_idx: torch.Tensor | np.ndarray,
    cls: torch.Tensor | np.ndarray,
    bboxes: torch.Tensor | np.ndarray = np.zeros(0, dtype=np.float32),
    confs: torch.Tensor | np.ndarray | None = None,
    masks: torch.Tensor | np.ndarray = np.zeros(0, dtype=np.uint8),
    kpts: torch.Tensor | np.ndarray = np.zeros((0, 51), dtype=np.float32),
    paths: list[str] | None = None,
    fname: str = "images.jpg",
    names: dict[int, str] | None = None,
    # on_plot: Callable = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.25,
) -> np.ndarray | None:
    """Plot image grid with labels, bounding boxes, masks, and keypoints.

    Args:
        images: Batch of images to plot. Shape: (batch_size, channels, height, width).
        batch_idx: Batch indices for each detection. Shape: (num_detections,).
        cls: Class labels for each detection. Shape: (num_detections,).
        bboxes: Bounding boxes for each detection. Shape: (num_detections, 4) or (num_detections, 5) for rotated boxes.
        confs: Confidence scores for each detection. Shape: (num_detections,).
        masks: Instance segmentation masks. Shape: (num_detections, height, width) or (1, height, width).
        kpts: Keypoints for each detection. Shape: (num_detections, 51).
        paths: List of file paths for each image in the batch.
        fname: Output filename for the plotted image grid.
        names: Dictionary mapping class indices to class names.
        on_plot: Optional callback function to be called after saving the plot.
        max_size: Maximum size of the output image grid.
        max_subplots: Maximum number of subplots in the image grid.
        save: Whether to save the plotted image grid to a file.
        conf_thres: Confidence threshold for displaying detections.

    Returns:
        np.ndarray: Plotted image grid as a numpy array if save is False, None otherwise.

    Note:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(
        mosaic,
        line_width=round(fs / 10),
        font_size=fs,
        pil=True,
        # example=names,
    )
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text(
                (x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220)
            )  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None

            if len(bboxes):
                boxes = bboxes[idx]
                conf = (
                    confs[idx] if confs is not None else None
                )  # check for confidence presence (label vs pred)
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                        boxes[..., [0, 2]] *= w  # scale to pixels
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5  # xywhr
                # boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                if is_obb:
                    msg = "Not implemented for OBB"
                    raise NotImplementedError(msg)
                boxes = ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color, rotated=is_obb)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f"{c}", txt_color=color, box_style=True)

            # Plot keypoints
            if len(kpts):
                msg = "Keypoints are not supported."
                raise NotImplementedError(msg)
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if (
                        kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01
                    ):  # if normalized with tolerance .01
                        kpts_[..., 0] *= w  # scale to pixels
                        kpts_[..., 1] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)

            # Plot masks
            if len(masks):
                if idx.shape[0] == masks.shape[0]:  # overlap_masks=False
                    image_masks = masks[idx]
                else:  # overlap_masks=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)

                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # save
    # if on_plot:
    #     on_plot(fname)
    return None


def plot_results(
    file: Path | str | None = "path/to/results.csv",
    dir: Path | str = "",
    # segment: bool = False,
    # pose: bool = False,
    # classify: bool = False,
    # on_plot: bool = None,
):
    """Plot training results from a results CSV file.

    The function supports various types of data including segmentation, pose estimation, and
    classification. Plots are saved as 'results.png' in the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to
            'path/to/results.csv'.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided.
            Defaults to ''.
        segment (bool, optional): Flag to indicate if the data is for segmentation.
            Defaults to False.
        pose (bool, optional): Flag to indicate if the data is for pose estimation.
            Defaults to False.
        classify (bool, optional): Flag to indicate if the data is for classification.
            Defaults to False.
        on_plot (callable, optional): Callback function to be executed after plotting. Takes
            filename as an argument. Defaults to None.

    Example:
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results("path/to/results.csv", segment=True)
        ```
    """
    import pandas as pd  # scope for faster 'import ultralytics'
    # from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    # if classify:
    #     fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
    #     index = [2, 5, 3, 4]
    # elif segment:
    #     fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
    #     index = [2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 8, 9, 12, 13]
    # elif pose:
    #     fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
    #     index = [2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 9, 10, 13, 14]
    # else:
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    index = [2, 3, 4, 5, 6, 9, 10, 11, 7, 8]

    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert files, f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate(index):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(
                    x, y, marker=".", label=f.stem, linewidth=2, markersize=8
                )  # actual results
                # ax[i].plot(
                #     x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2
                # )  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in {8, 9, 10}:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.warning("WARNING: Plotting error for %s: %s", f, e)
        else:
            ax[1].legend()
            fname = save_dir / "results.png"
            fig.savefig(fname, dpi=200)
            # if on_plot:
            #     on_plot(fname)
        finally:
            plt.close()


def output_to_target(
    output: list[torch.Tensor], max_det: int = 300
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets: list[torch.Tensor] = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, ops.xyxy2xywh(box), conf), 1))
    targets_ = torch.cat(targets, 0).numpy()
    return targets_[:, 0], targets_[:, 1], targets_[:, 2:-1], targets_[:, -1]


def output_to_target_3d(
    output: list[torch.Tensor], max_det: int = 300, axis: int = 2
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting.

    Results are flattened to 2d for easier visualization.
    """
    idx = [i for i in range(6) if i % 3 != axis]
    targets: list[torch.Tensor] = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :8].cpu().split((6, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, ops.xyxy2xywh(box[:, idx]), conf), 1))
    targets_ = torch.cat(targets, 0).numpy()
    return targets_[:, 0], targets_[:, 1], targets_[:, 2:-1], targets_[:, -1]
