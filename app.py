"""ON99 DTG / DTF 白墨通道生成工具。

從透明 PNG 自動產生 PrintEXP / Maintop 可匯入的 TIFF：
CMYK 合成色 + 名為 W1 的 Photoshop Spot 專色通道（白墨）。
右側通道面板可預覽產出，也可拖入既有 TIFF 檢查是否已有 W1。

啟動：
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import base64
import html
import io
import struct
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from tifffile import TiffFile, imwrite

from psdtags import (
    PsdBytesBlock,
    PsdFormat,
    PsdPascalStringsBlock,
    PsdResourceId,
    PsdStringsBlock,
    TiffImageResources,
)

WHITE_CHANNEL_NAME = "W1"
DEFAULT_CHOKE_PX = 2
DEFAULT_DPI = 300
DEFAULT_BLACK_CUTOFF = 18
DEFAULT_BLACK_FEATHER = 28
MAX_PREVIEW_EDGE = 640
WHITE_NAME_ALIASES = {
    "w1",
    "white",
    "white_ink",
    "whiteink",
    "spotwhite",
    "whiteinkchannel",
    "spot1",
    "spot_1",
}

ChannelPolarity = Literal["white_prints", "black_prints"]


@dataclass(frozen=True)
class ProcessResult:
    rgb: np.ndarray
    alpha: np.ndarray
    coverage: np.ndarray
    white: np.ndarray
    dpi: float
    source_name: str


@dataclass
class ChannelView:
    name: str
    image: np.ndarray
    shortcut: str = ""
    is_white: bool = False
    is_new: bool = False
    subtitle: str = ""


@dataclass
class ChannelDocument:
    filename: str
    width: int
    height: int
    dpi: float
    sample_count: int
    has_white: bool
    status: Literal["generated", "has_white", "missing_white", "empty"]
    channels: list[ChannelView] = field(default_factory=list)
    extra_names: list[str] = field(default_factory=list)
    note: str = ""
    rgb: np.ndarray | None = None
    white: np.ndarray | None = None
    alpha: np.ndarray | None = None
    coverage: np.ndarray | None = None


def _is_white_name(name: str) -> bool:
    compact = (
        name.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
        .replace(".", "")
    )
    return compact in WHITE_NAME_ALIASES


@st.cache_data(show_spinner=False)
def load_rgba(file_bytes: bytes, filename: str) -> tuple[np.ndarray, float]:
    with Image.open(io.BytesIO(file_bytes)) as img:
        dpi_info = img.info.get("dpi")
        if isinstance(dpi_info, tuple) and dpi_info[0]:
            dpi = float(dpi_info[0])
        else:
            dpi = float(DEFAULT_DPI)
        rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"無法讀取為 RGBA：{filename}")
    return rgba, dpi


def flatten_rgb(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = rgba[..., :3].copy()
    alpha = rgba[..., 3].copy()
    rgb[alpha == 0] = 0
    return rgb, alpha


def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    span = max(high - low, 1.0)
    t = np.clip((value - low) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ink_coverage(
    rgb: np.ndarray,
    alpha: np.ndarray,
    black_cutoff: int = DEFAULT_BLACK_CUTOFF,
    black_feather: int = DEFAULT_BLACK_FEATHER,
) -> np.ndarray:
    """Grayscale 0–255 white-ink amount. Keeps distressed holes; skips near-black."""
    a = alpha.astype(np.float32) / 255.0
    peak = rgb.max(axis=2).astype(np.float32)
    color = _smoothstep(peak, float(black_cutoff), float(black_cutoff + black_feather))
    return np.clip(a * color * 255.0, 0, 255).astype(np.uint8)


def choke_grayscale(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Shrink bright areas by N pixels. Keeps 0–255; never binarizes."""
    mask = mask.astype(np.uint8, copy=False)
    if pixels <= 0:
        return mask
    size = 2 * int(pixels) + 1
    try:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.erode(mask, kernel, iterations=1)
    except Exception:
        return np.array(Image.fromarray(mask).filter(ImageFilter.MinFilter(size)))


def build_white_channel(
    rgb: np.ndarray,
    alpha: np.ndarray,
    choke_px: int,
    black_cutoff: int,
    black_feather: int,
    polarity: ChannelPolarity,
) -> tuple[np.ndarray, np.ndarray]:
    coverage = ink_coverage(rgb, alpha, black_cutoff, black_feather)
    white = choke_grayscale(coverage, choke_px)
    if polarity == "black_prints":
        white = (255 - white).astype(np.uint8)
    return coverage, white


def process_artwork(
    file_bytes: bytes,
    filename: str,
    choke_px: int,
    black_cutoff: int,
    black_feather: int,
    polarity: ChannelPolarity,
    dpi_override: float | None = None,
) -> ProcessResult:
    rgba, dpi = load_rgba(file_bytes, filename)
    rgb, alpha = flatten_rgb(rgba)
    coverage, white = build_white_channel(
        rgb, alpha, choke_px, black_cutoff, black_feather, polarity
    )
    return ProcessResult(
        rgb=rgb,
        alpha=alpha,
        coverage=coverage,
        white=white,
        dpi=float(dpi_override or dpi),
        source_name=filename,
    )


def rgb_alpha_to_cmyk(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Convert RGB(+alpha) to CMYK ink amounts. Transparent pixels stay ink-free."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb 必須是 HxWx3")
    if alpha.shape != rgb.shape[:2]:
        raise ValueError("alpha 尺寸必須與 RGB 相同")

    r = rgb[:, :, 0].astype(np.float32) / 255.0
    g = rgb[:, :, 1].astype(np.float32) / 255.0
    b = rgb[:, :, 2].astype(np.float32) / 255.0
    a = alpha.astype(np.float32) / 255.0

    k = 1.0 - np.maximum(np.maximum(r, g), b)
    denom = np.maximum(1.0 - k, 1e-6)
    c = (1.0 - r - k) / denom
    m = (1.0 - g - k) / denom
    y = (1.0 - b - k) / denom
    pure_black = k >= (1.0 - 1e-6)
    c = np.where(pure_black, 0.0, c)
    m = np.where(pure_black, 0.0, m)
    y = np.where(pure_black, 0.0, y)

    cmyk = np.stack([c, m, y, k], axis=-1) * a[..., None]
    cmyk[alpha == 0] = 0.0
    return np.clip(np.rint(cmyk * 255.0), 0, 255).astype(np.uint8)


def cmyk_to_rgb_preview(cmyk: np.ndarray) -> np.ndarray:
    """Approximate CMYK → RGB preview for the channel panel."""
    c = cmyk[:, :, 0].astype(np.float32) / 255.0
    m = cmyk[:, :, 1].astype(np.float32) / 255.0
    y = cmyk[:, :, 2].astype(np.float32) / 255.0
    k = cmyk[:, :, 3].astype(np.float32) / 255.0
    r = 255.0 * (1.0 - c) * (1.0 - k)
    g = 255.0 * (1.0 - m) * (1.0 - k)
    b = 255.0 * (1.0 - y) * (1.0 - k)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def _spot_display_info(count: int) -> bytes:
    """Photoshop DisplayInfo (1077): kind=2 marks each extra channel as Spot."""
    parts = [struct.pack(">I", 1)]
    for _ in range(count):
        # HSB overlay similar to Photoshop default spot preview.
        parts.append(struct.pack(">h4HHb", 1, 0, 65535, 65535, 0, 0, 2))
    return b"".join(parts)


def _alternate_spot_colors(count: int, start_id: int = 3) -> bytes:
    """Photoshop Alternate Spot Colors (1067) with Lab placeholders."""
    parts = [struct.pack(">HH", 1, count)]
    for index in range(count):
        channel_id = start_id + index
        # Lab values copied from a Photoshop-saved W1 spot TIFF.
        parts.append(struct.pack(">Ih4H", channel_id, 7, 0x1535, 0x1F90, 0x1B4E, 0))
    return b"".join(parts)


def _photoshop_spot_tags(channel_names: list[str]) -> tuple:
    """ImageResources so PrintEXP / Photoshop treat extras as named Spot channels."""
    if not channel_names:
        raise ValueError("至少需要一個專色通道名稱")
    count = len(channel_names)
    start_id = 3
    alpha_ids = b"".join(struct.pack(">I", start_id + i) for i in range(count))
    resources = TiffImageResources(
        name=f"{channel_names[0]}.tif",
        psdformat=PsdFormat.BE32BIT,
        blocks=[
            PsdPascalStringsBlock(
                resourceid=PsdResourceId.ALPHA_NAMES_PASCAL,
                values=channel_names,
            ),
            PsdStringsBlock(
                resourceid=PsdResourceId.ALPHA_NAMES_UNICODE,
                values=channel_names,
            ),
            PsdBytesBlock(
                resourceid=PsdResourceId.DISPLAY_INFO,
                value=_spot_display_info(count),
            ),
            PsdBytesBlock(
                resourceid=PsdResourceId.ALPHA_IDENTIFIERS,
                value=alpha_ids,
            ),
            PsdBytesBlock(
                resourceid=PsdResourceId.ALTERNATE_SPOT_COLORS,
                value=_alternate_spot_colors(count, start_id=start_id),
            ),
        ],
    )
    return resources.tifftag()


def write_tiff_with_white(
    rgb: np.ndarray,
    white: np.ndarray,
    dpi: float,
    channel_name: str = WHITE_CHANNEL_NAME,
    compression: str = "none",
    alpha: np.ndarray | None = None,
) -> bytes:
    """Lossless CMYK TIFF + Spot channel (default W1) for PrintEXP / Maintop."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb 必須是 HxWx3")
    if white.shape != rgb.shape[:2]:
        raise ValueError("white 通道尺寸必須與 RGB 相同")
    if alpha is None:
        alpha = np.full(white.shape, 255, dtype=np.uint8)
    elif alpha.shape != white.shape:
        raise ValueError("alpha 尺寸必須與 white 相同")

    cmyk = rgb_alpha_to_cmyk(rgb, alpha)
    stacked = np.concatenate([cmyk, white[..., None]], axis=2).astype(np.uint8)
    buf = io.BytesIO()
    options = dict(
        photometric="separated",
        extrasamples=[0],
        planarconfig="contig",
        metadata=None,
        resolution=(float(dpi), float(dpi)),
        resolutionunit="inch",
        extratags=[_photoshop_spot_tags([channel_name])],
    )
    if compression == "zip":
        options["compression"] = "adobe_deflate"
    imwrite(buf, stacked, **options)
    return buf.getvalue()


def _pascal_string(text: str, pad: int = 2) -> bytes:
    raw = text.encode("ascii", "replace")[:255]
    data = bytes([len(raw)]) + raw
    if pad and len(data) % pad:
        data += b"\x00" * (pad - (len(data) % pad))
    return data


def _psd_resource(resource_id: int, data: bytes, name: str = "") -> bytes:
    name_bytes = _pascal_string(name, pad=2)
    size = len(data)
    if len(data) % 2:
        data = data + b"\x00"
    return b"8BIM" + struct.pack(">H", resource_id) + name_bytes + struct.pack(">I", size) + data


def _psd_resolution(dpi: float) -> bytes:
    fixed = int(round(dpi * 65536))
    return struct.pack(">IHH IHH", fixed, 1, 1, fixed, 1, 1)


def _psd_unicode_names(names: list[str]) -> bytes:
    out = bytearray()
    for name in names:
        chars = name + "\x00"
        out += struct.pack(">I", len(chars))
        out += chars.encode("utf-16-be")
    return bytes(out)


def write_psd_with_white(
    rgb: np.ndarray,
    white: np.ndarray,
    dpi: float,
    channel_name: str = WHITE_CHANNEL_NAME,
    alpha: np.ndarray | None = None,
) -> bytes:
    """CMYK + named spot channel PSD (PrintEXP-compatible channel layout)."""
    height, width = white.shape
    if alpha is None:
        alpha = np.full(white.shape, 255, dtype=np.uint8)
    cmyk = rgb_alpha_to_cmyk(rgb, alpha)
    header = b"".join(
        [
            b"8BPS",
            struct.pack(">H", 1),
            b"\x00" * 6,
            struct.pack(">H", 5),  # CMYK + spot
            struct.pack(">I", height),
            struct.pack(">I", width),
            struct.pack(">H", 8),
            struct.pack(">H", 4),  # CMYK mode
        ]
    )
    alpha_names = _pascal_string(channel_name, pad=1)
    resources = b"".join(
        [
            _psd_resource(1005, _psd_resolution(dpi)),
            _psd_resource(1006, alpha_names),
            _psd_resource(1045, _psd_unicode_names([channel_name])),
            _psd_resource(1053, struct.pack(">I", 3)),
        ]
    )
    image_data = b"".join(
        [
            struct.pack(">H", 0),
            cmyk[:, :, 0].tobytes(),
            cmyk[:, :, 1].tobytes(),
            cmyk[:, :, 2].tobytes(),
            cmyk[:, :, 3].tobytes(),
            white.tobytes(),
        ]
    )
    return b"".join(
        [
            header,
            struct.pack(">I", 0),
            struct.pack(">I", len(resources)),
            resources,
            struct.pack(">I", 0),
            image_data,
        ]
    )


def downscale_preview(*arrays: np.ndarray, max_edge: int = MAX_PREVIEW_EDGE) -> list[np.ndarray]:
    h, w = arrays[0].shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale >= 1:
        return [a.copy() for a in arrays]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    def _resize(arr: np.ndarray) -> np.ndarray:
        img = Image.fromarray(arr)
        resample = Image.Resampling.NEAREST if arr.ndim == 2 else Image.Resampling.BILINEAR
        return np.array(img.resize((new_w, new_h), resample))

    return [_resize(a) for a in arrays]


def checkerboard(h: int, w: int, cell: int = 12) -> np.ndarray:
    yy, xx = np.indices((h, w))
    field = ((xx // cell) + (yy // cell)) % 2
    light = np.where(field, 214, 248).astype(np.uint8)
    return np.dstack([light, light, light])


def composite_rgba_preview(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    bg = checkerboard(*rgb.shape[:2])
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    return (rgb.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def simulate_dtg(
    rgb: np.ndarray,
    alpha: np.ndarray,
    white: np.ndarray,
    garment_rgb: tuple[int, int, int] = (28, 28, 30),
    polarity: ChannelPolarity = "white_prints",
) -> np.ndarray:
    underbase = white.astype(np.float32)
    if polarity == "black_prints":
        underbase = 255.0 - underbase
    under = (underbase / 255.0)[..., None]
    color_a = (alpha.astype(np.float32) / 255.0)[..., None]
    garment = np.array(garment_rgb, dtype=np.float32)
    out = np.broadcast_to(garment, rgb.shape).astype(np.float32).copy()
    out = out * (1.0 - under) + 245.0 * under
    out = out * (1.0 - color_a) + rgb.astype(np.float32) * color_a
    return np.clip(out, 0, 255).astype(np.uint8)


def white_coverage_percent(white: np.ndarray, polarity: ChannelPolarity) -> float:
    ink = white if polarity == "white_prints" else (255 - white)
    return float(np.count_nonzero(ink > 16)) * 100.0 / ink.size


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr / 257.0).clip(0, 255).astype(np.uint8)
    data = arr.astype(np.float32)
    if data.max() <= 1.0:
        data *= 255.0
    return np.clip(data, 0, 255).astype(np.uint8)


def _planes_hwc(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 2:
        return data[..., None]
    if data.ndim != 3:
        raise ValueError("不支援的 TIFF 維度")
    if data.shape[0] <= 8 and data.shape[0] < min(data.shape[1], data.shape[2]):
        return np.moveaxis(data, 0, -1)
    return data


def _tiff_dpi(page) -> float:
    try:
        xres = page.tags.get("XResolution")
        if xres is None:
            return float(DEFAULT_DPI)
        value = xres.value
        if isinstance(value, tuple) and len(value) == 2 and value[1]:
            return float(value[0]) / float(value[1])
        return float(value)
    except Exception:
        return float(DEFAULT_DPI)


def extra_channel_names_from_page(page) -> list[str]:
    tag = page.tags.get(34377)
    if tag is None:
        return []
    raw = tag.value
    try:
        data = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
        ir = TiffImageResources.frombytes(data)
    except Exception:
        return []
    pascal: list[str] = []
    unicode_names: list[str] = []
    for block in ir.blocks:
        rid = int(block.resourceid)
        values = list(getattr(block, "values", []) or [])
        if rid == int(PsdResourceId.ALPHA_NAMES_PASCAL):
            pascal = values
        elif rid == int(PsdResourceId.ALPHA_NAMES_UNICODE):
            unicode_names = values
    return unicode_names or pascal


def _ps_composite(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Photoshop-style RGB thumbnail on black, not a white page."""
    if alpha is None:
        return rgb
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    return (rgb.astype(np.float32) * a).astype(np.uint8)


def _color_channel_views(rgb: np.ndarray) -> list[ChannelView]:
    return [
        ChannelView("RGB", rgb, shortcut="Ctrl+2", subtitle="合成"),
        ChannelView("红", rgb[:, :, 0], shortcut="Ctrl+3", subtitle="Red"),
        ChannelView("绿", rgb[:, :, 1], shortcut="Ctrl+4", subtitle="Green"),
        ChannelView("蓝", rgb[:, :, 2], shortcut="Ctrl+5", subtitle="Blue"),
    ]


def _cmyk_channel_views(cmyk: np.ndarray) -> list[ChannelView]:
    preview = cmyk_to_rgb_preview(cmyk)
    return [
        ChannelView("CMYK", preview, shortcut="Ctrl+2", subtitle="合成"),
        ChannelView("青", cmyk[:, :, 0], shortcut="Ctrl+3", subtitle="Cyan"),
        ChannelView("洋红", cmyk[:, :, 1], shortcut="Ctrl+4", subtitle="Magenta"),
        ChannelView("黄", cmyk[:, :, 2], shortcut="Ctrl+5", subtitle="Yellow"),
        ChannelView("黑", cmyk[:, :, 3], shortcut="Ctrl+6", subtitle="Black"),
    ]


def document_from_process(result: ProcessResult, channel_name: str) -> ChannelDocument:
    cmyk = rgb_alpha_to_cmyk(result.rgb, result.alpha)
    channels = _cmyk_channel_views(cmyk)
    channels.append(
        ChannelView(
            name=channel_name,
            image=result.white,
            shortcut="Ctrl+7",
            is_white=True,
            is_new=True,
            subtitle="白墨專色 Spot / PrintEXP",
        )
    )
    return ChannelDocument(
        filename=result.source_name,
        width=int(result.white.shape[1]),
        height=int(result.white.shape[0]),
        dpi=result.dpi,
        sample_count=5,
        has_white=True,
        status="generated",
        channels=channels,
        extra_names=[channel_name],
        note=f"已寫入 CMYK + Spot「{channel_name}」（PrintEXP / Maintop 格式）。灰階 = 白墨量。",
        rgb=result.rgb,
        white=result.white,
        alpha=result.alpha,
        coverage=result.coverage,
    )


def inspect_file(file_bytes: bytes, filename: str) -> ChannelDocument:
    lower = filename.lower()
    if lower.endswith((".tif", ".tiff")):
        return _inspect_tiff(file_bytes, filename)
    rgba, dpi = load_rgba(file_bytes, filename)
    rgb, alpha = flatten_rgb(rgba)
    preview = composite_rgba_preview(rgb, alpha)
    channels = _color_channel_views(preview)
    channels.append(
        ChannelView("Alpha", alpha, shortcut="⌘6", subtitle="透明度，不是白墨通道")
    )
    return ChannelDocument(
        filename=filename,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        dpi=dpi,
        sample_count=4,
        has_white=False,
        status="missing_white",
        channels=channels,
        extra_names=[],
        note="這是一般圖，沒有 RIP 用的 W1 專色通道。請用左側製作。",
        rgb=rgb,
        alpha=alpha,
    )


def _inspect_tiff(file_bytes: bytes, filename: str) -> ChannelDocument:
    with TiffFile(io.BytesIO(file_bytes)) as tif:
        page = tif.pages[0]
        data = _to_uint8(_planes_hwc(page.asarray()))
        extra_names = extra_channel_names_from_page(page)
        dpi = _tiff_dpi(page)
        photometric = int(getattr(page, "photometric", 2) or 2)

    height, width, samples = data.shape[0], data.shape[1], data.shape[2]
    is_cmyk = photometric == 5  # SEPARATED
    base_count = 4 if is_cmyk and samples >= 4 else min(3, samples)

    if is_cmyk and samples >= 4:
        cmyk = data[:, :, :4]
        rgb = cmyk_to_rgb_preview(cmyk)
        channels = _cmyk_channel_views(cmyk)
    else:
        rgb = data[:, :, :3] if samples >= 3 else np.repeat(data[:, :, :1], 3, axis=2)
        channels = _color_channel_views(_ps_composite(rgb, None))

    found_white = False
    resolved_extras: list[str] = []
    white_plane = None
    for index in range(base_count, samples):
        extra_index = index - base_count
        given = extra_names[extra_index] if extra_index < len(extra_names) else ""
        name = given or f"Extra {extra_index + 1}"
        resolved_extras.append(name)
        is_white = _is_white_name(name)
        found_white = found_white or is_white
        plane = data[:, :, index]
        if is_white:
            white_plane = plane
        channels.append(
            ChannelView(
                name=name,
                image=plane,
                shortcut=f"⌘{index + 3}",
                is_white=is_white,
                subtitle="白墨通道 W1" if is_white else "額外通道",
            )
        )

    if samples <= base_count:
        mode = "CMYK" if is_cmyk else "RGB"
        note = f"此 TIFF 只有 {mode}，沒有 W1 專色通道。可用左側透明 PNG 重新製作。"
        status: Literal["has_white", "missing_white"] = "missing_white"
    elif found_white:
        note = "已有 W1 / 白墨專色通道，PrintEXP 可直接匯入。"
        status = "has_white"
    else:
        extra_label = "、".join(resolved_extras) if resolved_extras else "未命名"
        note = f"有額外通道（{extra_label}），但名稱不是 W1。PrintEXP Spot 模式可能認不出白墨。"
        status = "missing_white"

    return ChannelDocument(
        filename=filename,
        width=width,
        height=height,
        dpi=dpi,
        sample_count=samples,
        has_white=found_white,
        status=status,
        channels=channels,
        extra_names=resolved_extras,
        note=note,
        rgb=rgb,
        white=white_plane,
    )


def _to_preview_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.dstack([arr, arr, arr])
    return arr


def _b64_thumb(arr: np.ndarray, size: int = 48) -> str:
    rgb = _to_preview_rgb(arr)
    img = Image.fromarray(rgb)
    img.thumbnail((size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (245, 245, 247))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_preview(arr: np.ndarray, max_edge: int = 420) -> str:
    rgb = _to_preview_rgb(arr)
    (preview,) = downscale_preview(rgb, max_edge=max_edge)
    img = Image.fromarray(preview)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_channel_panel(
    doc: ChannelDocument | None,
    preview_choice: str = "RGB",
) -> None:
    if doc is None:
        st.markdown(
            """
            <div class="empty-channels">
              <div class="empty-mark"></div>
              <div class="empty-title">通道</div>
              <div class="empty-copy">製作完成後，這裡會出現與 Photoshop / PrintEXP 相同的通道：<br>CMYK + W1 白墨專色。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if doc.status == "generated":
        tone, label = "ok", "已新增 W1"
    elif doc.status == "has_white":
        tone, label = "ok", "已有 W1，不必重做"
    else:
        tone, label = "warn", "沒有 W1 通道"

    preview_map = {ch.name: ch.image for ch in doc.channels}
    if doc.coverage is not None:
        preview_map["原始 Alpha"] = doc.coverage
    elif doc.alpha is not None:
        preview_map["原始 Alpha"] = doc.alpha
    if doc.white is not None:
        preview_map["內縮後白墨"] = doc.white

    preview_src = preview_map.get(preview_choice)
    if preview_src is None:
        preview_src = doc.channels[0].image if doc.channels else np.zeros((8, 8, 3), np.uint8)

    rows = []
    for ch in doc.channels:
        badge = ""
        row_class = "ch-row"
        white_previews = {"內縮後白墨", "white", "W1"}
        if ch.name == preview_choice or (preview_choice in white_previews and ch.is_white):
            row_class += " ch-active"
        if ch.is_white:
            row_class += " ch-white"
            badge = (
                '<span class="badge badge-new">新增</span>'
                if ch.is_new
                else '<span class="badge badge-ok">已有</span>'
            )
        eye = "eye-on" if ch.name in {"RGB", "CMYK"} else "eye-off"
        rows.append(
            f"""
            <div class="{row_class}">
              <span class="eye {eye}"></span>
              <img alt="" src="data:image/png;base64,{_b64_thumb(ch.image)}" />
              <div class="ch-meta">
                <div class="ch-name">{html.escape(ch.name)}</div>
                <div class="ch-sub">{html.escape(ch.subtitle)}</div>
              </div>
              <div class="ch-key">{html.escape(ch.shortcut)}</div>
              {badge}
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="status-pill {tone}">{html.escape(label)}</div>
        <div class="panel-kicker">{html.escape(doc.filename)} · {doc.width} × {doc.height} · {doc.dpi:.0f} dpi · {doc.sample_count} samples</div>
        <div class="note">{html.escape(doc.note)}</div>
        <div class="preview-frame preview-dark">
          <img alt="preview" src="data:image/png;base64,{_b64_preview(preview_src)}" />
        </div>
        <div class="ps-label">通道</div>
        <div class="ch-panel">{''.join(rows)}</div>
        """,
        unsafe_allow_html=True,
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
          font-family: "Inter", -apple-system, BlinkMacSystemFont, "SF Pro Text",
            "SF Pro Display", "Helvetica Neue", sans-serif;
        }
        .stApp { background: #f5f5f7; }
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
        .block-container {
          padding: 2.4rem 2rem 4rem;
          max-width: 1180px;
        }

        .hero-eyebrow {
          font-size: 12px;
          letter-spacing: 0.18em;
          font-weight: 600;
          color: #86868b;
          text-transform: uppercase;
          margin-bottom: 8px;
        }
        .hero h1 {
          font-size: 48px;
          line-height: 1.05;
          letter-spacing: -0.035em;
          font-weight: 600;
          color: #1d1d1f;
          margin: 0 0 10px;
        }
        .hero p {
          font-size: 17px;
          line-height: 1.45;
          color: #86868b;
          max-width: 560px;
          margin: 0 0 28px;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
          background: #ffffff;
          border-radius: 24px;
          padding: 8px 8px 18px;
          box-shadow: 0 8px 40px rgba(0,0,0,0.04);
          border: 1px solid rgba(0,0,0,0.04);
        }

        .panel-title {
          font-size: 21px;
          font-weight: 600;
          letter-spacing: -0.02em;
          color: #1d1d1f;
          margin: 10px 8px 4px;
        }
        .panel-copy {
          font-size: 14px;
          color: #86868b;
          margin: 0 8px 18px;
          line-height: 1.45;
        }
        .hairline {
          height: 1px;
          background: #e8e8ed;
          margin: 22px 8px;
        }

        [data-testid="stFileUploaderDropzone"] {
          background: #fbfbfd !important;
          border: 1px dashed #d2d2d7 !important;
          border-radius: 16px !important;
          min-height: 118px;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
          border-color: #0071e3 !important;
          background: #f5f9ff !important;
        }
        [data-testid="stFileUploaderDropzone"] section { max-width: 100%; }
        [data-testid="stFileUploaderDropzoneInstructions"] span { color: #1d1d1f; }

        .stDownloadButton button, .stButton button {
          background: #0071e3 !important;
          color: #fff !important;
          border: 0 !important;
          border-radius: 980px !important;
          height: 44px !important;
          font-weight: 500 !important;
          font-size: 15px !important;
          box-shadow: none !important;
        }
        .stDownloadButton button:hover, .stButton button:hover {
          background: #0077ed !important;
        }

        div[data-testid="stSlider"] label { font-weight: 500; color: #1d1d1f; }
        .stExpander { border: 0 !important; background: #fbfbfd; border-radius: 14px; }

        .status-pill {
          display: inline-flex;
          align-items: center;
          font-size: 12px;
          font-weight: 600;
          letter-spacing: 0.02em;
          padding: 6px 10px;
          border-radius: 980px;
          margin: 12px 8px 8px;
        }
        .status-pill.ok { background: #e8f8ee; color: #1f8a3b; }
        .status-pill.warn { background: #fff4e5; color: #9a5b00; }
        .panel-kicker {
          font-size: 12px;
          color: #86868b;
          margin: 0 8px 8px;
        }
        .note {
          font-size: 14px;
          color: #1d1d1f;
          margin: 0 8px 16px;
          line-height: 1.45;
        }
        .preview-frame {
          margin: 0 8px 16px;
          background: #f5f5f7;
          border-radius: 16px;
          overflow: hidden;
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 180px;
        }
        .preview-frame.preview-dark { background: #111; }
        .ps-label {
          font-size: 12px;
          font-weight: 600;
          color: #86868b;
          margin: 4px 10px 6px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .ch-row.ch-active { outline: 1px solid #0071e3; background: #eef5ff; }
        .eye {
          width: 14px; height: 14px;
          border-radius: 50%;
          flex: 0 0 14px;
        }
        .eye-on { background: #1d1d1f; }
        .eye-off { background: transparent; border: 1px solid #d2d2d7; }
        .ch-panel { margin: 0 4px 8px; }
        .ch-row {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 8px 10px;
          border-radius: 12px;
          margin-bottom: 4px;
        }
        .ch-row.ch-white { background: #f2f7ff; }
        .ch-row img {
          width: 48px; height: 48px;
          border-radius: 8px;
          border: 1px solid rgba(0,0,0,0.06);
        }
        .ch-meta { flex: 1; min-width: 0; }
        .ch-name { font-size: 14px; font-weight: 600; color: #1d1d1f; }
        .ch-sub { font-size: 12px; color: #86868b; }
        .ch-key { font-size: 12px; color: #a1a1a6; font-variant-numeric: tabular-nums; }
        .badge {
          font-size: 11px;
          font-weight: 600;
          padding: 3px 8px;
          border-radius: 980px;
          white-space: nowrap;
        }
        .badge-new { background: #0071e3; color: #fff; }
        .badge-ok { background: #e8f8ee; color: #1f8a3b; }

        .empty-channels {
          min-height: 520px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 40px 24px;
        }
        .empty-mark {
          width: 72px; height: 72px;
          border-radius: 18px;
          background: linear-gradient(180deg, #f5f5f7, #e8e8ed);
          margin-bottom: 18px;
          box-shadow: inset 0 0 0 1px rgba(0,0,0,0.04);
        }
        .empty-title { font-size: 21px; font-weight: 600; color: #1d1d1f; margin-bottom: 8px; }
        .empty-copy { font-size: 14px; color: #86868b; line-height: 1.5; }

        [data-testid="stCaptionContainer"] p { color: #86868b; }

        @media (max-width: 768px) {
          .block-container { padding: 1.1rem 0.7rem 3.2rem; }
          .hero h1 { font-size: 32px; }
          .hero p { font-size: 15px; margin-bottom: 18px; }
          .panel-title { font-size: 19px; }
          .empty-channels { min-height: 240px; padding: 24px 12px; }
          .preview-frame { min-height: 140px; }
          .ch-key { display: none; }
          .stDownloadButton button, .stButton button { height: 48px !important; }
          [data-testid="stFileUploaderDropzone"] { min-height: 96px; }
          div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            margin-bottom: 12px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app() -> None:
    st.set_page_config(
        page_title="ON99 White Channel",
        page_icon="⬜",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow">ON99</div>
          <h1>White Channel</h1>
          <p>上傳透明 PNG，輸出 PrintEXP / Maintop 可匯入的 CMYK + W1 專色 TIFF。右側可對比內縮前後的白墨。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            '<div class="panel-title">輸入</div>'
            '<div class="panel-copy">點選上傳透明 PNG。白墨為灰階（保留做舊），並向內縮一圈避免露白。</div>',
            unsafe_allow_html=True,
        )
        source = st.file_uploader(
            "製作白墨 TIFF",
            type=["png", "webp"],
            key="source_png",
            help="請使用去背透明 PNG。",
            label_visibility="collapsed",
        )

        choke_px = st.slider(
            "白墨內縮像素 (Choke Limit)",
            min_value=0,
            max_value=10,
            value=DEFAULT_CHOKE_PX,
            help="白墨比彩圖小一圈。0 仍可能露白，建議 2。",
        )
        with st.expander("進階"):
            black_cutoff = st.slider(
                "黑色不鋪白（保留做舊）",
                0,
                60,
                DEFAULT_BLACK_CUTOFF,
                help="夠黑的像素不噴白墨，磨損細點才不會變死白。",
            )
            black_feather = st.slider("黑色過渡", 4, 80, DEFAULT_BLACK_FEATHER)
            polarity = st.radio(
                "極性",
                ["white_prints", "black_prints"],
                format_func=lambda x: "白 = 噴白墨" if x == "white_prints" else "黑 = 噴白墨",
                horizontal=True,
            )
            channel_name = st.selectbox(
                "專色通道名稱（PrintEXP 必須為 W1）",
                ["W1", "W2", "white", "White"],
                index=0,
                help="PrintEXP / Maintop Spot 模式認 W1 為白墨。W2 通常是光油。",
            )
            dpi_override = st.number_input("DPI", min_value=72, max_value=600, value=DEFAULT_DPI)
            compression = st.radio(
                "TIFF",
                ["none", "zip"],
                format_func=lambda x: "無壓縮（RIP 最穩）" if x == "none" else "ZIP 無損（檔案較小）",
                horizontal=True,
            )

        result: ProcessResult | None = None
        generated_doc: ChannelDocument | None = None
        if source is not None:
            try:
                result = process_artwork(
                    file_bytes=source.getvalue(),
                    filename=source.name,
                    choke_px=int(choke_px),
                    black_cutoff=int(black_cutoff),
                    black_feather=int(black_feather),
                    polarity=polarity,  # type: ignore[arg-type]
                    dpi_override=float(dpi_override),
                )
                generated_doc = document_from_process(result, channel_name)
                tiff_bytes = write_tiff_with_white(
                    result.rgb,
                    result.white,
                    result.dpi,
                    channel_name=channel_name,
                    compression=compression,
                    alpha=result.alpha,
                )
                psd_bytes = write_psd_with_white(
                    result.rgb,
                    result.white,
                    result.dpi,
                    channel_name=channel_name,
                    alpha=result.alpha,
                )
                stem = source.name.rsplit(".", 1)[0]
                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "下載 TIFF",
                        data=tiff_bytes,
                        file_name=f"{stem}_{channel_name}.tif",
                        mime="image/tiff",
                        type="primary",
                        use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        "下載 PSD",
                        data=psd_bytes,
                        file_name=f"{stem}_{channel_name}.psd",
                        mime="image/vnd.adobe.photoshop",
                        use_container_width=True,
                    )
                st.caption("CMYK + W1 Spot · 灰階白墨 · ExtraSamples=UNSPECIFIED · PrintEXP Spot 可匯入")
            except Exception as exc:
                st.error(f"無法處理這張圖：{exc}")

        st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">檢查</div>'
            '<div class="panel-copy">拖入已做好的 TIFF 或 PNG，看通道裡有沒有 W1，避免重複製作。</div>',
            unsafe_allow_html=True,
        )
        inspect_upload = st.file_uploader(
            "檢查既有檔的通道",
            type=["tif", "tiff", "png", "webp"],
            key="inspect_file",
            label_visibility="collapsed",
        )

    inspect_doc: ChannelDocument | None = None
    if inspect_upload is not None:
        try:
            inspect_doc = inspect_file(inspect_upload.getvalue(), inspect_upload.name)
        except Exception as exc:
            with right:
                st.error(f"無法讀取通道：{exc}")

    with right:
        st.markdown(
            '<div class="panel-title">通道</div>'
            '<div class="panel-copy">與 Photoshop / PrintEXP 相同：CMYK + W1。可切換原始覆蓋與內縮後白墨。</div>',
            unsafe_allow_html=True,
        )
        preview_choice = st.radio(
            "白墨遮罩對比預覽",
            options=["CMYK", "W1", "原始 Alpha", "內縮後白墨"],
            index=0,
            horizontal=True,
            help="原始 Alpha = 未內縮覆蓋；內縮後白墨 / W1 = 實際寫入 TIFF 的專色通道。",
        )
        render_channel_panel(inspect_doc or generated_doc, preview_choice)


if __name__ == "__main__":
    render_app()
