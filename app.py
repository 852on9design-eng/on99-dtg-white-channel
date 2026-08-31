"""ON99 DTG 白墨通道生成工具。

從透明 PNG 自動產生第 5 通道 white，輸出無損 TIFF 給 DTG RIP。
右側通道面板可預覽產出，也可拖入既有 TIFF 檢查是否已有 white。

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
from PIL import Image
from scipy import ndimage
from tifffile import TiffFile, imwrite

from psdtags import (
    PsdFormat,
    PsdPascalStringsBlock,
    PsdResourceId,
    PsdStringsBlock,
    TiffImageResources,
)

WHITE_CHANNEL_NAME = "white"
DEFAULT_CHOKE_PX = 2
DEFAULT_DPI = 300
MAX_PREVIEW_EDGE = 640
WHITE_NAME_ALIASES = {
    "white",
    "white_ink",
    "whiteink",
    "w1",
    "spotwhite",
    "whiteinkchannel",
}

ChannelPolarity = Literal["white_prints", "black_prints"]
ChokeMode = Literal["soft", "hard"]


@dataclass(frozen=True)
class ProcessResult:
    rgb: np.ndarray
    alpha: np.ndarray
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


def _disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


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


def choke_alpha(
    alpha: np.ndarray,
    pixels: int,
    mode: ChokeMode = "soft",
    threshold: int = 8,
) -> np.ndarray:
    alpha = alpha.astype(np.uint8, copy=False)
    if threshold > 0:
        alpha = np.where(alpha >= threshold, alpha, 0).astype(np.uint8)
    if pixels <= 0:
        return alpha
    footprint = _disk(int(pixels))
    if mode == "hard":
        binary = alpha > max(threshold, 1)
        eroded = ndimage.binary_erosion(binary, structure=footprint)
        return np.where(eroded, 255, 0).astype(np.uint8)
    return ndimage.grey_erosion(alpha, footprint=footprint).astype(np.uint8)


def build_white_channel(
    alpha: np.ndarray,
    choke_px: int,
    threshold: int,
    mode: ChokeMode,
    polarity: ChannelPolarity,
) -> np.ndarray:
    white = choke_alpha(alpha, choke_px, mode=mode, threshold=threshold)
    if polarity == "black_prints":
        return (255 - white).astype(np.uint8)
    return white


def flatten_rgb(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = rgba[..., :3].copy()
    alpha = rgba[..., 3].copy()
    rgb[alpha == 0] = 0
    return rgb, alpha


def process_artwork(
    file_bytes: bytes,
    filename: str,
    choke_px: int,
    threshold: int,
    choke_mode: ChokeMode,
    polarity: ChannelPolarity,
    dpi_override: float | None = None,
) -> ProcessResult:
    rgba, dpi = load_rgba(file_bytes, filename)
    rgb, alpha = flatten_rgb(rgba)
    white = build_white_channel(alpha, choke_px, threshold, choke_mode, polarity)
    return ProcessResult(
        rgb=rgb,
        alpha=alpha,
        white=white,
        dpi=float(dpi_override or dpi),
        source_name=filename,
    )


def _photoshop_channel_tags(channel_name: str) -> tuple:
    resources = TiffImageResources(
        name=f"{channel_name}.tif",
        psdformat=PsdFormat.BE32BIT,
        blocks=[
            PsdPascalStringsBlock(
                resourceid=PsdResourceId.ALPHA_NAMES_PASCAL,
                values=[channel_name],
            ),
            PsdStringsBlock(
                resourceid=PsdResourceId.ALPHA_NAMES_UNICODE,
                values=[channel_name],
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
) -> bytes:
    """Lossless RGB TIFF + extra sample named white. Uncompressed by default for RIP."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb 必須是 HxWx3")
    if white.shape != rgb.shape[:2]:
        raise ValueError("white 通道尺寸必須與 RGB 相同")

    stacked = np.concatenate([rgb, white[..., None]], axis=2).astype(np.uint8)
    buf = io.BytesIO()
    options = dict(
        photometric="rgb",
        extrasamples="unspecified",
        planarconfig="contig",
        metadata=None,
        resolution=(float(dpi), float(dpi)),
        resolutionunit="inch",
        extratags=[_photoshop_channel_tags(channel_name)],
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
) -> bytes:
    height, width = white.shape
    header = b"".join(
        [
            b"8BPS",
            struct.pack(">H", 1),
            b"\x00" * 6,
            struct.pack(">H", 4),
            struct.pack(">I", height),
            struct.pack(">I", width),
            struct.pack(">H", 8),
            struct.pack(">H", 3),
        ]
    )
    alpha_names = _pascal_string(channel_name, pad=1)
    resources = b"".join(
        [
            _psd_resource(1005, _psd_resolution(dpi)),
            _psd_resource(1006, alpha_names),
            _psd_resource(1045, _psd_unicode_names([channel_name])),
            _psd_resource(1053, struct.pack(">I", 0)),
        ]
    )
    image_data = b"".join(
        [
            struct.pack(">H", 0),
            rgb[:, :, 0].tobytes(),
            rgb[:, :, 1].tobytes(),
            rgb[:, :, 2].tobytes(),
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


def _color_channel_views(rgb: np.ndarray) -> list[ChannelView]:
    return [
        ChannelView("RGB", rgb, shortcut="⌘2", subtitle="合成"),
        ChannelView("紅", rgb[:, :, 0], shortcut="⌘3", subtitle="Red"),
        ChannelView("綠", rgb[:, :, 1], shortcut="⌘4", subtitle="Green"),
        ChannelView("藍", rgb[:, :, 2], shortcut="⌘5", subtitle="Blue"),
    ]


def document_from_process(result: ProcessResult, channel_name: str) -> ChannelDocument:
    rgb_view = composite_rgba_preview(result.rgb, result.alpha)
    channels = _color_channel_views(rgb_view)
    channels.append(
        ChannelView(
            name=channel_name,
            image=result.white,
            shortcut="⌘6",
            is_white=True,
            is_new=True,
            subtitle="白墨 Extra / Spot",
        )
    )
    return ChannelDocument(
        filename=result.source_name,
        width=int(result.white.shape[1]),
        height=int(result.white.shape[0]),
        dpi=result.dpi,
        sample_count=4,
        has_white=True,
        status="generated",
        channels=channels,
        extra_names=[channel_name],
        note="已新增第 5 通道 white，可下載 TIFF 給 RIP。",
        rgb=result.rgb,
        white=result.white,
        alpha=result.alpha,
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
        note="這是一般圖，沒有 RIP 用的 white 專色通道。請用左側製作。",
        rgb=rgb,
        alpha=alpha,
    )


def _inspect_tiff(file_bytes: bytes, filename: str) -> ChannelDocument:
    with TiffFile(io.BytesIO(file_bytes)) as tif:
        page = tif.pages[0]
        data = _to_uint8(_planes_hwc(page.asarray()))
        extra_names = extra_channel_names_from_page(page)
        dpi = _tiff_dpi(page)

    height, width, samples = data.shape[0], data.shape[1], data.shape[2]
    rgb = data[:, :, :3] if samples >= 3 else np.repeat(data[:, :, :1], 3, axis=2)
    channels = _color_channel_views(rgb)

    found_white = False
    resolved_extras: list[str] = []
    white_plane = None
    for index in range(3, samples):
        extra_index = index - 3
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
                subtitle="白墨通道" if is_white else "額外通道",
            )
        )

    if samples <= 3:
        note = "此 TIFF 只有 RGB，沒有 white 通道。可用左側透明 PNG 重新製作。"
        status: Literal["has_white", "missing_white"] = "missing_white"
    elif found_white:
        note = "已有 white 通道，不必重做。"
        status = "has_white"
    else:
        extra_label = "、".join(resolved_extras) if resolved_extras else "未命名"
        note = f"有額外通道（{extra_label}），但名稱不是 white。RIP 可能認不出白墨。"
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


def render_channel_panel(doc: ChannelDocument | None) -> None:
    if doc is None:
        st.markdown(
            """
            <div class="empty-channels">
              <div class="empty-mark"></div>
              <div class="empty-title">通道</div>
              <div class="empty-copy">製作完成後，這裡會出現 RGB 與第 5 通道 white。<br>也可拖入既有 TIFF，確認是否已經有白墨通道。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if doc.status == "generated":
        tone, label = "ok", "已新增 white"
    elif doc.status == "has_white":
        tone, label = "ok", "已有 white，不必重做"
    else:
        tone, label = "warn", "沒有 white 通道"

    rows = []
    for ch in doc.channels:
        badge = ""
        row_class = "ch-row"
        if ch.is_white and ch.is_new:
            row_class += " ch-white"
            badge = '<span class="badge badge-new">新增</span>'
        elif ch.is_white:
            row_class += " ch-white"
            badge = '<span class="badge badge-ok">已有</span>'
        rows.append(
            f"""
            <div class="{row_class}">
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

    preview_src = doc.channels[0].image if doc.channels else np.zeros((8, 8, 3), np.uint8)
    st.markdown(
        f"""
        <div class="status-pill {tone}">{html.escape(label)}</div>
        <div class="panel-kicker">{html.escape(doc.filename)} · {doc.width} × {doc.height} · {doc.dpi:.0f} dpi · {doc.sample_count} samples</div>
        <div class="note">{html.escape(doc.note)}</div>
        <div class="preview-frame">
          <img alt="preview" src="data:image/png;base64,{_b64_preview(preview_src)}" />
        </div>
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
        .preview-frame img { width: 100%; display: block; }
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
          <p>上傳透明 PNG，產生含 white 通道的 TIFF。電腦與手機都可接著做；也可打開舊檔，確認是否已有白墨通道。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            '<div class="panel-title">輸入</div>'
            '<div class="panel-copy">點選上傳，從檔案或「檔案 App」選透明 PNG。白墨由 Alpha 自動生成並收邊。</div>',
            unsafe_allow_html=True,
        )
        source = st.file_uploader(
            "製作白墨 TIFF",
            type=["png", "webp"],
            key="source_png",
            help="請使用去背透明 PNG。",
            label_visibility="collapsed",
        )

        choke_px = st.slider("白墨收邊", min_value=0, max_value=12, value=DEFAULT_CHOKE_PX)
        with st.expander("進階"):
            threshold = st.slider("Alpha 門檻", 0, 128, 8)
            choke_mode = st.radio(
                "收邊",
                ["soft", "hard"],
                format_func=lambda x: "柔邊" if x == "soft" else "硬邊",
                horizontal=True,
            )
            polarity = st.radio(
                "極性",
                ["white_prints", "black_prints"],
                format_func=lambda x: "白 = 噴白墨" if x == "white_prints" else "黑 = 噴白墨",
                horizontal=True,
            )
            channel_name = st.selectbox(
                "通道名稱",
                ["white", "White", "WHITE", "White_Ink"],
                index=0,
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
                    threshold=int(threshold),
                    choke_mode=choke_mode,  # type: ignore[arg-type]
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
                )
                stem = source.name.rsplit(".", 1)[0]
                st.download_button(
                    "下載 TIFF",
                    data=tiff_bytes,
                    file_name=f"{stem}_{channel_name}.tif",
                    mime="image/tiff",
                    type="primary",
                    use_container_width=True,
                )
                st.caption("無損 TIFF · RGB + white 專色通道 · 給 DTG RIP 直接輸出")
            except Exception as exc:
                st.error(f"無法處理這張圖：{exc}")

        st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">檢查</div>'
            '<div class="panel-copy">拖入已做好的 TIFF 或 PNG，看通道裡有沒有 white，避免重複製作。</div>',
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
            '<div class="panel-copy">與 Photoshop 通道面板相同：RGB、紅、綠、藍，以及第 5 通道 white。</div>',
            unsafe_allow_html=True,
        )
        render_channel_panel(inspect_doc or generated_doc)


if __name__ == "__main__":
    render_app()
