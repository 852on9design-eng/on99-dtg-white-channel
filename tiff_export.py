"""PrintEXP-compatible TIFF export for ON99 white underbase.

Hoson PrintEXP Spot mode needs a Photoshop Spot Color Channel. ExtraSamples
labeled only with ALPHA_NAMES are treated as Alpha and often yield
``Invalid image format``.

Default export matches a Photoshop-saved print file shape:
**CMYK (Separated) + Spot ``white``**, with DisplayInfo kind=2 and Alternate
Spot Colors — the same pattern as known-good UV/DTF spot TIFFs.

RGB + Spot remains available as an alternate mode.
"""

from __future__ import annotations

import re
import struct
from datetime import datetime
from io import BytesIO
from typing import Literal

import numpy as np
from tifffile import TiffFile, imwrite

from psdtags import (
    PsdBytesBlock,
    PsdFormat,
    PsdPascalStringsBlock,
    PsdResourceId,
    PsdStringsBlock,
    PsdVersionBlock,
    TiffImageResources,
)

ExportMode = Literal["printexp_cmyk_spot", "printexp_rgb_spot", "legacy_extrasamples"]
ColorSpace = Literal["cmyk", "rgb"]

DEFAULT_SPOT_NAME = "white"
DEFAULT_VARNISH_NAME = "varnish"
DEFAULT_DPI = 300
SPOT_KIND = 2  # Photoshop DisplayInfo: 2 = spot color
SPOT_START_ID = 3


def safe_download_stem(name: str) -> str:
    """PrintEXP / Windows loaders often choke on spaces and parentheses."""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    stem = stem.strip().replace(" ", "_")
    stem = re.sub(r"[^\w.\-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "artwork"


def rgb_alpha_to_cmyk(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """RGB(+alpha) → CMYK ink. Fully transparent pixels stay ink-free."""
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


def _resolution_info_bytes(dpi: float) -> bytes:
    fixed = int(round(float(dpi) * 65536.0)) & 0xFFFFFFFF
    return struct.pack(">IHH IHH", fixed, 1, 1, fixed, 1, 1)


def _display_info_obsolete(count: int) -> bytes:
    parts: list[bytes] = []
    for _ in range(count):
        parts.append(struct.pack(">h4HHBB", 1, 0, 65535, 65535, 0, 100, SPOT_KIND, 0))
    return b"".join(parts)


def _display_info_cs3(count: int) -> bytes:
    parts = [struct.pack(">I", 1)]
    for _ in range(count):
        parts.append(struct.pack(">h4HHb", 1, 0, 65535, 65535, 0, 0, SPOT_KIND))
    return b"".join(parts)


def _alternate_spot_colors(count: int, start_id: int = SPOT_START_ID) -> bytes:
    parts = [struct.pack(">HH", 1, count)]
    for index in range(count):
        channel_id = start_id + index
        parts.append(struct.pack(">Ih4H", channel_id, 7, 0x1535, 0x1F90, 0x1B4E, 0))
    return b"".join(parts)


def _spot_halftone(count: int) -> bytes:
    base = bytes.fromhex(
        "00060002000000000001000000000000000000000000000000000001000000000000000000000000"
    )
    if count <= 1:
        return base
    return base + base[4:]


def _alpha_identifiers(count: int, start_id: int = SPOT_START_ID) -> bytes:
    return b"".join(struct.pack(">I", start_id + i) for i in range(count))


def photoshop_spot_image_resources(
    channel_names: list[str],
    dpi: float = DEFAULT_DPI,
) -> tuple:
    if not channel_names:
        raise ValueError("至少需要一個 Spot 通道名稱")
    count = len(channel_names)
    resources = TiffImageResources(
        name=f"{channel_names[0]}.tif",
        psdformat=PsdFormat.BE32BIT,
        blocks=[
            PsdBytesBlock(
                resourceid=PsdResourceId.RESOLUTION_INFO,
                value=_resolution_info_bytes(dpi),
            ),
            PsdPascalStringsBlock(
                resourceid=PsdResourceId.ALPHA_NAMES_PASCAL,
                values=channel_names,
            ),
            PsdBytesBlock(
                resourceid=PsdResourceId.DISPLAY_INFO_OBSOLETE,
                value=_display_info_obsolete(count),
            ),
            PsdStringsBlock(
                resourceid=PsdResourceId.ALPHA_NAMES_UNICODE,
                values=channel_names,
            ),
            PsdBytesBlock(
                resourceid=PsdResourceId.DISPLAY_INFO,
                value=_display_info_cs3(count),
            ),
            PsdBytesBlock(
                resourceid=PsdResourceId.ALPHA_IDENTIFIERS,
                value=_alpha_identifiers(count),
            ),
            PsdBytesBlock(
                resourceid=PsdResourceId.ALTERNATE_SPOT_COLORS,
                value=_alternate_spot_colors(count),
            ),
            PsdBytesBlock(
                resourceid=PsdResourceId.SPOT_HALFTONE,
                value=_spot_halftone(count),
            ),
            PsdVersionBlock(
                resourceid=PsdResourceId.VERSION_INFO,
                version=1,
                file_version=1,
                writer_name="Adobe Photoshop",
                reader_name="Adobe Photoshop",
                has_real_merged_data=True,
            ),
        ],
    )
    return resources.tifftag()


def photoshop_legacy_alpha_resources(channel_names: list[str]) -> tuple:
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
        ],
    )
    return resources.tifftag()


def write_tiff_with_spot(
    rgb: np.ndarray,
    white: np.ndarray,
    dpi: float = DEFAULT_DPI,
    *,
    mode: ExportMode = "printexp_cmyk_spot",
    channel_name: str = DEFAULT_SPOT_NAME,
    varnish: np.ndarray | None = None,
    varnish_name: str = DEFAULT_VARNISH_NAME,
    compression: str = "none",
    alpha: np.ndarray | None = None,
) -> bytes:
    """Write print TIFF + Spot plane(s).

    ``printexp_cmyk_spot`` (default): Photometric SEPARATED CMYK + Spot white.
    ``printexp_rgb_spot``: RGB + Spot white.
    ``legacy_extrasamples``: RGB + ExtraSamples with ALPHA_NAMES only.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb 必須是 HxWx3")
    if white.shape != rgb.shape[:2]:
        raise ValueError("white 通道尺寸必須與 RGB 相同")
    if varnish is not None and varnish.shape != white.shape:
        raise ValueError("varnish 尺寸必須與 white 相同")
    if alpha is None:
        alpha = np.full(white.shape, 255, dtype=np.uint8)
    elif alpha.shape != white.shape:
        raise ValueError("alpha 尺寸必須與 white 相同")

    names = [channel_name]
    use_cmyk = mode == "printexp_cmyk_spot"
    if use_cmyk:
        base = rgb_alpha_to_cmyk(rgb, alpha)
        photometric = "separated"
    else:
        base = rgb
        photometric = "rgb"

    planes: list[np.ndarray] = [base, white[..., None]]
    if varnish is not None:
        names.append(varnish_name)
        planes.append(varnish[..., None])

    stacked = np.concatenate(planes, axis=2).astype(np.uint8, copy=False)
    extras = tuple(0 for _ in names)

    if mode in {"printexp_cmyk_spot", "printexp_rgb_spot"}:
        extratags = [
            (254, "I", 1, 0, False),  # NewSubfileType
            (274, "H", 1, 1, False),  # Orientation
            photoshop_spot_image_resources(names, dpi=dpi),
        ]
        software = "Adobe Photoshop 25.0 (Windows)"
    elif mode == "legacy_extrasamples":
        extratags = [photoshop_legacy_alpha_resources(names)]
        software = "tifffile.py"
    else:
        raise ValueError(f"未知 export mode: {mode}")

    # Intentionally no ICC: some PrintEXP builds reject exotic profiles.
    options: dict = dict(
        photometric=photometric,
        extrasamples=extras,
        planarconfig="contig",
        metadata=False,
        resolution=(float(dpi), float(dpi)),
        resolutionunit="inch",
        software=software,
        datetime=datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
        extratags=extratags,
    )
    if compression == "zip":
        options["compression"] = "adobe_deflate"
    elif compression == "lzw":
        options["compression"] = "lzw"

    buf = BytesIO()
    imwrite(buf, stacked, **options)
    return buf.getvalue()


def inspect_spot_metadata(file_bytes: bytes) -> dict:
    """Parse TIFF ImageResources and report Spot vs Alpha markers."""
    with TiffFile(BytesIO(file_bytes)) as tif:
        page = tif.pages[0]
        photometric = int(page.photometric)
        samples = int(page.samplesperpixel)
        extras = tuple(int(x) for x in (page.extrasamples or ()))
        software = ""
        if "Software" in page.tags:
            software = str(page.tags["Software"].value)
        tag = page.tags.get(34377)
        ir_bytes = tag.value if tag is not None else None

    empty = {
        "photometric": photometric,
        "samples": samples,
        "extrasamples": extras,
        "software": software,
        "names": [],
        "has_display_info": False,
        "has_display_info_obsolete": False,
        "has_alternate_spot": False,
        "has_spot_halftone": False,
        "spot_kinds": [],
        "is_photoshop_spot": False,
        "color_space": "cmyk" if photometric == 5 else "rgb",
    }
    if ir_bytes is None:
        return empty

    ir = TiffImageResources.frombytes(ir_bytes)
    names: list[str] = []
    has_display = False
    has_alt = False
    has_obsolete_display = False
    has_spot_halftone = False
    spot_kinds: list[int] = []

    for block in ir.blocks:
        rid = int(block.resourceid)
        if rid == int(PsdResourceId.ALPHA_NAMES_UNICODE):
            names = list(block.values)
        elif rid == int(PsdResourceId.ALPHA_NAMES_PASCAL) and not names:
            names = list(block.values)
        elif rid == int(PsdResourceId.DISPLAY_INFO):
            has_display = True
            raw = block.value
            off = 4
            while off + 13 <= len(raw):
                spot_kinds.append(raw[off + 12])
                off += 13
        elif rid == int(PsdResourceId.DISPLAY_INFO_OBSOLETE):
            has_obsolete_display = True
        elif rid == int(PsdResourceId.ALTERNATE_SPOT_COLORS):
            has_alt = True
        elif rid == int(PsdResourceId.SPOT_HALFTONE):
            has_spot_halftone = True

    is_spot = bool(
        names
        and has_display
        and has_alt
        and spot_kinds
        and all(k == SPOT_KIND for k in spot_kinds)
    )
    return {
        "photometric": photometric,
        "samples": samples,
        "extrasamples": extras,
        "software": software,
        "names": names,
        "has_display_info": has_display,
        "has_display_info_obsolete": has_obsolete_display,
        "has_alternate_spot": has_alt,
        "has_spot_halftone": has_spot_halftone,
        "spot_kinds": spot_kinds,
        "is_photoshop_spot": is_spot,
        "color_space": "cmyk" if photometric == 5 else "rgb",
    }
