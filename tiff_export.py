"""PrintEXP-compatible TIFF export for ON99 white underbase.

Why old files failed
--------------------
Hoson PrintEXP (V5.8.x) Spot mode expects a Photoshop *Spot Color Channel*.
Writing RGB + ExtraSamples and labeling them with ALPHA_NAMES only creates an
Alpha/ExtraSamples channel. PrintEXP then reports ``Invalid image format``.

Factory workflow
----------------
Photoshop → Channels → New Spot Channel, name ``white`` (optional ``varnish``),
Solidity 100% → Save As TIFF with Spot Colors (+ Alpha Channels) checked →
PrintEXP Import with white Color Data Source Type = Spot.

This module writes Adobe TIFF ImageResources (tag 34377) so Photoshop shows a
true Spot Channel (DisplayInfo kind=2 + Alternate Spot Colors + Spot Halftone),
not a plain Alpha name.
"""

from __future__ import annotations

import struct
from datetime import datetime
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

ExportMode = Literal["printexp_spot", "legacy_extrasamples"]

DEFAULT_SPOT_NAME = "white"
DEFAULT_VARNISH_NAME = "varnish"
DEFAULT_DPI = 300
SPOT_KIND = 2  # Photoshop DisplayInfo: 0=selected, 1=protected, 2=spot color
SPOT_START_ID = 3


def _resolution_info_bytes(dpi: float) -> bytes:
    fixed = int(round(float(dpi) * 65536.0)) & 0xFFFFFFFF
    return struct.pack(">IHH IHH", fixed, 1, 1, fixed, 1, 1)


def _srgb_icc_profile() -> bytes | None:
    try:
        from PIL import ImageCms

        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        return None


def _display_info_obsolete(count: int) -> bytes:
    """Resource 1007 — older DisplayInfo with kind=2 (spot) + padding byte."""
    parts: list[bytes] = []
    for _ in range(count):
        # colorspace HSB, vivid overlay, opacity 100, kind=spot, pad
        parts.append(struct.pack(">h4HHBB", 1, 0, 65535, 65535, 0, 100, SPOT_KIND, 0))
    return b"".join(parts)


def _display_info_cs3(count: int) -> bytes:
    """Resource 1077 — CS3+ DisplayInfo; kind=2 marks Spot Color."""
    parts = [struct.pack(">I", 1)]  # version
    for _ in range(count):
        # Match Photoshop-saved spot TIFF (opacity field 0, kind=2).
        parts.append(struct.pack(">h4HHb", 1, 0, 65535, 65535, 0, 0, SPOT_KIND))
    return b"".join(parts)


def _alternate_spot_colors(count: int, start_id: int = SPOT_START_ID) -> bytes:
    """Resource 1067 — Alternate Spot Colors (Lab placeholders)."""
    parts = [struct.pack(">HH", 1, count)]
    for index in range(count):
        channel_id = start_id + index
        parts.append(struct.pack(">Ih4H", channel_id, 7, 0x1535, 0x1F90, 0x1B4E, 0))
    return b"".join(parts)


def _spot_halftone(count: int) -> bytes:
    """Resource 1043 — Spot Halftone, sized like Photoshop RGB/CMYK spot TIFFs."""
    # Copied structure from a Photoshop CC spot TIFF (one screen descriptor).
    base = bytes.fromhex(
        "00060002000000000001000000000000000000000000000000000001000000000000000000000000"
    )
    if count <= 1:
        return base
    # Duplicate screen block for additional spot plates.
    return base + base[4:]


def _alpha_identifiers(count: int, start_id: int = SPOT_START_ID) -> bytes:
    """Non-zero IDs — 0 would mean transparency alpha, not spot."""
    return b"".join(struct.pack(">I", start_id + i) for i in range(count))


def photoshop_spot_image_resources(
    channel_names: list[str],
    dpi: float = DEFAULT_DPI,
) -> tuple:
    """Full ImageResources so Photoshop/PrintEXP treat extras as Spot Colors."""
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
    """Legacy ExtraSamples labeling — Alpha names only (PrintEXP Spot will fail)."""
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
    mode: ExportMode = "printexp_spot",
    channel_name: str = DEFAULT_SPOT_NAME,
    varnish: np.ndarray | None = None,
    varnish_name: str = DEFAULT_VARNISH_NAME,
    compression: str = "none",
) -> bytes:
    """Write RGB TIFF + spot/extra plane(s).

    Parameters
    ----------
    mode:
        ``printexp_spot`` — Photoshop Spot Color metadata (default, PrintEXP).
        ``legacy_extrasamples`` — ALPHA_NAMES only (old App output, for compare).
    channel_name:
        Factory PrintEXP video uses ``white``. Maintop often uses ``W1``.
    varnish:
        Optional second spot plane (e.g. varnish / adhesive).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb 必須是 HxWx3")
    if white.shape != rgb.shape[:2]:
        raise ValueError("white 通道尺寸必須與 RGB 相同")
    if varnish is not None and varnish.shape != white.shape:
        raise ValueError("varnish 尺寸必須與 white 相同")

    names = [channel_name]
    planes: list[np.ndarray] = [rgb, white[..., None]]
    if varnish is not None:
        names.append(varnish_name)
        planes.append(varnish[..., None])

    stacked = np.concatenate(planes, axis=2).astype(np.uint8, copy=False)
    extras = tuple(0 for _ in names)

    if mode == "printexp_spot":
        extratags = [photoshop_spot_image_resources(names, dpi=dpi)]
        software = "Adobe Photoshop 25.0 (Windows)"
    elif mode == "legacy_extrasamples":
        extratags = [photoshop_legacy_alpha_resources(names)]
        software = "tifffile.py"
    else:
        raise ValueError(f"未知 export mode: {mode}")

    options: dict = dict(
        photometric="rgb",
        extrasamples=extras,
        planarconfig="contig",
        metadata=False,
        resolution=(float(dpi), float(dpi)),
        resolutionunit="inch",
        software=software,
        datetime=datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
        extratags=extratags,
    )
    if mode == "printexp_spot":
        icc = _srgb_icc_profile()
        if icc:
            options["iccprofile"] = icc
    if compression == "zip":
        options["compression"] = "adobe_deflate"
    elif compression == "lzw":
        options["compression"] = "lzw"

    from io import BytesIO

    buf = BytesIO()
    imwrite(buf, stacked, **options)
    return buf.getvalue()


def inspect_spot_metadata(file_bytes: bytes) -> dict:
    """Parse TIFF ImageResources and report Spot vs Alpha markers."""
    from io import BytesIO

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

    if ir_bytes is None:
        return {
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
        }

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
            # version(4) + repeated 13-byte records ending with kind
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
    }
