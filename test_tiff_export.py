"""Tests for PrintEXP Spot Color TIFF export."""

from __future__ import annotations

import numpy as np

from tiff_export import inspect_spot_metadata, write_tiff_with_spot


def _sample_rgb_white():
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[4:20, 4:20] = (220, 40, 40)
    white = np.zeros((24, 24), dtype=np.uint8)
    white[4:20, 4:20] = 255
    return rgb, white


def test_printexp_spot_is_photoshop_spot_named_white():
    rgb, white = _sample_rgb_white()
    data = write_tiff_with_spot(
        rgb, white, 300, mode="printexp_spot", channel_name="white"
    )
    meta = inspect_spot_metadata(data)
    assert meta["photometric"] == 2  # RGB
    assert meta["samples"] == 4
    assert meta["names"] == ["white"]
    assert meta["has_display_info"]
    assert meta["has_display_info_obsolete"]
    assert meta["has_alternate_spot"]
    assert meta["has_spot_halftone"]
    assert meta["spot_kinds"] == [2]
    assert meta["is_photoshop_spot"]
    assert "Photoshop" in meta["software"]


def test_legacy_extrasamples_is_not_photoshop_spot():
    rgb, white = _sample_rgb_white()
    data = write_tiff_with_spot(
        rgb, white, 300, mode="legacy_extrasamples", channel_name="white"
    )
    meta = inspect_spot_metadata(data)
    assert meta["names"] == ["white"]
    assert not meta["has_display_info"]
    assert not meta["has_alternate_spot"]
    assert not meta["is_photoshop_spot"]


def test_optional_varnish_spot():
    rgb, white = _sample_rgb_white()
    data = write_tiff_with_spot(
        rgb,
        white,
        300,
        mode="printexp_spot",
        channel_name="white",
        varnish=white,
        varnish_name="varnish",
    )
    meta = inspect_spot_metadata(data)
    assert meta["names"] == ["white", "varnish"]
    assert meta["samples"] == 5
    assert meta["spot_kinds"] == [2, 2]
    assert meta["is_photoshop_spot"]


if __name__ == "__main__":
    test_printexp_spot_is_photoshop_spot_named_white()
    test_legacy_extrasamples_is_not_photoshop_spot()
    test_optional_varnish_spot()
    print("OK")
