"""Tests for PrintEXP Spot Color TIFF export."""

from __future__ import annotations

import numpy as np

from tiff_export import (
    inspect_spot_metadata,
    safe_download_stem,
    write_tiff_with_spot,
)


def _sample():
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[4:20, 4:20] = (220, 40, 40)
    white = np.zeros((24, 24), dtype=np.uint8)
    white[4:20, 4:20] = 255
    alpha = np.zeros((24, 24), dtype=np.uint8)
    alpha[4:20, 4:20] = 255
    return rgb, white, alpha


def test_default_cmyk_spot_named_white():
    rgb, white, alpha = _sample()
    data = write_tiff_with_spot(
        rgb,
        white,
        300,
        mode="printexp_cmyk_spot",
        channel_name="white",
        alpha=alpha,
    )
    meta = inspect_spot_metadata(data)
    assert meta["color_space"] == "cmyk"
    assert meta["photometric"] == 5
    assert meta["samples"] == 5
    assert meta["names"] == ["white"]
    assert meta["is_photoshop_spot"]
    assert meta["spot_kinds"] == [2]
    assert "Photoshop" in meta["software"]


def test_rgb_spot_mode():
    rgb, white, alpha = _sample()
    data = write_tiff_with_spot(
        rgb,
        white,
        300,
        mode="printexp_rgb_spot",
        channel_name="white",
        alpha=alpha,
    )
    meta = inspect_spot_metadata(data)
    assert meta["color_space"] == "rgb"
    assert meta["samples"] == 4
    assert meta["is_photoshop_spot"]


def test_legacy_is_not_spot():
    rgb, white, alpha = _sample()
    data = write_tiff_with_spot(
        rgb,
        white,
        300,
        mode="legacy_extrasamples",
        channel_name="white",
        alpha=alpha,
    )
    meta = inspect_spot_metadata(data)
    assert not meta["is_photoshop_spot"]


def test_safe_download_stem_strips_parens():
    assert "(" not in safe_download_stem("vintage_print (12).png")
    assert " " not in safe_download_stem("vintage_print (12).png")
    assert safe_download_stem("vintage_print (12).png") == "vintage_print_12"


def test_spot_invert_and_mirror_helpers():
    from app import mirror_planes, spot_looks_inverted, spot_plane_for_export

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :4] = (200, 40, 40)
    alpha = np.zeros((8, 8), dtype=np.uint8)
    alpha[:, :4] = 255
    coverage = alpha.copy()
    white = alpha.copy()

    rgb_m, alpha_m, coverage_m, white_m = mirror_planes(rgb, alpha, coverage, white)
    assert alpha_m[0, 0] == 0 and alpha_m[0, 7] == 255
    assert white_m[0, 7] == 255

    exported = spot_plane_for_export(white, invert=True)
    assert exported[0, 0] == 0 and exported[0, 7] == 255
    assert spot_looks_inverted(exported, support=alpha)
    assert not spot_looks_inverted(white, support=alpha)


def test_offset_white_x_shifts_right_without_wrap():
    from app import build_white_channel, offset_white_x

    mask = np.zeros((6, 10), dtype=np.uint8)
    mask[:, 2:5] = 200

    right = offset_white_x(mask, 2)
    assert np.all(right[:, :4] == 0)
    assert np.all(right[:, 4:7] == 200)
    assert np.all(right[:, 7:] == 0)

    left = offset_white_x(mask, -2)
    assert np.all(left[:, 0:3] == 200)
    assert np.all(left[:, 3:] == 0)

    assert np.array_equal(offset_white_x(mask, 0), mask)

    alpha = np.zeros((8, 8), dtype=np.uint8)
    alpha[2:6, 2:5] = 255
    _coverage, white = build_white_channel(
        alpha, choke_px=0, polarity="white_prints", white_x_offset_px=2
    )
    assert np.all(white[:, :4] == 0)
    assert np.all(white[2:6, 4:7] == 255)


def test_soften_white_bottom_only_affects_lower_tenth():
    from app import soften_white_bottom

    mask = np.full((100, 8), 200, dtype=np.uint8)
    out = soften_white_bottom(mask, strength=10)
    assert np.array_equal(out[:90], mask[:90])
    assert out[99, 0] < out[90, 0] <= 200
    assert np.array_equal(soften_white_bottom(mask, 0), mask)

    alpha = np.full((100, 8), 255, dtype=np.uint8)
    from app import build_white_channel

    _c, white = build_white_channel(
        alpha, choke_px=0, polarity="white_prints", bottom_white_fade=5
    )
    assert int(white[0, 0]) == 255
    assert int(white[99, 0]) < 255


if __name__ == "__main__":
    test_default_cmyk_spot_named_white()
    test_rgb_spot_mode()
    test_legacy_is_not_spot()
    test_safe_download_stem_strips_parens()
    test_spot_invert_and_mirror_helpers()
    test_offset_white_x_shifts_right_without_wrap()
    test_soften_white_bottom_only_affects_lower_tenth()
    print("OK")
