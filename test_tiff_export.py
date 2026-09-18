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


def test_lead_in_bar_centered_expands_canvas():
    from app import (
        LEAD_IN_GAP_MM,
        LEAD_IN_HEIGHT_MM,
        LEAD_IN_WIDTH_MM,
        apply_lead_in_bar,
        mm_to_px,
        rgb_alpha_to_cmyk,
    )

    dpi = 300.0
    assert mm_to_px(25.4, 300) == 300
    assert mm_to_px(LEAD_IN_WIDTH_MM, dpi) == mm_to_px(40.0, dpi)

    h, w = 80, 120
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    # Content block near bottom-center
    alpha[20:60, 40:80] = 255
    rgb[20:60, 40:80] = (200, 40, 40)
    coverage = alpha.copy()
    white = alpha.copy()

    rgb2, alpha2, coverage2, white2 = apply_lead_in_bar(
        rgb, alpha, coverage, white, dpi=dpi, polarity="white_prints"
    )
    gap = mm_to_px(LEAD_IN_GAP_MM, dpi)
    bar_h = mm_to_px(LEAD_IN_HEIGHT_MM, dpi)
    bar_w = mm_to_px(LEAD_IN_WIDTH_MM, dpi)
    y1 = 59
    bar_top = y1 + 1 + gap
    expected_h = max(h, bar_top + bar_h)
    assert alpha2.shape[0] == expected_h
    assert alpha2.shape[1] >= w
    assert alpha2.shape[1] >= bar_w
    # Original content preserved (may shift right if canvas padded left for centering)
    assert int(alpha2[:h, :].max()) == 255
    assert int((alpha2[:h] > 0).sum()) == int((alpha > 0).sum())
    # Bar region has full white underbase near horizontal center
    cy = bar_top + bar_h // 2
    cx = alpha2.shape[1] // 2
    assert int(white2[cy, cx]) == 255
    assert int(alpha2[cy, cx]) == 255
    # CMYK export path sees ink in bar (all channels get some ink across segments)
    cmyk = rgb_alpha_to_cmyk(rgb2, alpha2)
    bar_slice = cmyk[bar_top : bar_top + bar_h]
    assert bar_slice[:, :, 0].max() > 0  # C
    assert bar_slice[:, :, 1].max() > 0  # M
    assert bar_slice[:, :, 2].max() > 0  # Y
    assert bar_slice[:, :, 3].max() > 0  # K
    assert bar_w >= 100


def test_registration_guides_color_block_height_graphic_x():
    from app import (
        GUIDE_ARM_MM,
        GUIDE_OFFSET_MM,
        GUIDE_STROKE_PX,
        apply_registration_guides,
        color_block_bbox,
        mm_to_px,
        rgb_alpha_to_cmyk,
    )

    dpi = 300.0
    offset = mm_to_px(GUIDE_OFFSET_MM, dpi)
    arm = mm_to_px(GUIDE_ARM_MM, dpi)
    assert arm == mm_to_px(4.0, dpi)
    assert GUIDE_STROKE_PX == 2
    assert GUIDE_OFFSET_MM == 10.0

    h, w = 120, 160
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    # Graphic (near-white) wider/taller
    gx0, gy0, gx1, gy1 = 30, 15, 129, 104
    alpha[gy0 : gy1 + 1, gx0 : gx1 + 1] = 255
    rgb[gy0 : gy1 + 1, gx0 : gx1 + 1] = (255, 255, 255)
    # Color Block (chromatic) — stem height follows this
    cx0, cy0, cx1, cy1 = 45, 35, 114, 84
    rgb[cy0 : cy1 + 1, cx0 : cx1 + 1] = (180, 60, 60)
    alpha[cy0 : cy1 + 1, cx0 : cx1 + 1] = 255
    coverage = alpha.copy()
    white = alpha.copy()
    block_h = cy1 - cy0 + 1

    detected = color_block_bbox(rgb, alpha)
    assert detected == (cx0, cy0, cx1, cy1)

    rgb2, alpha2, coverage2, white2, graphic2, color2 = apply_registration_guides(
        rgb,
        alpha,
        coverage,
        white,
        dpi=dpi,
        polarity="white_prints",
        graphic_box=(gx0, gy0, gx1, gy1),
        color_block_box=(cx0, cy0, cx1, cy1),
    )
    assert graphic2 is not None and color2 is not None
    assert color2[3] - color2[1] + 1 == block_h
    assert alpha2.shape[0] == h  # no top bar

    g2x0, _, g2x1, _ = graphic2
    left_x = g2x0 - offset
    right_x = g2x1 + offset
    by0, by1 = color2[1], color2[3]
    assert int(alpha2[by0, left_x]) == 255
    assert int(alpha2[by1, left_x]) == 255
    assert int(alpha2[by0 - 1, left_x]) == 0
    assert by1 < graphic2[3]
    assert int(alpha2[by1 + 1, left_x]) == 0
    assert int(alpha2[by0, left_x + arm - 1]) == 255
    assert int(alpha2[by0, right_x - arm + 1]) == 255
    assert int(white2[by0, left_x]) == 0
    assert int(coverage2[by0, left_x]) == 0
    cmyk = rgb_alpha_to_cmyk(rgb2, alpha2)
    assert int(cmyk[by0, left_x, 3]) > 200


def test_color_block_bbox_ignores_gray_distress():
    from app import color_block_bbox, graphic_bbox

    h, w = 80, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[10:70, 10:90] = 255
    rgb[10:70, 10:90] = (255, 255, 255)
    # gray grit should NOT become Color Block
    rgb[12, 12] = (200, 200, 200)
    rgb[60, 80] = (180, 180, 180)
    assert color_block_bbox(rgb, alpha) is None
    # real chroma block is detected
    rgb[25:55, 30:70] = (20, 90, 200)
    assert color_block_bbox(rgb, alpha) == (30, 25, 69, 54)
    assert graphic_bbox(alpha) == (10, 10, 89, 69)


def test_no_color_block_skips_registration_guides():
    from app import GUIDE_OFFSET_MM, apply_registration_guides, mm_to_px

    dpi = 300.0
    offset = mm_to_px(GUIDE_OFFSET_MM, dpi)
    h, w = 80, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[10:70, 10:90] = 255
    rgb[10:70, 10:90] = (255, 255, 255)
    coverage = alpha.copy()
    white = alpha.copy()

    rgb2, alpha2, coverage2, white2, graphic2, color2 = apply_registration_guides(
        rgb, alpha, coverage, white, dpi=dpi, polarity="white_prints"
    )
    assert graphic2 == (10, 10, 89, 69)
    assert color2 is None
    # No guides drawn outside graphic
    left_x = 10 - offset
    assert left_x < 0 or int(alpha2[:, max(0, left_x)].max()) == 0
    assert np.array_equal(rgb2, rgb)
    assert int(alpha2.sum()) == int(alpha.sum())


if __name__ == "__main__":
    test_default_cmyk_spot_named_white()
    test_rgb_spot_mode()
    test_legacy_is_not_spot()
    test_safe_download_stem_strips_parens()
    test_spot_invert_and_mirror_helpers()
    test_offset_white_x_shifts_right_without_wrap()
    test_soften_white_bottom_only_affects_lower_tenth()
    test_lead_in_bar_centered_expands_canvas()
    test_registration_guides_color_block_height_graphic_x()
    test_color_block_bbox_ignores_gray_distress()
    test_no_color_block_skips_registration_guides()
    print("OK")
