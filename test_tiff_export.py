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
    from app import mirror_planes, spot_plane_for_export

    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    alpha = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    coverage = alpha.copy()
    white = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    _rgb2, _a2, _c2, w2 = mirror_planes(rgb, alpha, coverage, white)
    assert int(w2[0, 0]) == 20
    inv = spot_plane_for_export(white, True)
    assert int(inv[0, 0]) == 245


def test_offset_white_x_shifts_right_without_wrap():
    from app import offset_white_x

    m = np.zeros((3, 5), dtype=np.uint8)
    m[:, 1] = 255
    out = offset_white_x(m, 2)
    assert int(out[0, 3]) == 255
    assert int(out[0, 1]) == 0


def test_soften_white_bottom_only_affects_lower_tenth():
    from app import soften_white_bottom

    white = np.full((100, 4), 255, dtype=np.uint8)
    out = soften_white_bottom(white, 10)
    assert int(out[0, 0]) == 255
    assert int(out[99, 0]) < 255


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
    assert int(alpha2[:h, :].max()) == 255
    assert int((alpha2[:h] > 0).sum()) == int((alpha > 0).sum())
    cy = bar_top + bar_h // 2
    cx = alpha2.shape[1] // 2
    assert int(white2[cy, cx]) == 255
    assert int(alpha2[cy, cx]) == 255
    cmyk = rgb_alpha_to_cmyk(rgb2, alpha2)
    bar_slice = cmyk[bar_top : bar_top + bar_h]
    assert bar_slice[:, :, 0].max() > 0
    assert bar_slice[:, :, 1].max() > 0
    assert bar_slice[:, :, 2].max() > 0
    assert bar_slice[:, :, 3].max() > 0
    assert bar_w >= 100


def test_guides_x_on_graphic_height_on_color_block():
    """X = 大圖案外側 1cm；豎段高度 = Color Block。無色塊唔畫。"""
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
    assert GUIDE_STROKE_PX == 2
    assert GUIDE_OFFSET_MM == 10.0
    assert GUIDE_ARM_MM == 4.0

    h, w = 160, 200
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    # Wide white graphic (大圖案)
    gx0, gy0, gx1, gy1 = 10, 10, 189, 140
    alpha[gy0 : gy1 + 1, gx0 : gx1 + 1] = 255
    rgb[gy0 : gy1 + 1, gx0 : gx1 + 1] = (255, 255, 255)

    # Narrow solid Color Block — shorter than graphic
    cx0, cy0, cx1, cy1 = 70, 50, 129, 99
    rgb[cy0 : cy1 + 1, cx0 : cx1 + 1] = (180, 60, 60)
    alpha[cy0 : cy1 + 1, cx0 : cx1 + 1] = 255

    detected = color_block_bbox(rgb, alpha)
    assert detected is not None, detected
    dx0, dy0, dx1, dy1 = detected
    assert abs(dx0 - cx0) <= 4 and abs(dy0 - cy0) <= 4
    assert abs(dx1 - cx1) <= 4 and abs(dy1 - cy1) <= 4

    coverage = alpha.copy()
    white = alpha.copy()
    rgb2, alpha2, _c2, white2, graphic2, color2 = apply_registration_guides(
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
    g2x0, _g2y0, g2x1, _g2y1 = graphic2
    bx0, by0, bx1, by1 = color2
    assert by1 - by0 + 1 == (cy1 - cy0 + 1)
    assert bx1 - bx0 + 1 == (cx1 - cx0 + 1)

    left_x = g2x0 - offset
    right_x = g2x1 + offset
    # X follows graphic, NOT color block
    assert left_x != bx0 - offset
    assert right_x != bx1 + offset

    # Stem height = Color Block only
    assert int(alpha2[by0, left_x]) == 255
    assert int(alpha2[by1, left_x]) == 255
    assert int(alpha2[by0, right_x]) == 255
    assert int(alpha2[by1, right_x]) == 255
    if by0 > 0:
        assert int(alpha2[by0 - 1, left_x]) == 0
    if by1 + 1 < alpha2.shape[0]:
        assert int(alpha2[by1 + 1, left_x]) == 0
    # Must not span full graphic height
    assert by0 > graphic2[1]
    assert by1 < graphic2[3]

    assert int(alpha2[by0, left_x + arm - 1]) == 255
    assert int(alpha2[by0, right_x - arm + 1]) == 255
    assert int(white2[by0, left_x]) == 255  # white underbase for RIP / mask tape
    assert tuple(int(v) for v in rgb2[by0, left_x]) == (0, 0, 0)
    cmyk = rgb_alpha_to_cmyk(rgb2, alpha2)
    assert int(cmyk[by0, left_x, 3]) > 200
    assert alpha2.shape[0] == h  # no top bar


def test_color_block_detects_solid_grey_ignores_distress():
    from app import color_block_bbox

    h, w = 100, 120
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    rng = np.random.default_rng(0)
    alpha[5:70, 5:115] = 255
    noise = rng.integers(40, 220, size=(65, 110, 3), dtype=np.uint8)
    rgb[5:70, 5:115] = noise

    rgb[78:92, 40:80] = (160, 160, 160)
    alpha[78:92, 40:80] = 255

    box = color_block_bbox(rgb, alpha)
    assert box is not None
    x0, y0, x1, y1 = box
    assert y0 >= 78 and y1 <= 91
    assert x0 >= 40 and x1 <= 79


def test_white_color_block_under_distress_is_detected():
    from app import color_block_bbox

    h, w = 100, 120
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    rng = np.random.default_rng(1)
    alpha[5:70, 5:115] = 255
    rgb[5:70, 5:115] = rng.integers(40, 220, size=(65, 110, 3), dtype=np.uint8)
    rgb[78:92, 35:85] = (255, 255, 255)
    alpha[78:92, 35:85] = 255
    box = color_block_bbox(rgb, alpha)
    assert box is not None, box
    x0, y0, x1, y1 = box
    assert y0 >= 77 and y1 <= 92
    assert x0 >= 34 and x1 <= 85


def test_white_only_art_still_draws_guides_at_graphic_height():
    from app import GUIDE_OFFSET_MM, apply_registration_guides, color_block_bbox, mm_to_px

    dpi = 300.0
    offset = mm_to_px(GUIDE_OFFSET_MM, dpi)
    h, w = 80, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[10:70, 10:90] = 255
    rgb[10:70, 10:90] = (255, 255, 255)
    assert color_block_bbox(rgb, alpha) is None

    rgb2, alpha2, _c, white2, g2, c2 = apply_registration_guides(
        rgb, alpha, alpha.copy(), alpha.copy(), dpi=dpi
    )
    assert g2 is not None and c2 is not None
    assert c2[1] == g2[1] and c2[3] == g2[3]
    assert g2[3] - g2[1] == 69 - 10
    left_x = g2[0] - offset
    assert int(alpha2[g2[1], left_x]) == 255
    assert int(alpha2[g2[3], left_x]) == 255
    assert int(white2[g2[1], left_x]) == 255
    assert tuple(int(v) for v in rgb2[g2[1], left_x]) == (0, 0, 0)


if __name__ == "__main__":
    test_default_cmyk_spot_named_white()
    test_rgb_spot_mode()
    test_legacy_is_not_spot()
    test_safe_download_stem_strips_parens()
    test_spot_invert_and_mirror_helpers()
    test_offset_white_x_shifts_right_without_wrap()
    test_soften_white_bottom_only_affects_lower_tenth()
    test_lead_in_bar_centered_expands_canvas()
    test_guides_x_on_graphic_height_on_color_block()
    test_color_block_detects_solid_grey_ignores_distress()
    test_white_color_block_under_distress_is_detected()
    test_white_only_art_still_draws_guides_at_graphic_height()
    print("OK")
