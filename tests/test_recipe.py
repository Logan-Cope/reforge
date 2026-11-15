from reforge.render.recipe import compute_crop, ASPECT

def test_landscape_center_crop_is_9x16():
    cw, ch, x, y = compute_crop(3840, 2160)           # 4K landscape
    assert abs((cw / ch) - ASPECT) < 0.01
    assert ch == 2160 and x > 0 and y == 0            # full height, centered horizontally

def test_zoom_punches_in():
    cw, ch, _, _ = compute_crop(3840, 2160, zoom=0.8)
    assert ch < 2160                                  # smaller window = punch-in
