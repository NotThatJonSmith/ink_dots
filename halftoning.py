import math
import pathlib
import time

import cProfile

from PIL import Image


INK_C, INK_M, INK_Y, INK_W = 0b001, 0b010, 0b100, 0b000
INK_R, INK_G, INK_B, INK_K = 0b110, 0b101, 0b011, 0b111
ink_to_rgb = [(255, 255, 255), (000, 255, 255), (255, 000, 255), (000, 000, 255),
              (255, 255, 000), (000, 255, 000), (255, 000, 000), (000, 000, 000)]

def apply_matrix_to_vector(matrix, vector):
    x1, x2  = vector
    a11 = matrix[0][0]
    a12 = matrix[0][1]
    a21 = matrix[1][0]
    a22 = matrix[1][1]
    return a11*x1 + a12*x2, a21*x1 + a22*x2

def invert_2x2(M):
    a, b = M[0]
    c, d = M[1]
    det = a*d - b*c
    if abs(det) < 1e-12:
        raise ValueError("Matrix not invertible")
    s = 1.0 / det
    return [
        [ d*s, -b*s ],
        [ -c*s, a*s ]
    ]

def rgb_to_cmyk(rgb):
    r, g, b = rgb
    c = (255.0 - r) / 255.0
    m = (255.0 - g) / 255.0
    y = (255.0 - b) / 255.0
    k = min(c, m, y) / 10.0
    c -= k
    m -= k
    y -= k
    return c, m, y, k

import numpy as np

def render_like_a_math_person(image):
    cmyk_angles = [math.radians(x) for x in [15, 75, 1, 45]]
    cmyk_pitches = [5,5,5,5]
    ink_image = []
    for i in range(image.height):
        ink_image.append([0]*image.width)

    t1 = time.time()
    rgb = np.array(image).astype(float) / 255.0
    K = (1 - np.max(rgb, axis=2)) / 10.0
    C = (1 - rgb[...,0] - K)
    M = (1 - rgb[...,1] - K)
    Y = (1 - rgb[...,2] - K)
    cmyk_image = (np.dstack((C, M, Y, K))).astype(float)
    t2 = time.time()

    for cmyk_idx in [0, 1, 2, 3]:
        ink_grid_basis_vectors = (cmyk_pitches[cmyk_idx]*math.cos(cmyk_angles[cmyk_idx]),
                                  cmyk_pitches[cmyk_idx]*math.sin(cmyk_angles[cmyk_idx])), \
                                 (cmyk_pitches[cmyk_idx]*math.cos(cmyk_angles[cmyk_idx] + math.radians(90.0)),
                                  cmyk_pitches[cmyk_idx]*math.sin(cmyk_angles[cmyk_idx] + math.radians(90.0)))
        ink_to_pixel_transform = [[ink_grid_basis_vectors[0][0],ink_grid_basis_vectors[1][0]],
                                  [ink_grid_basis_vectors[0][1],ink_grid_basis_vectors[1][1]]]
        pixel_to_ink_transform = invert_2x2(ink_to_pixel_transform)
        minxdb, maxxdb, minydb, maxydb = math.inf, -1.0*math.inf, math.inf, -1.0*math.inf
        for sample_point in [(0, 0), (0, image.height), (image.width, 0), (image.width, image.height)]:
            xdb, ydb = apply_matrix_to_vector(pixel_to_ink_transform, sample_point)
            xdb, ydb = round(xdb), round(ydb)
            minxdb, maxxdb, minydb, maxydb = min(xdb,minxdb), max(xdb,maxxdb), min(ydb,minydb), max(ydb,maxydb)
        xdb_boost = 0 if minxdb >= 0 else -1 * minxdb
        ydb_boost = 0 if minydb >= 0 else -1 * minydb
        dots_map = [(0.0,0)] * (maxxdb+xdb_boost+1) * (maxydb+ydb_boost+1)
        t1 = time.time()
        for y in range(image.height):
            for x in range(image.width):
                x_db, y_db = apply_matrix_to_vector(pixel_to_ink_transform, (x, y))
                x_db, y_db = round(x_db)+xdb_boost, round(y_db)+ydb_boost
                dot_index = y_db * (maxxdb+xdb_boost+1) + x_db
                value = cmyk_image[y,x,cmyk_idx]
                prev_cmyk, prev_sample_count = dots_map[dot_index]
                new_sample_count = prev_sample_count + 1
                new_value = (prev_cmyk*prev_sample_count+value)/new_sample_count
                dots_map[dot_index] = (new_value, new_sample_count)
        t2 = time.time()
        print(f'phase 1 took {t2-t1}')
        t1 = time.time()
        for dot_index in range(len(dots_map)):
            value, _ = dots_map[dot_index]
            ink = [INK_C, INK_M, INK_Y, INK_K][cmyk_idx]
            radius = cmyk_pitches[cmyk_idx] * value
            y_db = dot_index / (maxxdb+xdb_boost+1)
            x_db = dot_index % (maxxdb+xdb_boost+1)
            x_c, y_c = apply_matrix_to_vector(ink_to_pixel_transform, (x_db-xdb_boost, y_db-ydb_boost))
            x_0, y_0 = max(0, int(x_c - radius)), max(0, int(y_c - radius))
            x_1, y_1 = min(len(ink_image[0]), int(x_c + radius)), min(len(ink_image), int(y_c + radius))
            for y in range(y_0, y_1):
                for x in range(x_0, x_1):
                    if math.sqrt((x-x_c)**2 + (y-y_c)**2) > radius:
                        continue
                    ink_image[y][x] |= ink
        t2 = time.time()
        print(f'phase 2 took {t2-t1}')
    t1 = time.time()
    ink_array = np.asarray(ink_image)                 # shape (H, W)
    ink_lut = np.asarray(ink_to_rgb, dtype=np.uint8)  # shape (8, 3)
    rgb_array = ink_lut[ink_array]                    # fancy indexing
    new_image = Image.fromarray(rgb_array, mode="RGB")
    t2 = time.time()
    print(f'final raster took {t2-t1}')
    return new_image

def main():
    out_path = pathlib.Path('outputs')
    if not out_path.exists():
        out_path.mkdir()
    for path in pathlib.Path('sample-inputs').iterdir():
        # if int(path.stem.split('-')[-1]) > 3:
        #     continue # XXX: For now.
        print(f'processing {path}')
        image = Image.open(path)
        t1 = time.time()
        new_image_2 = render_like_a_math_person(image)
        t2 = time.time()
        print(f'Render took {t2-t1} seconds')
        new_image_2.save(out_path / f'{path.stem}-halftoned-new{path.suffix}')

if __name__ == '__main__':
    main()
