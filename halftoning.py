import math
import pathlib
import time

import cProfile

from PIL import Image


INK_C, INK_M, INK_Y, INK_W = 0b001, 0b010, 0b100, 0b000
INK_R, INK_G, INK_B, INK_K = 0b110, 0b101, 0b011, 0b111
ink_to_rgb = [(255, 255, 255), (000, 255, 255), (255, 000, 255), (000, 000, 255),
              (255, 255, 000), (000, 255, 000), (255, 000, 000), (000, 000, 000)]

def avg_rgb_to_ratio_of_cmky_ink(rgb, ink):
    percent_c = (255.0 - rgb[0]) / 255.0
    percent_m = (255.0 - rgb[1]) / 255.0
    percent_y = (255.0 - rgb[2]) / 255.0
    percent_k = 0.0 # min(percent_c, percent_m, percent_y)
    percent_c -= percent_k
    percent_m -= percent_k
    percent_y -= percent_k
    if ink == INK_C:
        return percent_c
    if ink == INK_M:
        return percent_m
    if ink == INK_Y:
        return percent_y
    if ink == INK_K:
        return percent_k
    raise ValueError

def print_dot(ink_image, dot):
    x_c, y_c, radius, ink = dot
    x_0, y_0 = int(x_c - radius), int(y_c - radius)
    x_1, y_1 = int(x_c + radius), int(y_c + radius)
    for y in range(y_0, y_1):
        for x in range(x_0, x_1):
            if y < 0 or y >= len(ink_image) or x < 0 or x >= len(ink_image[0]):
                continue
            if math.sqrt((x-x_c)**2 + (y-y_c)**2) > radius:
                continue
            ink_image[y][x] |= ink

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

def render_like_a_math_person(image):
    cmyk_angles = [math.radians(x) for x in [15, 75, 1, 45]]
    cmyk_pitches = [50, 70, 60, 40]
    ink_image = []
    for i in range(image.height):
        ink_image.append([0]*image.width)
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
        # print(f'transformed basis "x" range is ({minxdb},{maxxdb}), "y" range is ({minydb},{maxydb})')
        xdb_boost = 0 if minxdb >= 0 else -1 * minxdb
        ydb_boost = 0 if minydb >= 0 else -1 * minydb
        dots_map = []
        for ydb in range(maxydb+ydb_boost+1):
            dots_map.append([((0.0,0.0,0.0),0)] * (maxxdb+xdb_boost+1))
        # print(f'Dots map dimensions are {len(dots_map)} X {len(dots_map[0])}')
        t1 = time.time()
        for y in range(image.height):
            for x in range(image.width):
                x_db, y_db = apply_matrix_to_vector(pixel_to_ink_transform, (x, y))
                x_db = round(x_db)+xdb_boost
                y_db = round(y_db)+ydb_boost
                rgb = image.getpixel((x, y))
                prev_rgb, prev_sample_count = dots_map[y_db][x_db]
                new_sample_count = prev_sample_count + 1
                new_rgb = ((prev_rgb[0]*prev_sample_count+rgb[0])/new_sample_count,
                           (prev_rgb[1]*prev_sample_count+rgb[1])/new_sample_count,
                           (prev_rgb[2]*prev_sample_count+rgb[2])/new_sample_count)
                dots_map[y_db][x_db] = (new_rgb, new_sample_count)

        t2 = time.time()
        for y_db in range(len(dots_map)):
            for x_db in range(len(dots_map[y_db])):
                rgb_average, _ = dots_map[y_db][x_db]
                ink = [INK_C, INK_M, INK_Y, INK_K][cmyk_idx]
                radius = cmyk_pitches[cmyk_idx] * avg_rgb_to_ratio_of_cmky_ink(rgb_average, ink)
                x_c, y_c = apply_matrix_to_vector(ink_to_pixel_transform, (x_db-xdb_boost, y_db-ydb_boost))
                dot = x_c, y_c, radius, ink
                print_dot(ink_image, dot)

        t3 = time.time()

        print(f'phase 1 took {t2-t1}')
        print(f'phase 2 took {t3-t2}')
    t1 = time.time()
    new_image = Image.new('RGB', (image.width, image.height))
    for y in range(image.height):
        for x in range(image.width):
            new_image.putpixel((x, y), ink_to_rgb[ink_image[y][x]])
    t2 = time.time()
    print(f'final raster took {t2-t1}')
    return new_image

def main():
    out_path = pathlib.Path('outputs')
    if not out_path.exists():
        out_path.mkdir()
    for path in pathlib.Path('sample-inputs').iterdir():
        if path.stem != 'lorem-picsum-3':
            continue # XXX: For now.
        print(f'processing {path}')
        image = Image.open(path)
        t1 = time.time()
        new_image_2 = render_like_a_math_person(image)
        t2 = time.time()
        print(f'Render took {t2-t1} seconds')
        new_image_2.save(out_path / f'{path.stem}-halftoned-new{path.suffix}')

if __name__ == '__main__':
    # cProfile.run('main()')
    main()
