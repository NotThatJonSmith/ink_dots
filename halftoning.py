import math
import numpy as np
import pathlib
import time

from PIL import Image

from numba import jit

@jit(nopython=True, fastmath=True)
def fast_bincount_avg(idx, values, n_dots):
    sums = np.zeros(n_dots, dtype=np.float32)
    counts = np.zeros(n_dots, dtype=np.int32)
    for i in range(len(idx)):
        sums[idx[i]] += values[i]
        counts[idx[i]] += 1
    return sums, counts

def render(image):
    cmyk_angles = [math.radians(x) for x in [15, 75, 1, 45]]
    cmyk_pitches = [5, 5, 5, 5]
    black_ratio = 0.1
    rgb = np.ascontiguousarray(image).astype(np.float32) / 255.0
    k_image = (1 - np.max(rgb, axis=2)) * black_ratio
    c_image, m_image, y_image = (1 - rgb[...,0] - k_image), (1 - rgb[...,1] - k_image), (1 - rgb[...,2] - k_image)
    new_rgb_array = np.ones_like(rgb)
    pixel_to_pixel_coordinates = np.indices((image.height, image.width)).transpose(1, 2, 0)
    anti_colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,1]], dtype=np.float32)
    for cmyk_idx in [0, 1, 2, 3]:

        t1 = time.time() # JDS-Comment
        angle = cmyk_angles[cmyk_idx]
        pitch = cmyk_pitches[cmyk_idx]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dot_to_pixel_transform = np.array([
            [pitch * cos_a, -pitch * sin_a],
            [pitch * sin_a, pitch * cos_a]
        ], dtype=np.float32)
        pixel_to_dot_transform = np.linalg.inv(dot_to_pixel_transform)
        minxdb, maxxdb, minydb, maxydb = math.inf, -1.0*math.inf, math.inf, -1.0*math.inf
        for sample_point in [(0, 0), (image.height, 0,), (0, image.width), (image.height, image.width)]:
            ydb, xdb = sample_point @ pixel_to_dot_transform.T
            ydb, xdb = round(ydb), round(xdb)
            minxdb,          maxxdb,          minydb,          maxydb = \
            min(xdb,minxdb), max(xdb,maxxdb), min(ydb,minydb), max(ydb,maxydb)
        ydb_ofs, xdb_ofs = -minydb, -minxdb
        dot_image_height, dot_image_width = maxydb + ydb_ofs + 1, maxxdb + xdb_ofs + 1
        t2 = time.time() # JDS-Comment
        print('1: '+'#'*round((t2-t1)*500)+f' {round((t2-t1) * 1000)}ms') # JDS-Comment

        t1 = time.time() # JDS-Comment
        pixel_to_dot_coords_float = pixel_to_pixel_coordinates @ pixel_to_dot_transform.T
        pixel_to_dot_coords_rounded = np.rint(pixel_to_dot_coords_float)
        pixel_to_dot_coordinates = (pixel_to_dot_coords_rounded + (ydb_ofs, xdb_ofs)).astype(np.int32)
        y = pixel_to_dot_coordinates[..., 0].ravel()
        x = pixel_to_dot_coordinates[..., 1].ravel()
        v = [c_image, m_image, y_image, k_image][cmyk_idx].ravel()
        idx = y * dot_image_width + x
        n_dots = dot_image_height * dot_image_width
        t2 = time.time() # JDS-Comment
        print('2: '+'#'*round((t2-t1)*500)+f' {round((t2-t1) * 1000)}ms') # JDS-Comment

        t1 = time.time() # JDS-Comment
        # numba-jit way - saves about 50ms per render: sum_flat, count_flat = fast_bincount_avg(idx, v, n_dots)
        sum_flat = np.bincount(idx, weights=v, minlength=n_dots)
        count_flat = np.bincount(idx, minlength=n_dots)
        dot_radii_sq = np.divide(sum_flat, count_flat, out=np.zeros_like(sum_flat, dtype=float), where=(count_flat != 0))
        dot_radii_sq = dot_radii_sq.reshape(dot_image_height, dot_image_width)
        dot_radii_sq = (dot_radii_sq * cmyk_pitches[cmyk_idx]) ** 2
        pixel_to_dot_radius_sq = dot_radii_sq[pixel_to_dot_coordinates[...,0], pixel_to_dot_coordinates[...,1]]
        t2 = time.time() # JDS-Comment
        print('3: '+'#'*round((t2-t1)*500)+f' {round((t2-t1) * 1000)}ms') # JDS-Comment
    
        t1 = time.time() # JDS-Comment
        dot_space_offset = pixel_to_dot_coords_float - pixel_to_dot_coords_rounded
        pixel_to_dot_center_distance_sq = np.sum(dot_space_offset**2, axis=2) * (pitch**2)
        t2 = time.time() # JDS-Comment
        print('4: '+'#'*round((t2-t1)*500)+f' {round((t2-t1) * 1000)}ms') # JDS-Comment

        t1 = time.time() # JDS-Comment
        subtractive_image = np.zeros_like(new_rgb_array)
        subtractive_image[pixel_to_dot_radius_sq >= pixel_to_dot_center_distance_sq] = anti_colors[cmyk_idx]
        new_rgb_array -= subtractive_image
        t2 = time.time() # JDS-Comment
        print('5: '+'#'*round((t2-t1)*500)+f' {round((t2-t1) * 1000)}ms') # JDS-Comment
        print() # JDS-Comment

    return Image.fromarray((new_rgb_array * 255.0).astype(np.uint8), mode="RGB")

def main():
    out_path = pathlib.Path('outputs')
    if not out_path.exists():
        out_path.mkdir()
    times = []
    for path in pathlib.Path('sample-inputs').iterdir():
        print(f'processing {path}')
        image = Image.open(path)
        t1 = time.time()
        new_image_2 = render(image)
        t2 = time.time()
        print(f'Render took {round((t2-t1)*1000)}ms')
        times.append((t2-t1)*1000)
        new_image_2.save(out_path / f'{path.stem}-halftoned-new{path.suffix}')
    print(f'average sans first run: {round( sum(times[1:]) / (len(times)-1) ) }ms')
if __name__ == '__main__':
    main()
