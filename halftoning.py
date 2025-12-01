import math
import numpy as np
import pathlib
import time

from PIL import Image

from numba import jit

# The slower, but pure-numpy way is:
# sum_flat, count_flat = np.bincount(idx, weights=cmyk_images_flat[cmyk_idx], minlength=n_dots), np.bincount(idx, minlength=n_dots)
@jit(nopython=True, fastmath=True)
def fast_bincount_avg(idx, values, n_dots):
    sums, counts = np.zeros(n_dots, dtype=np.float32), np.zeros(n_dots, dtype=np.int32)
    for i in range(len(idx)):
        sums[idx[i]] += values[i]
        counts[idx[i]] += 1
    return sums, counts

def render(image):
    buckets = [0.0] * 4
    t1 = time.time()
    angles = [math.radians(x) for x in [15, 75, 1, 45]]
    pitches = [5, 5, 5, 5]
    black_ratio = 0.1
    rgb = np.ascontiguousarray(image).astype(np.float32) / 255.0
    k_image = (1 - np.max(rgb, axis=2)) * black_ratio
    c_image, m_image, y_image = (1 - rgb[...,0] - k_image), (1 - rgb[...,1] - k_image), (1 - rgb[...,2] - k_image)
    cmyk_images_flat = [c_image.ravel(), m_image.ravel(), y_image.ravel(), k_image.ravel()]
    new_rgb_array = np.ones_like(rgb)
    pixel_to_pixel_coordinates = np.indices((image.height, image.width), dtype=np.float32).transpose(1, 2, 0)
    anti_colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,1]], dtype=np.float32)
    coses_a, sines_a = np.cos(angles), np.sin(angles)
    dot_to_pixel_transforms = np.array([
        [[pitches[i] * coses_a[i], -pitches[i] * sines_a[i]],
        [pitches[i] * sines_a[i],  pitches[i] * coses_a[i]]]
        for i in range(4)
    ], dtype=np.float32)
    pixel_to_dot_transforms = [x.T for x in np.linalg.inv(dot_to_pixel_transforms)]
    ydb_ofs, xdb_ofs, dot_image_heights, dot_image_widths, n_dots = [0]*4, [0]*4, [0]*4, [0]*4, [0]*4
    for cmyk_idx in [0, 1, 2, 3]:
        minxdb, maxxdb, minydb, maxydb = math.inf, -1.0*math.inf, math.inf, -1.0*math.inf
        for sample_point in [(0, 0), (image.height, 0,), (0, image.width), (image.height, image.width)]:
            ydb, xdb = sample_point @ pixel_to_dot_transforms[cmyk_idx]
            ydb, xdb = round(ydb), round(xdb)
            minxdb,          maxxdb,          minydb,          maxydb = \
            min(xdb,minxdb), max(xdb,maxxdb), min(ydb,minydb), max(ydb,maxydb)
        ydb_ofs[cmyk_idx], xdb_ofs[cmyk_idx] = -minydb, -minxdb
        dot_image_heights[cmyk_idx], dot_image_widths[cmyk_idx] = maxydb + ydb_ofs[cmyk_idx] + 1, maxxdb + xdb_ofs[cmyk_idx] + 1
        n_dots[cmyk_idx] = dot_image_heights[cmyk_idx] * dot_image_widths[cmyk_idx]
    buckets[0] += time.time() - t1
    for cmyk_idx in range(4):
        t1 = time.time()
        _pixel_to_dot_coords_float = pixel_to_pixel_coordinates @ pixel_to_dot_transforms[cmyk_idx]
        pixel_to_dot_coords_rounded = np.rint(_pixel_to_dot_coords_float)
        _pixel_to_dot_coords_float -= pixel_to_dot_coords_rounded
        pixel_to_dot_center_distance_sq = np.einsum('ijk,ijk->ij', _pixel_to_dot_coords_float, _pixel_to_dot_coords_float)
        buckets[1] += time.time() - t1
        t1 = time.time()
        pixel_to_dot_coordinates = pixel_to_dot_coords_rounded.astype(np.int32)
        coords_flat = pixel_to_dot_coordinates.reshape(-1, 2)
        idx = (coords_flat[:, 0] + ydb_ofs[cmyk_idx]) * dot_image_widths[cmyk_idx] + (coords_flat[:, 1] + xdb_ofs[cmyk_idx])
        sum_flat, count_flat = fast_bincount_avg(idx, cmyk_images_flat[cmyk_idx], n_dots[cmyk_idx])
        dot_radii_sq = np.divide(sum_flat, count_flat, out=np.zeros_like(sum_flat, dtype=float), where=(count_flat != 0)) ** 2
        pixel_to_dot_radius_sq = dot_radii_sq[idx].reshape(image.height, image.width)
        buckets[2] += time.time() - t1
        t1 = time.time()
        subtractive_image = np.zeros_like(new_rgb_array)
        subtractive_image[pixel_to_dot_radius_sq >= pixel_to_dot_center_distance_sq] = anti_colors[cmyk_idx]
        new_rgb_array -= subtractive_image
        buckets[3] += time.time() - t1

    for bucket in buckets:
        print(f'{"#"*round(bucket*100)} {round(bucket*1000)}ms')
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
        if len(times) > 1:
            print(f'average so far: {round( sum(times[1:]) / (len(times)-1) ) }ms')
        new_image_2.save(out_path / f'{path.stem}-halftoned-new{path.suffix}')
    print(f'average sans first run: {round( sum(times[1:]) / (len(times)-1) ) }ms')

if __name__ == '__main__':
    main()
