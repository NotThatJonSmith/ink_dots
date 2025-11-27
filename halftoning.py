import math
import numpy as np
import pathlib
import time

from PIL import Image

def render(image):
    cmyk_angles = [math.radians(x) for x in [15, 75, 1, 45]]
    cmyk_pitches = [5, 5, 5, 5]
    black_ratio = 0.1

    rgb = np.array(image).astype(np.float32) / 255.0
    k_image = (1 - np.max(rgb, axis=2)) * black_ratio
    c_image, m_image, y_image = (1 - rgb[...,0] - k_image), (1 - rgb[...,1] - k_image), (1 - rgb[...,2] - k_image)
    new_rgb_array = np.ones_like(rgb)
    for cmyk_idx in [0, 1, 2, 3]:
        anti_color = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)][cmyk_idx]
        dot_grid_basis_vectors = (cmyk_pitches[cmyk_idx]*math.cos(cmyk_angles[cmyk_idx]), cmyk_pitches[cmyk_idx]*math.sin(cmyk_angles[cmyk_idx])), \
                                 (cmyk_pitches[cmyk_idx]*math.cos(cmyk_angles[cmyk_idx] + math.radians(90.0)), cmyk_pitches[cmyk_idx]*math.sin(cmyk_angles[cmyk_idx] + math.radians(90.0)))
        dot_to_pixel_transform = np.array([[dot_grid_basis_vectors[0][0],dot_grid_basis_vectors[1][0]],
                                           [dot_grid_basis_vectors[0][1],dot_grid_basis_vectors[1][1]]])
        pixel_to_dot_transform = np.linalg.inv(dot_to_pixel_transform)
        minxdb, maxxdb, minydb, maxydb = math.inf, -1.0*math.inf, math.inf, -1.0*math.inf
        for sample_point in [(0, 0), (image.height, 0,), (0, image.width), (image.height, image.width)]:
            ydb, xdb = np.dot(pixel_to_dot_transform, sample_point)
            ydb, xdb = round(ydb), round(xdb)
            minxdb, maxxdb, minydb, maxydb = min(xdb,minxdb), max(xdb,maxxdb), min(ydb,minydb), max(ydb,maxydb)
        ydb_ofs = -1 * minydb
        xdb_ofs = -1 * minxdb
        Hdot, Wdot = maxydb + ydb_ofs + 1, maxxdb + xdb_ofs + 1
        pixel_to_pixel_coordinates = np.indices((image.height, image.width)).transpose(1, 2, 0)
        pixel_to_dot_coordinates = np.rint(pixel_to_pixel_coordinates @ pixel_to_dot_transform.T).astype(int) + (ydb_ofs, xdb_ofs)
        y = pixel_to_dot_coordinates[..., 0].ravel()
        x = pixel_to_dot_coordinates[..., 1].ravel()
        v = [c_image, m_image, y_image, k_image][cmyk_idx].ravel()
        idx = y * Wdot + x
        N = Hdot * Wdot
        sum_flat = np.bincount(idx, weights=v, minlength=N)
        count_flat = np.bincount(idx, minlength=N)
        dot_radii_sq = (np.divide(sum_flat, count_flat, out=np.zeros_like(sum_flat, dtype=float), where=(count_flat != 0)).reshape(Hdot, Wdot) * cmyk_pitches[cmyk_idx]) ** 2
        pixel_to_dot_radius_sq = dot_radii_sq[pixel_to_dot_coordinates[...,0], pixel_to_dot_coordinates[...,1]]
        pixel_to_dot_center_distance_sq = np.sum((pixel_to_pixel_coordinates - (np.rint(pixel_to_pixel_coordinates @ pixel_to_dot_transform) @ dot_to_pixel_transform))**2, axis=2)
        subtractive_image = np.zeros_like(new_rgb_array)
        subtractive_image[pixel_to_dot_radius_sq >= pixel_to_dot_center_distance_sq] = anti_color
        new_rgb_array -= subtractive_image
    return Image.fromarray((new_rgb_array * 255.0).astype(np.uint8), mode="RGB")

def main():
    out_path = pathlib.Path('outputs')
    if not out_path.exists():
        out_path.mkdir()
    for path in pathlib.Path('sample-inputs').iterdir():
        print(f'processing {path}')
        image = Image.open(path)
        t1 = time.time()
        new_image_2 = render(image)
        t2 = time.time()
        print(f'Render took {round((t2-t1)*1000)}ms')
        new_image_2.save(out_path / f'{path.stem}-halftoned-new{path.suffix}')

if __name__ == '__main__':
    main()
