#!/usr/bin/env python3

import argparse
import math
import pathlib
import subprocess
import sys
import time
try:
    import numpy as np
    from PIL import Image
    from numba import jit
    from scipy.ndimage import gaussian_filter, convolve
except ImportError as e:
    if sys.prefix.endswith('venv'):
        print("I already tried to set up a venv and install the required packages, but it still failed... Boo!")
        raise e
    print(f'With system python: {e.msg}; using a local virtual environment')
    venv_path = pathlib.Path(__file__).parent / 'inkdots_venv'
    if not venv_path.exists():
        print('No venv found; setting up a new one...')
        subprocess.run(['python3', '-m', 'venv', 'inkdots_venv'])
        subprocess.run(['inkdots_venv/bin/pip3', 'install', '--upgrade', 'pip'])
        subprocess.run(['inkdots_venv/bin/pip3', 'install', 'numpy', 'Pillow', 'numba'])
        subprocess.run(['inkdots_venv/bin/pip3', 'install', 'scipy'])  # scipy is a big package; install separately
    print('Re-running self in the venv...')
    subprocess.run(['inkdots_venv/bin/python', __file__] + sys.argv[1:])
    sys.exit(0)

def render(image):
    angles = [math.radians(x) for x in [15, 75, 1, 45]]
    pitches = [3] * 4
    black_ratio = 0.05
    dot_size_factor = 0.7
    levels = 4 # Color quantization levels - fewer is starker colors
    edge_threshold = 0.3 # Edge detection threshold
    edge_sigma = 2.0 # Ink-blottyness of the pen lines sorta
    rgb = np.ascontiguousarray(image).astype(np.float32) / 255.0
    rgb_levelized = np.floor(rgb * levels) / (levels - 1)
    k_image = (1 - np.max(rgb_levelized, axis=2)) * black_ratio
    c_image, m_image, y_image = (1 - rgb_levelized[...,0] - k_image), (1 - rgb_levelized[...,1] - k_image), (1 - rgb_levelized[...,2] - k_image)
    cmyk_images = [c_image, m_image, y_image, k_image]
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
    subtractive_image = np.zeros_like(new_rgb_array)
    for cmyk_idx in range(4):
        _pixel_to_dot_coords_float = pixel_to_pixel_coordinates @ pixel_to_dot_transforms[cmyk_idx]
        pixel_to_dot_coords_rounded = np.rint(_pixel_to_dot_coords_float)
        _pixel_to_dot_coords_float -= pixel_to_dot_coords_rounded
        pixel_to_dot_center_distance_sq = np.einsum('ijk,ijk->ij', _pixel_to_dot_coords_float, _pixel_to_dot_coords_float)
        dot_centers_in_pixel_space = (pixel_to_dot_coords_rounded @ dot_to_pixel_transforms[cmyk_idx].T).astype(np.int32)
        center_y = np.clip(dot_centers_in_pixel_space[..., 0], 0, image.height - 1)
        center_x = np.clip(dot_centers_in_pixel_space[..., 1], 0, image.width - 1)
        pixel_to_dot_radius_sq = cmyk_images[cmyk_idx][center_y, center_x] ** 2 * (dot_size_factor ** 2)
        subtractive_image.fill(0)  # Reset to zeros
        subtractive_image[pixel_to_dot_radius_sq >= pixel_to_dot_center_distance_sq] = anti_colors[cmyk_idx]
        new_rgb_array -= subtractive_image
    gray_image = np.dot(rgb, [0.2989, 0.5870, 0.1140])
    blurred = gaussian_filter(gray_image, sigma=edge_sigma)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_x = convolve(blurred, sobel_x)
    grad_y = convolve(blurred, sobel_y)
    gradient_magnitude = np.hypot(grad_x, grad_y)
    edges = gradient_magnitude > edge_threshold
    edge_color = np.array([0.0, 0.0, 0.0])
    new_rgb_array[edges] = edge_color
    return Image.fromarray((new_rgb_array * 255.0).astype(np.uint8), mode="RGB")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path, help='Path to image or dir')
    args = parser.parse_args()

    in_images = []
    if args.input.is_dir():
        in_images = list(args.input.iterdir())
    else:
        in_images = [args.input]

    times = []
    for path in in_images:
        print(f'printing {path}')
        image = Image.open(path)
        t1 = time.time()
        render(image).save( path.parent / f'{path.stem}-halftoned.{path.suffix}')
        t2 = time.time()
        print(f'Render took {round((t2-t1)*1000)}ms')
        times.append((t2-t1)*1000)
        if len(times) > 1:
            print(f'average: {round( sum(times[1:]) / (len(times)-1) ) }ms')

if __name__ == '__main__':
    main()
