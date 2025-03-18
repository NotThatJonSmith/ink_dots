import math
import pathlib
import urllib.request

from PIL import Image


INK_C, INK_M, INK_Y, INK_W = 0b001, 0b010, 0b100, 0b000
INK_R, INK_G, INK_B, INK_K = 0b110, 0b101, 0b011, 0b111
ink_to_rgb = [(255, 255, 255), (000, 255, 255), (255, 000, 255), (000, 000, 255),
              (255, 255, 000), (000, 255, 000), (255, 000, 000), (000, 000, 000)]


def rotate_vector(x, y, theta):
    return float(x) * math.cos(theta) - float(y) * math.sin(theta),\
           float(x) * math.sin(theta) - float(y) * math.cos(theta)


def floatrange(start, stop, step):
    while start < stop:
        yield start
        start += step


def sample_point(img, x, y):
    x_floor, y_floor, x_ceil, y_ceil = math.floor(x), math.floor(y), math.ceil(x), math.ceil(y)
    if x_floor < 0 or x_ceil >= img.width or y_floor < 0 or y_ceil >= img.height:
        return None
    ff, fc = (x - x_floor) * (y - y_floor), (x - x_floor) * (y_ceil - y)
    cf, cc = (x_ceil - x) * (y - y_floor), (x_ceil - x) * (y_ceil - y)
    ff_color, fc_color = img.getpixel((x_floor, y_floor)), img.getpixel((x_floor, y_ceil))
    cf_color, cc_color = img.getpixel((x_ceil, y_floor)), img.getpixel((x_ceil, y_ceil))
    return (cc*ff_color[0] + cf*fc_color[0] + fc*cf_color[0] + ff*cc_color[0],
            cc*ff_color[1] + cf*fc_color[1] + fc*cf_color[1] + ff*cc_color[1],
            cc*ff_color[2] + cf*fc_color[2] + fc*cf_color[2] + ff*cc_color[2])


# Return the average RGB of the pixels in a square patch
def sample_patch(img, x, y, theta, size):
    colors = []
    for x_in_square in range(size):
        for y_in_square in range(size):
            rot_x_in_square, rot_y_in_square = rotate_vector(float(x_in_square) - (size/2.0), float(y_in_square) - (size/2.0), theta)
            x_in_image, y_in_image = x + rot_x_in_square, y + rot_y_in_square
            interpolated_color = sample_point(img, x_in_image, y_in_image)
            if interpolated_color is not None:
                colors.append(interpolated_color)
    if len(colors) == 0:
        return None
    return (sum([color[0] for color in colors]) / len(colors),
            sum([color[1] for color in colors]) / len(colors),
            sum([color[2] for color in colors]) / len(colors))


def avg_rgb_to_ratio_of_cmky_ink(rgb, ink):
    brightest = max(rgb)
    percent_k = (255.0 - brightest) / 255.0
    percent_c = (brightest - rgb[0]) / 255.0
    percent_m = (brightest - rgb[1]) / 255.0
    percent_y = (brightest - rgb[2]) / 255.0
    if ink == INK_C:
        return percent_c
    if ink == INK_M:
        return percent_m
    if ink == INK_Y:
        return percent_y
    if ink == INK_K:
        return percent_k
    raise ValueError


def dot_strip(image, start_x, start_y, theta, stride, ink):
    stripe_x, stripe_y = start_x, start_y
    walk_x, walk_y = rotate_vector(stride, 0, theta)
    while 0 <= stripe_x < image.width and 0 <= stripe_y < image.height:
        rgb = sample_patch(image, stripe_x, stripe_y, theta, stride)
        radius = 0
        if rgb is not None:
            radius = stride * avg_rgb_to_ratio_of_cmky_ink(rgb, ink)
        yield stripe_x, stripe_y, radius, ink
        stripe_x, stripe_y = stripe_x + walk_x, stripe_y + walk_y


def add_grid(image, stride, theta, ink):
    for y in floatrange(0.0, image.height, stride / math.sin((math.pi / 2.0) - theta)):
        for dot in dot_strip(image, 0.0, y, theta, stride, ink):
            yield dot
    for x in floatrange(0.0, image.width, stride / math.sin(theta)):
        for dot in dot_strip(image, x, 0.0, theta, stride, ink):
            yield dot


def print_dot(ink_image, dot):
    x_c, y_c, radius, ink = dot
    x_0, y_0 = int(x_c) - int(radius), int(y_c) - int(radius)
    for y in range(y_0, y_0+2*int(radius)):
        for x in range(x_0, x_0+2*int(radius)):
            if y < 0 or y >= len(ink_image) or x < 0 or x >= len(ink_image[0]):
                continue
            if math.sqrt((x-x_c)**2 + (y-y_c)**2) > radius:
                continue
            ink_image[y][x] |= ink


def render(image):
    print('computing dots')
    cmyk_angles = [math.radians(x) for x in [15, 75, 1, 45]]
    cmyk_pitches = [10, 10, 10, 10]
    dots = \
        list(add_grid(image, cmyk_pitches[0], cmyk_angles[0], INK_C)) + \
        list(add_grid(image, cmyk_pitches[1], cmyk_angles[1], INK_M)) + \
        list(add_grid(image, cmyk_pitches[2], cmyk_angles[2], INK_Y)) + \
        list(add_grid(image, cmyk_pitches[3], cmyk_angles[3], INK_K))

    print('printing/blending dots')
    ink_image = []
    for i in range(image.height):
        ink_image.append([0]*image.width)
    for dot in dots:
        print_dot(ink_image, dot)

    print('blit')
    new_image = Image.new('RGB', (image.width, image.height))
    for y in range(image.height):
        for x in range(image.width):
            new_image.putpixel((x, y), ink_to_rgb[ink_image[y][x]])
    return new_image


def main():
    out_path = pathlib.Path('outputs')
    if not out_path.exists():
        out_path.mkdir()
    for path in pathlib.Path('sample-inputs').iterdir():
        print(f'processing {path}')
        image = Image.open(path)
        new_image = render(image)
        new_image.save(out_path / f'{path.stem}-halftoned.{path.suffix}')
        image.show()
        new_image.show()


if __name__ == '__main__':
    main()
