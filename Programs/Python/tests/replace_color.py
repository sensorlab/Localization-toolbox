from PIL import Image


def replace_png_color(old_color, new_color, old_file, new_file):
    im = Image.open(old_file)
    pixels = im.load()

    width, height = im.size
    for x in range(width):
        for y in range(height):
            pixel_color = pixels[x, y]
            r = pixel_color[0]
            g = pixel_color[1]
            b = pixel_color[2]
            a = 0
            if len(pixel_color) > 3:
                a = pixel_color[3]
            if (r, g, b) == (old_color[0], old_color[1], old_color[2]):
                pixels[x, y] = (new_color[0], new_color[1], new_color[2], a)
    im.save(new_file)

"""
OLD_PATH = r'c:\path\to\images\arrow_white.png'
NEW_PATH = r'c:\path\to\images\arrow_cyan.png'

R_OLD, G_OLD, B_OLD = (255, 255, 255)
R_NEW, G_NEW, B_NEW = (0, 174, 239)

import Image
im = Image.open(OLD_PATH)
pixels = im.load()

width, height = im.size
for x in range(width):
    for y in range(height):
        r, g, b, a = pixels[x, y]
        if (r, g, b) == (R_OLD, G_OLD, B_OLD):
            pixels[x, y] = (R_NEW, G_NEW, B_NEW, a)
im.save(NEW_PATH)
"""

if __name__ == '__main__':
    # files = ["x_Icon.png", "s_Icon.png", "o_Icon.png", "antenna_tower_Icon.png"]
    # files = ["x_Icon.png", "o_Icon.png", "antenna_tower_Icon.png"]
    files = ["s_Icon.png"]
    new_files = []
    Color = "B"
    old_color = [255, 0, 0]
    new_color = [0, 0, 255]

    for old_file in files:
        new_file = Color + "_" + old_file
        print old_file
        replace_png_color(old_color, new_color, old_file, new_file)


