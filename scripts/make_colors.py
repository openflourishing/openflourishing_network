import distinctipy

def to_color_rgb(color, luminance=255):
    r, g, b = color
    return int(r * luminance), int(g * luminance), int(b * luminance)


def to_color_hex(color):
    r, g, b = to_color_rgb(color)
    return f"#{r:02X}{g:02X}{b:02X}"


def get_colors(N):
    colors = distinctipy.get_colors(
        N, pastel_factor=0.5, rng=0
    )
    communities = []
    for i, col in enumerate(colors):
        community = {}
        community["color_rgb"] = to_color_rgb(col)
        community["color"] = to_color_hex(col)
        communities.append(community)
    return communities

if __name__ == '__main__':
    colors = get_colors(21)
    for color in colors:
        print(color)