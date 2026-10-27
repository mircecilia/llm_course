"""将实际输出排成对照图；仅增加标签，不改动实验图像内容。"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def panel_grid(items, columns, target):
    size, label, gap = 512, 50, 12
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * (size + gap) + gap,
                               rows * (size + label + gap) + gap), 'white')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 25)
    draw = ImageDraw.Draw(canvas)
    for index, (caption, source) in enumerate(items):
        picture = Image.open(source).convert('RGB')
        assert picture.size == (512, 512)
        x = gap + index % columns * (size + gap)
        y = gap + index // columns * (size + label + gap)
        draw.text((x + 5, y + 9), caption, font=font, fill='black')
        canvas.paste(picture, (x, y + label))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


if __name__ == '__main__':
    panel_grid([(name, ROOT / 'output/A' / ('A_normal_0001.png' if name == 'Normal' else f'A_{name.lower()}.png'))
                for name in ['Clay', 'Depth', 'Normal']], 3,
               ROOT / 'output/figures/control_maps.png')
    panel_grid([
        ('A: 4 steps, control 0.9', ROOT / 'output/A/A_photoreal.png'),
        ('B: 8 steps, control 0.9', ROOT / 'output/B/B_photoreal.png'),
        ('C: 4 steps, control 0.5', ROOT / 'output/C/C_photoreal.png'),
        ('D: human base', ROOT / 'output/D/D_photoreal.png'),
        ('E: geometric composition', ROOT / 'output/E/headfig_face.png'),
        ('E: img2img harmonization', ROOT / 'output/E/headfig_face_harmonized.png'),
    ], 3, ROOT / 'output/figures/results.png')
    print('FIGURES_READY')
