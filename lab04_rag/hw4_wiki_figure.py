"""仅重排图 3：读取原条目文字，不加载模型、不重跑实验。"""
from pathlib import Path
import ast
import os

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
CACHE = ROOT / 'tmp' / 'wiki_plot_cache'
CACHE.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(CACHE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


def main():
    # 从源码读取常量，避免 import hw4_rag 时触发实验依赖。
    tree = ast.parse((CODE_DIR / 'hw4_rag.py').read_text(encoding='utf-8'))
    entries = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'ENTRIES' for t in n.targets))
    font_file = 'C:/Windows/Fonts/msyh.ttc'
    body_font = FontProperties(fname=font_file, size=12.3)
    title_font = FontProperties(fname=font_file, size=12.8, weight='bold')
    source_font = FontProperties(fname=font_file, size=9.5)
    width, dpi = 7.0, 300
    fig = plt.figure(figsize=(width, 1), dpi=dpi)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def wrap_exact(value, font, inches):
        # 按实际字宽换行，保留原文中的每一个字符（包括空格）。
        lines, line = [], ''
        for char in value:
            trial = line + char
            if line and renderer.get_text_width_height_descent(trial, font, False)[0] > inches*dpi:
                # 句末标点不单独出现在下一行行首，仍然逐字保留原文。
                if char in '。；，、！？：）】”’' and len(line) > 1:
                    lines.append(line[:-1])
                    line = line[-1] + char
                    continue
                lines.append(line)
                line = char
            else:
                line = trial
        lines.append(line)
        assert ''.join(lines) == value
        return lines

    rows = []
    for title, body, docs in entries:
        title_lines = wrap_exact(title, title_font, 1.50)
        if len(title_lines) > 1 and len(title_lines[-1]) == 1:
            title_lines[-1] = title_lines[-2][-2:] + title_lines[-1]
            title_lines[-2] = title_lines[-2][:-2]
        assert ''.join(title_lines) == title
        body_lines = wrap_exact(body, body_font, 4.85)
        source_lines = wrap_exact(' / '.join(docs), source_font, 1.50)
        left_height = len(title_lines)*16 + 6 + len(source_lines)*12
        height_pt = max(left_height, len(body_lines)*16) + 16
        rows.append((title_lines, body_lines, source_lines, height_pt))

    height_pt = sum(row[3] for row in rows) + 10
    fig.set_size_inches(width, height_pt/72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width*72)
    ax.set_ylim(0, height_pt)
    ax.axis('off')
    top = height_pt-5
    for index, (title, body, source, row_height) in enumerate(rows):
        bottom = top-row_height
        ax.add_patch(Rectangle((5, bottom), width*72-10, row_height,
                               facecolor='#edf7fb' if index % 2 == 0 else '#ffffff',
                               edgecolor='#9baeb8', linewidth=0.55))
        ax.plot([128,128], [bottom,top], color='#b7c5cc', linewidth=0.55)
        for j, line in enumerate(title):
            ax.text(13, top-8-j*16, line, va='top', fontproperties=title_font, color='#182c39')
        source_y = top-8-len(title)*16-6
        for j, line in enumerate(source):
            ax.text(13, source_y-j*12, line, va='top', fontproperties=source_font, color='#345b70')
        for j, line in enumerate(body):
            ax.text(137, top-8-j*16, line, va='top', fontproperties=body_font, color='#17212a')
        top = bottom
    # 每个实际文本框都应落在画布内，避免裁掉文字。
    fig.canvas.draw()
    box = fig.bbox
    for item in ax.texts:
        extent = item.get_window_extent(fig.canvas.get_renderer())
        assert box.contains(extent.x0, extent.y0) and box.contains(extent.x1, extent.y1)
    output = ROOT / 'lab04_outputs' / 'wiki_entries_compact.png'
    fig.savefig(output, dpi=dpi, facecolor='white')
    plt.close(fig)
    print('Saved:', output)
    print('Preserved:', len(entries), 'titles, full bodies and original source identifiers.')


if __name__ == '__main__':
    main()
