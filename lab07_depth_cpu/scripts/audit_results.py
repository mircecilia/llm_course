"""只读核对五组结果、单变量对照和CPU执行记录，不启动模型、不覆盖产物。"""
from pathlib import Path
import hashlib
import json
import re
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def pixel_hash(path):
    return hashlib.sha256(Image.open(path).convert('RGB').tobytes()).hexdigest()


if __name__ == '__main__':
    records = {n: json.loads((ROOT / 'output' / n / 'run.json').read_text(encoding='utf-8'))
               for n in 'ABCDE'}
    for name, run in records.items():
        assert run['status'] == 'complete'
        assert run['seed'] == 42 and run['device'] == 'CPU'
        command = run['sd_cli']['command']
        assert command[command.index('--backend') + 1] == 'cpu'
        assert run['vae_tiling'] and run['threads'] == 16
        log = (ROOT / 'output' / name / 'sd-cli.log').read_text(encoding='utf-8', errors='replace')
        measured = float(re.search(r'sampling completed, taking ([\d.]+)s', log)[1])
        assert measured == run['sampling_seconds']
        assert 'VRAM 0.00' in log
        print(f"{name}: sampling={measured:.2f}s, VAE={run['vae_decode_seconds']:.2f}s, total={run['end_to_end_seconds']:.2f}s")
    for kind, filename in [('depth', '{n}_depth.png'), ('clay', '{n}_clay.png')]:
        hashes = {pixel_hash(ROOT / 'output' / n / filename.format(n=n)) for n in 'ABC'}
        assert len(hashes) == 1, f'{kind}对照不一致'
    for key in ['resolution', 'cfg', 'control_strength', 'sampler', 'seed', 'prompt', 'negative_prompt']:
        assert records['A'][key] == records['B'][key]
    for key in ['resolution', 'cfg', 'steps', 'sampler', 'seed', 'prompt', 'negative_prompt']:
        assert records['A'][key] == records['C'][key]
    assert records['A']['steps'] == 4 and records['B']['steps'] == 8
    assert records['A']['control_strength'] == 0.9 and records['C']['control_strength'] == 0.5
    print('PASS: A/B仅步数变化；A/C仅控制强度变化；同一控制图、白模和CPU内存策略。')
    print('E配置6步、strength=0.35；实际采样循环3步，见原始sd-cli.log。')
