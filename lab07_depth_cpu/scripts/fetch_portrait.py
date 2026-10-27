"""下载选做使用的公开NASA样例；固定版本和摘要，不需要个人照片或API密钥。"""
from pathlib import Path
import base64
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SHA = '88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5'

if __name__ == '__main__':
    target = ROOT / 'assets/astronaut_source.png'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        request = urllib.request.Request(
            'https://api.github.com/repos/scikit-image/scikit-image/contents/skimage/data/astronaut.png?ref=v0.25.2',
            headers={'User-Agent': 'course-lab07'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = base64.b64decode(json.load(response)['content'])
        assert hashlib.sha256(data).hexdigest() == SHA
        target.write_bytes(data)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == SHA
    print('PUBLIC_PORTRAIT_VERIFIED')
