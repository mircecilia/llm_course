"""从镜像下载固定版本资源；按上游 SHA-256 验证后才解压便携软件。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
CHUNK = 16 * 1024 * 1024
RESOURCES = [
    dict(name='backend', path='cache/downloads/sdcpp-cpu.zip', size=18080575,
         sha256='f0f76c8176a90cafc431ca7d5049cbe8c5cea61ca7cb92b78e07361d41d8fb21',
         url='https://ghfast.top/https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1017-be48649/sd-master-be48649-bin-win-cpu-x64.zip',
         extract='vendor/sdcpp-ext'),
    dict(name='blender', path='cache/downloads/blender-4.5.13-windows-x64.zip', size=398648740,
         sha256='b5fdf800ce65fa2f209e8f68d02667e4d720fa1c42f247c72d1882ab04decba6',
         url='https://mirrors.tuna.tsinghua.edu.cn/blender/release/Blender4.5/blender-4.5.13-windows-x64.zip',
         extract='vendor/blender'),
    dict(name='sd_turbo', path='models/sd_turbo-q4_0.gguf', size=1847909248,
         sha256='d069ef2b2d3d5c2465662d706c242a1016db56163c109bd7e3008a94cdd00359',
         url='https://hf-mirror.com/samoudi/sd-turbo-gguf/resolve/0b5ee51e9a43b33b2d32865781719a85a2248c12/sd_turbo_q4_0.gguf?download=true'),
    dict(name='control_depth_sd21', path='models/control_v11p_sd21_depth.safetensors', size=728494536,
         sha256='ed33ec158b6ea3bf61494ee71adc16d45886d5f6611d108345a184ebe7d08d81',
         url='https://hf-mirror.com/thibaud/controlnet-sd21/resolve/ba159a8ce47185aa1821db7fe10137cf2f5e04dd/control_v11p_sd21_depth.safetensors?download=true')
]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def download(item):
    target = ROOT/item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == item['size'] and sha256(target) == item['sha256']:
        print(item['name'], 'already verified', flush=True)
    else:
        parts = ROOT/'cache/downloads'/item['name']
        parts.mkdir(parents=True, exist_ok=True)
        total = (item['size']+CHUNK-1)//CHUNK

        def piece(i):
            start, end = i*CHUNK, min((i+1)*CHUNK, item['size'])-1
            p = parts/f'{i:04d}.part'
            if p.exists() and p.stat().st_size == end-start+1:
                return i
            for attempt in range(4):
                try:
                    # 系统curl使用Schannel，避免本机Python/OpenSSL下载末尾分片时的TLS EOF。
                    curl=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/curl.exe'
                    result=subprocess.run([str(curl),'--location','--fail','--silent','--show-error',
                        '--connect-timeout','20','--max-time','180','--retry','3','--retry-all-errors',
                        '--retry-delay','1','--range',f'{start}-{end}',
                        '--output',str(p.with_suffix('.tmp')),'--write-out','%{http_code}',
                        '--url',item['url']],capture_output=True,text=True,timeout=750)
                    assert result.returncode==0 and result.stdout.strip()=='206',result.stderr[-250:]
                    assert p.with_suffix('.tmp').stat().st_size == end-start+1
                    os.replace(p.with_suffix('.tmp'), p)
                    return i
                except Exception as exc:
                    print(item['name'],'part',i,'retry',attempt+1,type(exc).__name__,flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(2)

        print(item['name'], 'download starts', item['size'], 'bytes', flush=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(piece, i) for i in range(total)]
            for done, f in enumerate(as_completed(futures), 1):
                f.result()
                if done == total or done % max(1,total//10) == 0:
                    print(item['name'], f'{done}/{total} parts', flush=True)
        with target.with_suffix(target.suffix+'.partial').open('wb') as out:
            for i in range(total):
                with (parts/f'{i:04d}.part').open('rb') as f:
                    while b := f.read(8*1024*1024):
                        out.write(b)
        pending = target.with_suffix(target.suffix+'.partial')
        assert sha256(pending) == item['sha256'], f"{item['name']} SHA-256 mismatch"
        os.replace(pending, target)
    if 'extract' in item:
        destination = ROOT/item['extract']
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target) as z:
            for name in z.namelist():
                assert (destination/name).resolve().is_relative_to(destination.resolve())
            z.extractall(destination)
    print(item['name'], 'SHA256_VERIFIED', flush=True)
    return item


if __name__ == '__main__':
    (ROOT/'logs').mkdir(exist_ok=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        verified = list(ex.map(download, RESOURCES))
    record = dict(seconds=time.perf_counter()-started, files=verified)
    (ROOT/'logs/resource_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ALL_RESOURCES_VERIFIED', flush=True)
