"""真实CPU流水线：Blender控制图导出 -> SD-Turbo/SD2.1 ControlNet。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def pixel_sha(path):
    return hashlib.sha256(Image.open(path).convert('RGB').tobytes()).hexdigest()


def ensure_rgb(path):
    with Image.open(path) as im:
        if im.mode != 'RGB':
            im.convert('RGB').save(path)


def latest(pattern):
    import glob
    files=glob.glob(str(pattern))
    return Path(max(files,key=os.path.getmtime)) if files else None


def launch(command, log, cwd=ROOT):
    """保存完整终端输出，并独立记录子进程墙钟及峰值工作集。"""
    started = time.perf_counter()
    peak = 0
    with log.open('wb') as f:
        proc = subprocess.Popen(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES=''))
        ps = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                processes = [ps]+ps.children(recursive=True)
                peak = max(peak,sum(p.memory_info().rss for p in processes if p.is_running()))
            except psutil.Error:
                pass
            time.sleep(0.5)
    elapsed = time.perf_counter()-started
    assert proc.returncode == 0, f'命令失败，原始日志：{log}'
    return dict(command=[str(c) for c in command],wall_seconds=elapsed,
                peak_working_set_mib=peak/1024**2,returncode=proc.returncode)


def read_times(log):
    text = log.read_bytes().decode('utf-8',errors='replace')
    text = re.sub(r'\x1b\[[0-9;]*m','',text)
    rules = {
        'sampling_seconds':[
            r'sampling completed[^\r\n]*?(?:taking|in)\s+([\d.]+)\s*s',
            r'sampling time\s*[:=]\s*([\d.]+)\s*s'],
        'vae_decode_seconds':[
            r'(?:vae\s+decode|decoding|decode_first_stage)\s+completed[^\r\n]*?(?:taking|in)\s+([\d.]+)\s*s',
            r'(?:vae\s+decode|decoding)\s+time\s*[:=]\s*([\d.]+)\s*s']}
    result = {}
    for key,patterns in rules.items():
        for pattern in patterns:
            matches = re.findall(pattern,text,flags=re.I)
            if matches:
                result[key] = float(matches[-1])
                break
        else:
            result[key] = None
    result['timing_lines'] = [line for line in text.splitlines()
        if re.search(r'(completed|time).*(taking|\d+\.?\d*\s*s)',line,re.I)]
    return result


def run(args):
    protocol = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    output = ROOT/'output'/args.name
    assert not output.exists(), '运行目录已存在，不覆盖'
    output.mkdir()
    started = time.perf_counter()
    record = dict(name=args.name,resolution=args.res,steps=args.steps,cfg=args.cfg,
        control_strength=args.control_strength,seed=args.seed,threads=args.threads,
        sampler='euler_a',scheduler='discrete',device='CPU',protocol=protocol,
        prompt=args.prompt or protocol['prompt'],negative_prompt=args.negative or protocol['negative_prompt'],
        vae_tiling=True,vae_tile_size='256x256',vae_tile_overlap=0.5,
        available_ram_gib_at_start=psutil.virtual_memory().available/1024**3)
    if args.control_image:
        depth = Path(args.control_image).resolve()
        assert Path(args.control_image).is_absolute()
        assert Image.open(depth).size==(args.res,args.res)
        record['blender'] = None
    else:
        blender = Path(os.environ['BLENDER_EXE']) if 'BLENDER_EXE' in os.environ else next((ROOT/'vendor/blender').glob('*/blender.exe'))
        command = [str(blender),'-b',str(Path(args.blend).resolve() if args.blend else ROOT/'assets/demo_white_model.blend'),
            '-P',str(ROOT/'blender/export_control_maps.py'),'--','--outdir',str(output),
            '--name',args.name,'--res',str(args.res),'--samples','64','--threads',str(args.threads)]
        record['blender'] = launch(command,output/'blender.log')
        depth = latest(output/f'{args.name}_depth_*.png')
        for kind in ['depth_0001','normal_0001','clay']:
            record[kind+'_pixel_sha256'] = pixel_sha(output/f'{args.name}_{kind}.png')
    # Explicit RGB conversion avoids stb/ControlNet failing on grayscale inputs.
    control_rgb = output/f'{args.name}_depth.png'
    shutil.copyfile(depth,control_rgb)
    ensure_rgb(control_rgb)
    target = output/f'{args.name}_photoreal.png'
    command = [str(ROOT/'vendor/sdcpp-ext/sd-cli.exe'),
        '-m',str(ROOT/'models/sd_turbo-q4_0.gguf'),
        '--control-net',str(ROOT/'models/control_depth_sd21-fp16.gguf'),
        '--control-image',str(control_rgb),'--control-strength',str(args.control_strength),
        '-W',str(args.res),'-H',str(args.res),'--steps',str(args.steps),
        '--cfg-scale',str(args.cfg),'--sampling-method','euler_a','--scheduler','discrete',
        '--rng','cpu','--sampler-rng','cpu','--seed',str(args.seed),
        '--threads',str(args.threads),'--backend','cpu','--params-backend','cpu','--auto-fit','off',
        '--vae-tiling','--vae-tile-size','256x256','--vae-tile-overlap','0.5',
        '-p',record['prompt'],'-n',record['negative_prompt'],'-o',str(target)]
    dump(output/'run_started.json',record)
    print('INFERENCE_STARTED',args.name,flush=True)
    record['sd_cli'] = launch(command,output/'sd-cli.log',cwd=ROOT/'vendor/sdcpp-ext')
    record.update(read_times(output/'sd-cli.log'))
    assert target.exists(), '后端未产出预期图片，请检查原始日志'
    record['photoreal_pixel_sha256'] = pixel_sha(target)
    record['end_to_end_seconds'] = time.perf_counter()-started
    record['status'] = 'complete'
    dump(output/'run.json',record)
    print('RUN_COMPLETE',json.dumps({k:record[k] for k in [
        'name','sampling_seconds','vae_decode_seconds','end_to_end_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--name',default='test512')
    p.add_argument('--res',type=int,choices=[512,768],default=512)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--cfg',type=float,default=1.2)
    p.add_argument('--control-strength',type=float,default=0.9)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=16)
    p.add_argument('--control-image')
    p.add_argument('--blend',help='可选：替换白模场景；默认采用附录演示场景')
    p.add_argument('--prompt')
    p.add_argument('--negative')
    run(p.parse_args())
