"""按锁定协议串行完成A/B/C与头像融合拓展；每个结果目录只创建一次。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

from run_pipeline import ROOT,dump,launch,read_times,pixel_sha


def call(script,*args):
    subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)


def summarize():
    rows=[]
    for name in ['A','B','C','D','E']:
        r=json.loads((ROOT/'output'/name/'run.json').read_text(encoding='utf-8'))
        rows.append(dict(run=name,resolution=r['resolution'],steps=r['steps'],cfg=r['cfg'],
            control_strength=r['control_strength'],seed=r['seed'],threads=r['threads'],
            sampler=r['sampler'],vae_tiling=r['vae_tiling'],sampling_seconds=r['sampling_seconds'],
            vae_decode_seconds=r['vae_decode_seconds'],
            blender_wall_seconds=r['blender']['wall_seconds'] if r.get('blender') else 0,
            inference_wall_seconds=r['sd_cli']['wall_seconds'],
            end_to_end_seconds=r['end_to_end_seconds'],
            peak_working_set_mib=r['sd_cli']['peak_working_set_mib']))
    with (ROOT/'output/results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    main=[json.loads((ROOT/'output'/n/'run.json').read_text(encoding='utf-8')) for n in ['A','B','C']]
    assert len({r['depth_0001_pixel_sha256'] for r in main})==1,'对照控制图不一致'
    assert len({r['clay_pixel_sha256'] for r in main})==1,'对白模场景不一致'
    dump(ROOT/'output/comparison_check.json',dict(status='passed',same_control_image=True,
        same_clay=True,same_seed=True,changes={'A_vs_B':'steps only','A_vs_C':'control_strength only'},
        guidance='All runs use the same CPU/VAE-tiling profile. D/E are optional, not single-variable comparisons.'))
    print('ALL_EXPERIMENTS_COMPLETE',flush=True)


if __name__=='__main__':
    (ROOT/'output').mkdir(exist_ok=True)
    for name in ['A','B','C','D','E']:
        assert not (ROOT/'output'/name).exists(),f'{name}结果已存在，不覆盖'
    guide=ROOT.parent/'作业七_白膜转写实纯CPU深度控制流水线_实验指南（含代码附录）.pdf'
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(guide.read_bytes()).hexdigest()==protocol['guide_sha256']
    print('WAITING_FOR_VERIFIED_MODEL_DOWNLOADS',flush=True)
    while not (ROOT/'logs/resource_manifest.json').exists():
        time.sleep(5)
    call('scripts/prepare_control_model.py')
    for r in protocol['runs']:
        call('scripts/run_pipeline.py','--name',r['name'],'--res',r['resolution'],
            '--steps',r['steps'],'--cfg',r['cfg'],'--control-strength',r['control_strength'],
            '--seed',protocol['seed'],'--threads',protocol['threads'])
    body_prompt=('photorealistic full-length front view portrait of a standing woman, arms hanging down, '
        'wearing a simple fitted white top and gray trousers, studio soft lighting, realistic skin, '
        'centered composition, plain gray background, preserve original 3d pose, sharp')
    call('scripts/run_pipeline.py','--name','D','--blend',ROOT/'assets/headfig_white_model.blend',
        '--res',512,'--steps',8,'--cfg',1.2,'--control-strength',0.7,
        '--seed',42,'--threads',16,'--prompt',body_prompt)
    output=ROOT/'output/E';output.mkdir()
    started=time.perf_counter()
    face=output/'headfig_face.png'
    composition=launch([sys.executable,str(ROOT/'scripts/compose_head.py'),
        '--base',str(ROOT/'output/D/D_photoreal.png'),'--depth',str(ROOT/'output/D/D_depth.png'),
        '--head',str(ROOT/'assets/portrait_white.png'),'--face','0.08,0.02,0.95,0.82',
        '--out',str(face)],output/'compose.log')
    target=output/'headfig_face_harmonized.png'
    command=[str(ROOT/'vendor/sdcpp-ext/sd-cli.exe'),'-m',str(ROOT/'models/sd_turbo-q4_0.gguf'),
        '-i',str(face),'--strength','0.35','--control-net',str(ROOT/'models/control_depth_sd21-fp16.gguf'),
        '--control-image',str(ROOT/'output/D/D_depth.png'),'--control-strength','0.7',
        '-W','512','-H','512','--steps','6','--cfg-scale','1.3',
        '--sampling-method','euler_a','--scheduler','discrete','--rng','cpu','--sampler-rng','cpu',
        '--seed','42','--threads','16','--backend','cpu','--params-backend','cpu','--auto-fit','off',
        '--vae-tiling','--vae-tile-size','256x256','--vae-tile-overlap','0.5',
        '-p',body_prompt,'-n',protocol['negative_prompt'],'-o',str(target)]
    record=dict(name='E',resolution=512,steps=6,cfg=1.3,control_strength=0.7,strength=0.35,
        seed=42,threads=16,sampler='euler_a',scheduler='discrete',device='CPU',
        vae_tiling=True,vae_tile_size='256x256',vae_tile_overlap=0.5,
        blender=None,composition=composition,prompt=body_prompt,negative_prompt=protocol['negative_prompt'])
    record['sd_cli']=launch(command,output/'sd-cli.log',cwd=ROOT/'vendor/sdcpp-ext')
    record.update(read_times(output/'sd-cli.log'))
    assert target.exists()
    record['photoreal_pixel_sha256']=pixel_sha(target)
    record['end_to_end_seconds']=time.perf_counter()-started
    record['status']='complete'
    dump(output/'run.json',record)
    summarize()
