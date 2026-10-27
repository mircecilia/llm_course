"""将固定SD2.1深度ControlNet的FP16权重封装为GGUF，不训练/改权重。"""
from pathlib import Path
import collections
import hashlib
import json
import struct
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def gguf_info(path):
    with path.open('rb') as f:
        assert f.read(4)==b'GGUF'
        version,tensors,entries=struct.unpack('<IQQ',f.read(20))
        def string():
            n=struct.unpack('<Q',f.read(8))[0]
            return f.read(n).decode('utf-8')
        def value(kind):
            formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
            if kind==8:return string()
            if kind==9:
                item,n=struct.unpack('<IQ',f.read(12))
                return [value(item) for _ in range(n)]
            return struct.unpack('<'+formats[kind],f.read(struct.calcsize(formats[kind])))[0]
        metadata={}
        for _ in range(entries):
            key=string();metadata[key]=value(struct.unpack('<I',f.read(4))[0])
        types=collections.Counter()
        for _ in range(tensors):
            name=string()
            dims=struct.unpack('<I',f.read(4))[0]
            shape=struct.unpack('<'+'Q'*dims,f.read(8*dims))
            kind,offset=struct.unpack('<IQ',f.read(12))
            types[kind]+=1
    return dict(version=version,tensor_count=tensors,type_counts=dict(types),metadata=metadata,
                bytes=path.stat().st_size,sha256=sha(path))


if __name__=='__main__':
    source=ROOT/'models/control_v11p_sd21_depth.safetensors'
    target=ROOT/'models/control_depth_sd21-fp16.gguf'
    assert source.exists(),'等待模型下载完成'
    assert sha(source)=='ed33ec158b6ea3bf61494ee71adc16d45886d5f6611d108345a184ebe7d08d81'
    assert not target.exists(),'转换文件已存在，不覆盖'
    with source.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
    assert all(v['dtype']=='F16' for k,v in header.items() if k!='__metadata__')
    assert all(v['shape'][-1]==1024 for k,v in header.items() if k.endswith('attn2.to_k.weight'))
    cmd=[str(ROOT/'vendor/sdcpp-ext/sd-cli.exe'),'-M','convert','-m',str(source),
         '--type','f16','-o',str(target),'-t','4']
    started=time.perf_counter()
    with (ROOT/'logs/convert_control.log').open('wb') as f:
        result=subprocess.run(cmd,cwd=ROOT/'vendor/sdcpp-ext',stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0,'转换失败，检查convert_control.log'
    info=gguf_info(target)
    assert info['tensor_count']==340
    record=dict(command=cmd,seconds=time.perf_counter()-started,
        description='Source tensors are already FP16; package conversion only, no training',
        source_sha256=sha(source),control_gguf=info,
        main_gguf=gguf_info(ROOT/'models/sd_turbo-q4_0.gguf'))
    (ROOT/'logs/model_info.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CONTROL_GGUF_READY',info['bytes'],info['sha256'],flush=True)
