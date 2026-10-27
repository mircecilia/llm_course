"""按新版指南A.5：深度定位、预乘alpha、光度/阴影/Poisson/颗粒融合。"""
from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np
from PIL import Image


def head_alpha(rgb,face,ellipse_soft=3.0):
    key=np.clip((255.0-rgb.min(axis=2)-6.0)/18.0,0.0,1.0)
    left,top,right,chin=face
    cx,cy=(left+right)/2,(top+chin)/2+(chin-top)*0.03
    rx,ry=(right-left)/2*1.05,(chin-top)/2*1.06
    emask=np.zeros(rgb.shape[:2],np.uint8)
    cv2.ellipse(emask,(int(cx),int(cy)),(int(rx),int(ry)),0,0,360,255,-1)
    emask=cv2.GaussianBlur(emask,(0,0),ellipse_soft).astype(np.float32)/255
    alpha=np.clip(key*emask,0,1)
    hard=(alpha>0.55).astype(np.uint8)
    hard=cv2.erode(hard,np.ones((3,3),np.uint8),iterations=1)
    return cv2.GaussianBlur(hard.astype(np.float32),(0,0),1.2)


def resize_premul(rgb,alpha,size):
    a=cv2.resize(alpha,size,interpolation=cv2.INTER_AREA)
    p=cv2.resize(rgb.astype(np.float32)*alpha[...,None],size,interpolation=cv2.INTER_AREA)
    safe=np.maximum(a[...,None],1e-3)
    un=np.where(a[...,None]>1e-3,p/safe,0)
    return np.clip(un,0,255).astype(np.uint8),a,p


def locate_head(depth):
    """在深度图上部中央的前景中，依头部到颈部的宽度收缩定位。"""
    h,w=depth.shape
    mask=(depth>8).astype(np.uint8)
    mask[:, :int(w*0.2)]=0
    mask[:, int(w*0.8):]=0
    mask[int(h*0.55):]=0
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>20]
    component=min(candidates,key=lambda i:stats[i,cv2.CC_STAT_TOP]+abs(centers[i,0]-w/2)*0.3)
    selected=labels==component
    y0=int(stats[component,cv2.CC_STAT_TOP])
    widths=selected.sum(axis=1)
    peak_end=min(h,y0+int(h*0.13))
    peak_row=y0+int(np.argmax(widths[y0:peak_end]))
    peak_width=widths[peak_row]
    neck=next((y for y in range(peak_row+2,min(h,y0+int(h*0.23)))
               if widths[y]<peak_width*0.65),min(h,y0+int(peak_width*1.25)))
    ys,xs=np.nonzero(selected[y0:neck])
    return [int(xs.min()),y0,int(xs.max()+1),neck],dict(threshold=8,top_fraction=0.55,
        head_peak_row=peak_row,head_peak_width=int(peak_width),neck_row=neck)


def parse_four(s):
    return tuple(map(float,s.split(',')))


def compose(args):
    started=time.perf_counter()
    out=Path(args.out).resolve()
    assert not out.exists(),'合成结果已存在，不覆盖'
    base=np.asarray(Image.open(args.base).convert('RGB')).copy()
    head=np.asarray(Image.open(args.head).convert('RGB'))
    depth=np.asarray(Image.open(args.depth).convert('L'))
    box,diagnostic=locate_head(depth)
    raw_box=box.copy()
    h,w=base.shape[:2]
    grow=parse_four(args.box_grow)
    bw,bh=box[2]-box[0],box[3]-box[1]
    box=[max(0,int(box[0]-bw*grow[0])),max(0,int(box[1]-bh*grow[1])),
         min(w,int(box[2]+bw*grow[2])),min(h,int(box[3]+bh*grow[3]))]
    face_ratio=parse_four(args.face)
    face=tuple(v*(head.shape[1] if i%2==0 else head.shape[0]) for i,v in enumerate(face_ratio))
    alpha=head_alpha(head,face)
    ys,xs=np.nonzero(alpha>0.01)
    crop=head[ys.min():ys.max()+1,xs.min():xs.max()+1]
    alpha=alpha[ys.min():ys.max()+1,xs.min():xs.max()+1]
    scale=min((box[2]-box[0])/crop.shape[1],(box[3]-box[1])/crop.shape[0])
    size=(max(1,int(round(crop.shape[1]*scale))),max(1,int(round(crop.shape[0]*scale))))
    if args.warp_fill:
        size=(box[2]-box[0],box[3]-box[1])
    face_rgb,a,p=resize_premul(crop,alpha,size)
    x=int((box[0]+box[2]-size[0])/2)
    y=box[1]+(box[3]-box[1]-size[1])//2
    patch=base[y:y+size[1],x:x+size[0]].astype(np.float32)
    use=a>0.5
    # 温和亮度对齐，限制整体缩放幅度，避免破坏脸部细节。
    ratio=np.clip(patch.mean()/max(1,face_rgb[use].mean()),0.75,1.25)
    face_rgb=np.clip(face_rgb.astype(np.float32)*ratio,0,255).astype(np.uint8)
    p=face_rgb.astype(np.float32)*a[...,None]
    composite=base.astype(np.float32)
    shadow=np.zeros((h,w),np.float32)
    cv2.ellipse(shadow,(x+size[0]//2,y+size[1]-1),(max(1,size[0]//3),max(1,size[1]//12)),0,0,360,args.shadow,-1)
    shadow=cv2.GaussianBlur(shadow,(0,0),2.0)
    composite*=1-shadow[...,None]
    patch=composite[y:y+size[1],x:x+size[0]]
    composite[y:y+size[1],x:x+size[0]]=p+patch*(1-a[...,None])
    premul=np.clip(composite,0,255).astype(np.uint8)
    mask=(a*255).astype(np.uint8)
    center=(x+size[0]//2,y+size[1]//2)
    clone=cv2.seamlessClone(cv2.cvtColor(face_rgb,cv2.COLOR_RGB2BGR),
        cv2.cvtColor(premul,cv2.COLOR_RGB2BGR),mask,center,cv2.NORMAL_CLONE)
    clone=cv2.cvtColor(clone,cv2.COLOR_BGR2RGB)
    blend=premul.astype(np.float32)*(1-args.clone)+clone.astype(np.float32)*args.clone
    # 只在头部区域加入匹配背景高频标准差的固定种子颗粒。
    residual=base.astype(np.float32)-cv2.GaussianBlur(base.astype(np.float32),(0,0),1.0)
    sigma=min(3.0,float(residual.std())*0.2)*args.grain
    rng=np.random.default_rng(42)
    area=blend[y:y+size[1],x:x+size[0]]
    area+=rng.normal(0,sigma,area.shape)*a[...,None]
    out.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(premul).save(out.with_name(out.stem+'_premul.png'))
    Image.fromarray(np.rint(np.clip(blend,0,255)).astype(np.uint8)).save(out)
    Image.fromarray(mask).save(out.with_name(out.stem+'_alpha.png'))
    record=dict(base=str(Path(args.base).resolve()),head=str(Path(args.head).resolve()),depth=str(Path(args.depth).resolve()),
        raw_head_box=raw_box,grown_head_box=box,placed_box=[x,y,x+size[0],y+size[1]],
        locator=diagnostic,face_ratios=face_ratio,box_grow=grow,clone=args.clone,shadow=args.shadow,
        grain=args.grain,grain_sigma=sigma,brightness_ratio=float(ratio),warp_fill=args.warp_fill,
        seconds=time.perf_counter()-started,code_basis='Guide appendix A.5 head_alpha / resize_premul / seamlessClone')
    out.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPOSE_COMPLETE',json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',required=True)
    p.add_argument('--head',required=True)
    p.add_argument('--depth',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--face',default='0.16,0.02,0.85,0.71')
    p.add_argument('--box-grow',default='0.13,0.14,0.13,0.06')
    p.add_argument('--clone',type=float,default=0.6)
    p.add_argument('--shadow',type=float,default=0.35)
    p.add_argument('--grain',type=float,default=1.0)
    p.add_argument('--warp-fill',action='store_true')
    compose(p.parse_args())
