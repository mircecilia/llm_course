"""将公开NASA头像样例裁成白底素材；保留原图和裁剪/遮罩记录。"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    source=ROOT/'assets/astronaut_source.png'
    target=ROOT/'assets/portrait_white.png'
    assert not target.exists(),'头像素材已准备，不覆盖'
    crop_box=[135,0,310,240]
    rgb=np.asarray(Image.open(source).convert('RGB').crop(crop_box))
    h,w=rgb.shape[:2]
    mask=np.zeros((h,w),np.uint8)
    cv2.ellipse(mask,(89,101),(78,99),0,0,360,255,-1)
    alpha=cv2.GaussianBlur(mask.astype(np.float32)/255,(0,0),0.8)
    white=rgb.astype(np.float32)*alpha[...,None]+255*(1-alpha[...,None])
    Image.fromarray(np.rint(white).astype(np.uint8)).save(target)
    (ROOT/'logs/portrait_preparation.json').write_text(json.dumps(dict(
        source='Public NASA / scikit-image astronaut sample',crop_box=crop_box,
        white_background='elliptical crop of source head placed on white',
        ellipse_center=[89,101],ellipse_axes=[78,99],
        compose_face_ratios=[0.08,0.02,0.95,0.82],
        note='Demonstration material, not a private portrait or identity benchmark'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    print('PORTRAIT_PREPARED',target)
