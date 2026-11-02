# 实验七 纯 CPU 深度控制生成

按“实验指南（含代码附录）”A.1～A.5补齐可运行程序。白模生成、Mist/Normal导出、SD-Turbo与SD2.1深度ControlNet均在CPU执行。主模型为Q4_0混合精度GGUF，控制模型为FP16；不需要PyTorch或GPU。

## 环境与资源

从仓库根目录执行，使用已有Python创建独立venv。下面的`python`应指向你已安装的Python3.12，不要往base安装依赖。

```powershell
python -m venv .\lab07_depth_cpu\.venv
$py = (Resolve-Path .\lab07_depth_cpu\.venv\Scripts\python.exe).Path
& $py -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r .\lab07_depth_cpu\requirements.txt
& $py .\lab07_depth_cpu\download_resources.py
```

下载固定版本便携Blender4.5.13、stable-diffusion.cpp CPU版be48649、SD-Turbo GGUF与SD2.1 depth权重。软件走TUNA/GitHub镜像，模型走hf-mirror；每个资源核对SHA-256。模型、软件及环境都保存在本目录，不进入Git。下载约3GB，分片缓存也占磁盘；速度由网络决定。

指南写有已提供GGUF，但上传资料没有附带模型，因此脚本下载固定公开权重，并由后端把原本FP16的ControlNet重新封装为GGUF；没有训练或改变权重精度。SD2.1交叉注意力为1024维，不能换成SD1.5控制模型。

## 从头复现

`protocol.json`随源码固定模型、种子、参数和指南摘要，不需要课程PDF才能运行。若仓库根目录有对应PDF，入口会核对摘要；没有PDF时直接使用同一固定协议。先完成上面的资源下载，再在**首次运行、没有A～E结果时**执行：

```powershell
$env:BLENDER_USER_CONFIG = Join-Path (Resolve-Path .\lab07_depth_cpu).Path 'cache\blender_config'
$env:CUDA_VISIBLE_DEVICES = ''
& $py .\lab07_depth_cpu\blender\make_demo_scene.py
& $py .\lab07_depth_cpu\blender\make_human_scene.py
& $py .\lab07_depth_cpu\scripts\fetch_portrait.py
& $py .\lab07_depth_cpu\scripts\prepare_portrait.py
& $py -u .\lab07_depth_cpu\scripts\run_all.py
& $py .\lab07_depth_cpu\scripts\audit_results.py
& $py .\lab07_depth_cpu\scripts\make_report_figures.py
```

公开头像为NASA的Eileen Collins照片，来自scikit-image v0.25.2的astronaut样例。来源及公共领域说明见[官方文档](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)。它用于演示合成，不是用户照片，也不是人脸身份一致性基准。

`run_all.py`先封装ControlNet，再串行完成：

|组|配置与目的|
|---|---|
|A|512²、4步、CFG1.2、控制0.9，必做基准|
|B|仅改为8步，必做步数对照|
|C|仅改为控制0.5，思考题控制强度对照|
|D|人体白模、512²、8步、CFG1.2、控制0.7，选做底图|
|E|深度定位头像、预乘alpha合成，再以strength0.35、配置6步、CFG1.3、控制0.7做img2img|

固定seed42、Euler A、discrete调度、16线程。所有组都采用256×256 VAE分块、overlap0.5及串行阶段，以适应本机16GB物理内存；不修改页面文件或系统设置。指南建议至少6GB可用内存，运行时仍应给系统留出余量。本次实际峰值工作集约2.9GiB，不代表包含文件映射和提交内存在内的全部内存需求。

缺少`logs/resource_manifest.json`时，入口提示先执行下载程序，不会一直等待。这个文件由下载器在全部资源通过SHA-256校验后生成；不要手工伪造。下载器支持校验已有文件和续传分片，但会解压便携软件并重新写入下载清单，属于准备步骤，不是只读检查。

套件拒绝覆盖已有A～E目录；再次独立运行必做组可使用新名称，例如：

```powershell
& $py .\lab07_depth_cpu\scripts\run_pipeline.py --name repeat_A --res 512 --steps 4 --cfg 1.2 --control-strength 0.9 --seed 42 --threads 16
```

不要把上述独立运行结果覆盖到报告原始组。重新完整复现应另建工作副本，不复制旧`output/`和生成的`.blend`；也不复制已封装的ControlNet目标GGUF，套件会从原始safetensors生成它。

## 输出与计时口径

- `output/A`～`E`：原始日志、run.json、图片；`output/results.csv`是实测汇总。
- `output/comparison_check.json`与只读`audit_results.py`核对A/B/C相同白模、深度像素、seed及控制变量。
- clay/depth/normal在同一次Cycles渲染联合导出，`*_export.json`记录联合时间；不虚构每张独立渲染耗时。
- `sampling_seconds`与`vae_decode_seconds`从后端日志解析；端到端包含加载、Blender等。E复用D深度，另含合成、VAE编码，不重复计算Blender。
- E虽然设置6步，strength0.35在该版本后端实际执行3次采样循环；不能直接用6除采样时间与txt2img比较。
- 数据图按Raw导出，避免AgX改变深度/法线数值；白模视觉参考仍用AgX。Normal仅检查方向，不输入ControlNet。
- `output/figures`只给实际图像排版加标签，不造图、不改实验数据。

## 头像选做的实际实现

`compose_head.py`从人体深度图上部中央前景定位目标头框；源头像的人脸框比例通过`--face`给定，不运行自动人脸检测。软抠图后内缩、羽化，使用预乘alpha等比缩放；默认不横向拉伸。随后执行亮度匹配、接触阴影、Poisson融合和固定seed颗粒匹配，再由`run_all.py`执行低强度img2img与Depth ControlNet协调。全部步骤的参数及实际命令保存在E组JSON与日志中；完成链路不代表已经解决比例、接缝或身份保持问题。

本次结果含真实失败：A/B后方立方体丢失、圆柱开口化，B背景条纹，C球体环化；选做协调重绘改变了原头像外观。固定种子可帮助复现，本机以外的后端版本、CPU指令和浮点归约仍可能引起图像差异。不要把运行成功等同于几何或身份完全保持。
