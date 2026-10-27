"""按新版指南A.2/A.3导出Mist深度、法线和白模图，使用Cycles CPU。"""
from pathlib import Path
import argparse
import json
import sys
import time


def scene_depth_range(scene, camera):
    """按附录：仅取视锥内顶点，使用到相机的欧氏距离2%/98%分位。"""
    import bpy
    import numpy as np
    from bpy_extras.object_utils import world_to_camera_view
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_loc = camera.matrix_world.translation
    distances = []
    for obj in scene.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        for vert in evaluated.data.vertices:
            world_co = evaluated.matrix_world@vert.co
            ndc = world_to_camera_view(scene,camera,world_co)
            if ndc.z <= 0 or not (0 <= ndc.x <= 1 and 0 <= ndc.y <= 1):
                continue
            distances.append((world_co-cam_loc).length)
    if not distances:
        return 0.1,100.0,0
    near,far = map(float,np.percentile(distances,[2,98]))
    return near,far if far>near else near+1,len(distances)


def make_clay_material():
    import bpy
    material = bpy.data.materials.new('Export_clay_white')
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value=(0.82,0.82,0.82,1)
    bsdf.inputs['Roughness'].default_value=0.7
    return material


def setup_compositor(scene,outdir,name):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new('CompositorNodeRLayers')
    invert = tree.nodes.new('CompositorNodeInvert')
    invert.inputs['Fac'].default_value=1
    if 'Invert Color' in invert.inputs:
        invert.inputs['Invert Color'].default_value=True
    mul = tree.nodes.new('CompositorNodeMixRGB')
    mul.blend_type='MULTIPLY'
    mul.inputs[0].default_value=1.0
    mul.inputs[2].default_value=(0.5,0.5,0.5,1)
    add = tree.nodes.new('CompositorNodeMixRGB')
    add.blend_type='ADD'
    add.inputs[0].default_value=1.0
    add.inputs[2].default_value=(0.5,0.5,0.5,1)
    # 显式补齐混合因子为1，才能落实附录注明的 n*0.5+0.5。
    for node in [mul,add]:
        node.use_clamp=False
    depth_out=tree.nodes.new('CompositorNodeOutputFile')
    normal_out=tree.nodes.new('CompositorNodeOutputFile')
    for node,label in [(depth_out,'depth'),(normal_out,'normal')]:
        node.base_path=str(outdir)
        node.file_slots[0].path=f'{name}_{label}_'
        node.format.file_format='PNG'
        node.format.color_mode='RGB'
        node.format.color_depth='8'
        # 数据通道保留数值映射；clay仍使用场景的AgX显示变换。
        node.format.color_management='OVERRIDE'
        node.format.view_settings.view_transform='Raw'
        node.format.view_settings.exposure=0
        node.format.view_settings.gamma=1
        if hasattr(node.file_slots[0],'save_as_render'):
            node.file_slots[0].save_as_render=False
    depth_in = invert.inputs['Color'] if 'Color' in invert.inputs else invert.inputs['Image']
    tree.links.new(rl.outputs['Mist'],depth_in)
    tree.links.new(invert.outputs[0],depth_out.inputs[0])
    tree.links.new(rl.outputs['Normal'],mul.inputs[1])
    tree.links.new(mul.outputs[0],add.inputs[1])
    tree.links.new(add.outputs[0],normal_out.inputs[0])
    composite=tree.nodes.new('CompositorNodeComposite')
    tree.links.new(rl.outputs['Image'],composite.inputs[0])


def export(args):
    import bpy
    started=time.perf_counter()
    scene=bpy.context.scene
    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=args.samples
    scene.cycles.use_denoising=True
    scene.cycles.seed=42
    scene.render.resolution_x=scene.render.resolution_y=args.res
    scene.render.resolution_percentage=100
    scene.render.threads_mode,scene.render.threads='FIXED',args.threads
    out=Path(args.outdir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/f'{args.name}_depth_0001.png').exists(),'控制图已存在，不覆盖'
    near,far,count=scene_depth_range(scene,scene.camera)
    scene.world.mist_settings.start=near
    scene.world.mist_settings.depth=far-near
    scene.world.mist_settings.falloff='LINEAR'
    layer=scene.view_layers[0]
    layer.use_pass_mist=layer.use_pass_normal=True
    layer.material_override=make_clay_material()
    setup_compositor(scene,out,args.name)
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'
    scene.render.filepath=str(out/f'{args.name}_clay.png')
    t=time.perf_counter()
    bpy.ops.render.render(write_still=True)
    record=dict(blender=bpy.app.version_string,device='CPU',resolution=args.res,samples=args.samples,
        threads=args.threads,visible_vertices=count,mist_start=near,mist_end=far,mist_depth=far-near,
        percentiles=[2,98],range_metric='Euclidean distance to camera, guide appendix A.2',
        cycles_render_seconds=time.perf_counter()-t,export_seconds=time.perf_counter()-started,
        compositor='guide appendix A.3: inverse Mist and Normal*0.5+0.5',normal_mix_factors=[1.0,1.0],
        data_color_transform='Raw (avoid applying AgX to depth/normal data)')
    (out/f'{args.name}_export.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXPORT_COMPLETE',json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--outdir',default='output')
    p.add_argument('--name',default='demo')
    p.add_argument('--res',type=int,choices=[512,768],default=512)
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--threads',type=int,default=16)
    export(p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []))
