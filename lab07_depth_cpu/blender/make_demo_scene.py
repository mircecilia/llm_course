"""建立固定白模。用普通Python启动本脚本时，自动转交便携Blender。"""
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def blender_exe():
    return Path(os.environ.get('BLENDER_EXE', next((ROOT/'vendor/blender').glob('*/blender.exe'))))


def create_scene():
    import bpy
    from mathutils import Vector
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.cycles.seed = 42
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 16
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.view_settings.view_transform = 'AgX'
    white = bpy.data.materials.new('White_matte')
    white.use_nodes = True
    bsdf = white.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.82,0.82,0.82,1)
    bsdf.inputs['Roughness'].default_value = 0.7

    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.62, location=(-0.95,0,0.62))
    sphere = bpy.context.object
    sphere.name = 'Sphere'
    bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_torus_add(major_segments=64, minor_segments=24,
        major_radius=0.5, minor_radius=0.18, location=(0.95,0.15,0.68), rotation=(math.radians(75),0,0))
    torus = bpy.context.object
    torus.name = 'Torus'
    bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-0.9,1.2,0.5), rotation=(0,0,math.radians(12)))
    cube = bpy.context.object
    cube.name = 'Beveled_cube'
    bevel = cube.modifiers.new('Bevel','BEVEL')
    bevel.width, bevel.segments = 0.13, 4
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.4, depth=1.2, location=(0.65,-1.0,0.6))
    cylinder = bpy.context.object
    cylinder.name = 'Cylinder'
    bevel = cylinder.modifiers.new('Edge_bevel','BEVEL')
    bevel.width, bevel.segments = 0.06, 3
    bpy.ops.mesh.primitive_plane_add(size=24, location=(0,0,0))
    ground = bpy.context.object
    ground.name = 'Ground'
    for obj in [sphere,torus,cube,cylinder,ground]:
        obj.data.materials.append(white)
    for name,location,energy,size,target_point in [
        ('Key',(3.0,-3.2,4.2),300,3.5,(0,0,0.5)),
        ('Fill',(-3.6,-2.8,2.2),80,5.0,(0,0,0.6)),
        ('Rim',(0.5,3.5,3.2),220,2.2,(0,0,0.8))]:
        light = bpy.data.lights.new(name,'AREA')
        light.energy, light.shape, light.size = energy,'DISK',size
        obj = bpy.data.objects.new(name,light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector(target_point)-obj.location).to_track_quat('-Z','Y').to_euler()
    world = bpy.data.worlds.new('Studio_world')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.35,0.35,0.35,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.35
    scene.world = world
    camera = bpy.data.cameras.new('Camera')
    camera.lens = 63
    camera.clip_start, camera.clip_end = 0.1, 100
    obj = bpy.data.objects.new('Camera',camera)
    scene.collection.objects.link(obj)
    obj.location = (4.8,-6.0,2.8)
    obj.rotation_euler = (Vector((0,0,0.55))-obj.location).to_track_quat('-Z','Y').to_euler()
    scene.camera = obj
    scene.render.film_transparent = False
    scene.view_layers[0].use_pass_mist = True
    scene.view_layers[0].use_pass_normal = True
    target = ROOT/'assets/demo_white_model.blend'
    target.parent.mkdir(exist_ok=True)
    assert not target.exists(), '白模场景已存在，不覆盖'
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    record = dict(blender=bpy.app.version_string, device='CPU', lens_mm=63,
                  objects={o.name:list(o.location) for o in [sphere,torus,cube,cylinder]},
                  camera_location=list(obj.location),resolution=512,samples=64)
    (ROOT/'logs/scene.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SCENE_COMPLETE',target,flush=True)


if __name__ == '__main__':
    try:
        import bpy
    except ImportError:
        subprocess.run([str(blender_exe()),'-b','--factory-startup','-P',str(Path(__file__).resolve())],check=True)
    else:
        create_scene()
