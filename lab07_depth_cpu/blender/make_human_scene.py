"""指南允许的场景替换：用关节与图元构造正面站姿，供头像融合拓展。"""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def create():
    import bpy
    from mathutils import Vector
    bpy.ops.wm.open_mainfile(filepath=str(ROOT/'assets/demo_white_model.blend'))
    for name in ['Sphere','Torus','Beveled_cube','Cylinder']:
        bpy.data.objects.remove(bpy.data.objects[name],do_unlink=True)
    material=bpy.data.materials['White_matte']
    def ellipsoid(name,location,scale):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,radius=1,location=location)
        obj=bpy.context.object
        obj.name=name
        obj.scale=scale
        obj.data.materials.append(material)
        bpy.ops.object.shade_smooth()
    ellipsoid('Head',(0,0,2.56),(0.25,0.23,0.32))
    ellipsoid('Neck',(0,0,2.22),(0.12,0.12,0.19))
    ellipsoid('Torso',(0,0,1.66),(0.36,0.22,0.56))
    ellipsoid('Hips',(0,0,1.13),(0.32,0.23,0.22))
    for side in [-1,1]:
        ellipsoid(f'Arm_{side}',(side*0.46,0,1.66),(0.1,0.13,0.49))
        ellipsoid(f'Hand_{side}',(side*0.47,0,1.12),(0.1,0.1,0.14))
        ellipsoid(f'Leg_{side}',(side*0.17,0,0.62),(0.13,0.15,0.52))
        ellipsoid(f'Foot_{side}',(side*0.17,-0.12,0.12),(0.15,0.27,0.1))
    scene=bpy.context.scene
    camera=scene.camera
    camera.location=(0,-7.1,2.4)
    camera.rotation_euler=(Vector((0,0,1.45))-camera.location).to_track_quat('-Z','Y').to_euler()
    for name in ['Key','Fill','Rim']:
        light=bpy.data.objects[name]
        light.rotation_euler=(Vector((0,0,1.45))-light.location).to_track_quat('-Z','Y').to_euler()
    target=ROOT/'assets/headfig_white_model.blend'
    assert not target.exists(),'人体白模已存在，不覆盖'
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    print('HUMAN_SCENE_COMPLETE',target,flush=True)


if __name__=='__main__':
    try:
        import bpy
    except ImportError:
        blender=Path(os.environ['BLENDER_EXE']) if 'BLENDER_EXE' in os.environ else next((ROOT/'vendor/blender').glob('*/blender.exe'))
        subprocess.run([str(blender),'-b','--factory-startup','-P',str(Path(__file__).resolve())],check=True)
    else:
        create()
