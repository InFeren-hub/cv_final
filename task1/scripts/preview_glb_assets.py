import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def bounds_of(objects):
    pts = []
    for obj in objects:
        for corner in obj.bound_box:
            pts.append(obj.matrix_world @ Vector(corner))
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def normalize(objects):
    lo, hi = bounds_of(objects)
    center = (lo + hi) * 0.5
    height = max(hi.z - lo.z, hi.y - lo.y, hi.x - lo.x, 1e-6)
    scale = 2.4 / height
    parent = bpy.data.objects.new("asset_root", None)
    bpy.context.collection.objects.link(parent)
    for obj in objects:
        obj.parent = parent
        obj.location = (obj.location - center) * scale
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()
    return parent


def setup_scene(output):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 80
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"
    bpy.context.scene.render.resolution_x = 1200
    bpy.context.scene.render.resolution_y = 1200
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.color = (1.0, 1.0, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(0.0, -4.0, 5.0))
    light = bpy.context.object
    light.data.energy = 500
    light.data.size = 5
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 5.0))
    sun = bpy.context.object
    sun.rotation_euler = (math.radians(45), 0.0, math.radians(20))
    sun.data.energy = 1.1

    bpy.ops.object.camera_add(location=(0.0, -5.0, 1.0), rotation=(math.radians(78), 0.0, 0.0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.lens = 58
    bpy.context.scene.render.filepath = str(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv)

    clear_scene()
    objects = import_glb(Path(args.input))
    normalize(objects)
    setup_scene(Path(args.output))
    bpy.ops.render.render(write_still=True)

    lo, hi = bounds_of(objects)
    meta = {
        "input": args.input,
        "output": args.output,
        "mesh_count": len(objects),
        "object_names": [o.name for o in objects],
        "bounds_min": [lo.x, lo.y, lo.z],
        "bounds_max": [hi.x, hi.y, hi.z],
    }
    Path(args.metadata).write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
