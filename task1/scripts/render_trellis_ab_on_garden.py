import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def argv_after_separator():
    argv = sys.argv
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def parse_vec(text):
    return tuple(float(v.strip()) for v in text.split(","))


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]


def bounds_for(objects):
    bpy.context.view_layer.update()
    corners = []
    for obj in objects:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))
    min_v = Vector((min(p.x for p in corners), min(p.y for p in corners), min(p.z for p in corners)))
    max_v = Vector((max(p.x for p in corners), max(p.y for p in corners), max(p.z for p in corners)))
    return min_v, max_v


def normalize_and_place(objects, name, target_height, location, rotation_deg):
    parent = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(parent)
    min_v, max_v = bounds_for(objects)
    center = (min_v + max_v) * 0.5
    height = max(max_v.z - min_v.z, 1e-6)
    scale = target_height / height

    for obj in objects:
        obj.parent = parent
        obj.location = (obj.location - center) * scale
        obj.scale = obj.scale * scale

    parent.rotation_euler = tuple(math.radians(v) for v in rotation_deg)
    parent.location = Vector(location)
    return parent


def brighten_materials(objects, emission_strength):
    seen = set()
    for obj in objects:
        for slot in obj.material_slots:
            mat = slot.material
            if not mat or mat.name in seen:
                continue
            seen.add(mat.name)
            mat.use_nodes = True
            mat.blend_method = "BLEND"
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if not bsdf:
                continue
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.42
            if "Emission Color" in bsdf.inputs:
                base = bsdf.inputs.get("Base Color")
                emission = bsdf.inputs["Emission Color"]
                for link in list(emission.links):
                    mat.node_tree.links.remove(link)
                if base and base.is_linked:
                    mat.node_tree.links.new(base.links[0].from_socket, emission)
                elif base:
                    emission.default_value = base.default_value
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_strength


def create_flat_shadow(location, radius_x, radius_y, alpha):
    mat = bpy.data.materials.new("soft_contact_shadow")
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, alpha)
        bsdf.inputs["Alpha"].default_value = alpha
        bsdf.inputs["Roughness"].default_value = 1.0
    bpy.ops.mesh.primitive_circle_add(vertices=96, radius=1.0, fill_type="TRIFAN", location=location)
    shadow = bpy.context.object
    shadow.name = "soft_contact_shadow"
    shadow.scale = (radius_x, radius_y, 1.0)
    shadow.data.materials.append(mat)
    return shadow


def setup_lighting():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (1.0, 1.0, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(0.0, -2.0, 5.0))
    key = bpy.context.object
    key.name = "large_soft_key"
    key.data.energy = 850
    key.data.size = 6.0

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 4.0))
    sun = bpy.context.object
    sun.name = "garden_sun"
    sun.rotation_euler = (math.radians(48), 0.0, math.radians(32))
    sun.data.energy = 1.5

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 0.0))
    fill = bpy.context.object
    fill.name = "camera_fill"
    fill.data.energy = 520
    fill.data.size = 4.0
    return fill


def setup_camera(width, height):
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.eevee.taa_render_samples = 64
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    return cam


def apply_garden_camera(cam, entry):
    c2w = Matrix(entry["rotation"]).to_4x4()
    c2w.translation = Vector(entry["position"])
    opencv_to_blender = Matrix(
        ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1))
    )
    cam.matrix_world = c2w @ opencv_to_blender

    fx = float(entry["fx"])
    original_width = float(entry["width"])
    cam.data.angle_x = 2.0 * math.atan(original_width / (2.0 * fx))
    cam.data.clip_start = 0.01
    cam.data.clip_end = 1000.0


def update_fill(fill, camera):
    fill.location = camera.location
    fill.rotation_euler = camera.rotation_euler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--a_glb", required=True)
    parser.add_argument("--b_glb", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--width", type=int, default=1297)
    parser.add_argument("--height", type=int, default=840)
    parser.add_argument("--frame_step", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--a_loc", default="-0.55,0.0,0.0")
    parser.add_argument("--b_loc", default="0.55,0.0,0.0")
    parser.add_argument("--a_height", type=float, default=0.55)
    parser.add_argument("--b_height", type=float, default=0.62)
    parser.add_argument("--a_rot", default="0,0,0")
    parser.add_argument("--b_rot", default="0,0,0")
    parser.add_argument("--emission", type=float, default=0.18)
    parser.add_argument("--shadows", action="store_true")
    return parser.parse_args(argv_after_separator())


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    fill = setup_lighting()
    camera = setup_camera(args.width, args.height)

    a_objects = import_glb(Path(args.a_glb))
    normalize_and_place(a_objects, "object_A_trellis", args.a_height, parse_vec(args.a_loc), parse_vec(args.a_rot))
    brighten_materials(a_objects, args.emission)

    b_objects = import_glb(Path(args.b_glb))
    normalize_and_place(b_objects, "object_B_trellis", args.b_height, parse_vec(args.b_loc), parse_vec(args.b_rot))
    brighten_materials(b_objects, args.emission)

    if args.shadows:
        a_loc = parse_vec(args.a_loc)
        b_loc = parse_vec(args.b_loc)
        create_flat_shadow((a_loc[0], a_loc[1], a_loc[2] - args.a_height * 0.50), args.a_height * 0.42, args.a_height * 0.22, 0.20)
        create_flat_shadow((b_loc[0], b_loc[1], b_loc[2] - args.b_height * 0.50), args.b_height * 0.34, args.b_height * 0.22, 0.18)

    with open(args.cameras, "r", encoding="utf-8") as f:
        cameras = json.load(f)
    selected = cameras[:: max(args.frame_step, 1)]
    if args.max_frames > 0:
        selected = selected[: args.max_frames]

    for out_idx, entry in enumerate(selected):
        apply_garden_camera(camera, entry)
        update_fill(fill, camera)
        bpy.context.scene.frame_set(out_idx)
        bpy.context.scene.render.filepath = str(out_dir / f"{out_idx:04d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
