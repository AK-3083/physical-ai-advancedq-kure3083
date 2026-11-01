"""草地の20m山、既知の安全経路、色マーカーと同色負例を生成する。"""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

from sos.markers import color_board
from sos.terrain import Terrain


PLACEMENT_LAYOUTS = {
    0: ((1, 5), (2, 4, 5)),
    1: ((4, 2), (1, 3, 4)),
    2: ((3, 0), (2, 4, 3)),
    3: ((1, 3), (5, 2, 3)),
    4: ((5, 2), (4, 1, 5)),
    5: ((4, 0), (3, 1, 4)),
}


def mesh_file(path, heights, origin, step):
    """地表高度グリッドのXYZ・UV・三角形をColladaとして保存する。"""
    rows, columns = heights.shape
    positions, uv, triangles = [], [], []
    for j in range(rows):
        for i in range(columns):
            positions.extend([origin[0] + i * step, origin[1] + j * step, heights[j, i]])
            uv.extend([i / (columns - 1), j / (rows - 1)])
    for j in range(rows - 1):
        for i in range(columns - 1):
            a = j * columns + i
            triangles.extend([a, a+1, a+columns+1, a, a+columns+1, a+columns])
    count = rows * columns
    points = " ".join(f"{value:.5f}" for value in positions)
    coords = " ".join(f"{value:.6f}" for value in uv)
    indices = " ".join(f"{value} {value}" for value in triangles)
    path.write_text(f"""<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
 <library_geometries><geometry id="terrain"><mesh>
  <source id="positions"><float_array id="position-array" count="{count*3}">{points}</float_array>
   <technique_common><accessor source="#position-array" count="{count}" stride="3">
    <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
   </accessor></technique_common></source>
  <source id="uv"><float_array id="uv-array" count="{count*2}">{coords}</float_array>
   <technique_common><accessor source="#uv-array" count="{count}" stride="2">
    <param name="S" type="float"/><param name="T" type="float"/>
   </accessor></technique_common></source>
  <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
  <triangles count="{len(triangles)//3}">
   <input semantic="VERTEX" source="#vertices" offset="0"/>
   <input semantic="TEXCOORD" source="#uv" offset="1" set="0"/><p>{indices}</p>
  </triangles>
 </mesh></geometry></library_geometries>
 <library_visual_scenes><visual_scene id="scene"><node id="ground">
  <instance_geometry url="#terrain"/>
 </node></visual_scene></library_visual_scenes>
 <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>""", encoding="utf-8")


def generate_mountain(directory, seed=1, count=2, layout=0):
    """固定地形・経路の山に、seedとlayoutに応じた物体を配置する。"""
    if count not in (0, 2):
        raise ValueError("Color scenario marker count must be 0 or 2")
    if layout not in PLACEMENT_LAYOUTS:
        raise ValueError(f"Unknown placement layout: {layout}")
    summit, length, site_count = 20., 60., 6
    bend, radius_limit, clearing_radius = 4., 70., 3.
    directory = Path(directory).resolve()
    textures = directory / "materials" / "textures"
    materials = directory / "materials" / "scripts"
    textures.mkdir(parents=True, exist_ok=True)
    materials.mkdir(parents=True, exist_ok=True)
    axis = np.arange(-80., 141., 2.)
    step = float(axis[1] - axis[0])
    xx, yy = np.meshgrid(axis, axis)
    radius = np.hypot(xx - length, yy)
    heights = summit/2 * (1 + np.cos(np.pi * np.minimum(radius / radius_limit, 1)))
    sites = []
    for index, x in enumerate(np.linspace(0, length, site_count)):
        y = float(bend*np.sin(2*np.pi*x/length))
        z = float(summit/2 * (1 + np.cos(np.pi * min(np.hypot(x-length, y) / radius_limit, 1))))
        if index == 0:
            z = 0.
        sites.append({"id": index, "x": float(x), "y": y, "z": z})
        distance = np.hypot(xx-x, yy-y)
        weight = np.clip((clearing_radius+3-distance) / 3, 0, 1)
        heights = heights * (1-weight) + z * weight
    # Include the flattened terrain points and short intermediate segments.
    coordinates = sorted(set(np.linspace(0, length, 31).tolist()
                             + [site["x"] for site in sites]))
    path = [{"x": x, "y": float(bend*np.sin(2*np.pi*x/length)),
             "capture": False, "site": None} for x in coordinates]
    data = {"mode": "mountain", "origin": [float(axis[0])]*2, "step": step,
            "heights": heights.tolist(), "sites": sites, "path": path, "trees": [],
            "summit_height_m": summit, "scope": "Camera-visible known trail corridor",
            "scheme": "color", "transit_agl_m": 10., "capture_agl_m": 5.}
    terrain = Terrain(data)
    tree_rng = np.random.default_rng(950)
    trees = []
    for _ in range(120):
        x, y = tree_rng.uniform(-20, 90), tree_rng.uniform(-40, 40)
        if min(np.hypot(x-item["x"], y-item["y"]) for item in path) < 9:
            continue
        trees.append({"x": float(x), "y": float(y), "z": terrain.height(x, y),
                      "height": float(tree_rng.uniform(6, 10)), "radius": 2.5})
    data["trees"] = trees
    terrain = Terrain(data)
    terrain.search_route()
    (directory / "terrain.json").write_text(json.dumps(data), encoding="utf-8")
    mesh_file(directory / "mountain.dae", heights, (axis[0], axis[0]), step)
    texture_rng = np.random.default_rng(800)
    texture = np.empty((1024, 1024, 3), dtype=np.uint8)
    noise = texture_rng.integers(-15, 16, (1024, 1024))
    texture[:] = (50, 87, 70)
    texture = np.clip(texture.astype(int) + noise[:, :, None], 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(textures / "ground.png"), texture[::-1]):
        raise OSError("Could not save mountain texture")
    definitions = ["""material SOS/Mountain
{
 technique
 {
  pass
  {
   lighting off
   texture_unit
   {
    texture ground.png
   }
  }
 }
}"""]
    uri = f"<uri>file://{escape(str(materials))}</uri><uri>file://{escape(str(textures))}</uri>"
    mesh = f"<mesh><uri>file://{escape(str(directory / 'mountain.dae'))}</uri></mesh>"
    objects = [f"""<model name="mountain"><static>true</static><link name="ground">
      <collision name="ground"><geometry>{mesh}</geometry></collision>
      <visual name="ground"><geometry>{mesh}</geometry>
       <material><script>{uri}<name>SOS/Mountain</name></script></material>
      </visual></link></model>"""]
    for index, tree in enumerate(trees):
        height = tree["height"]
        objects.append(f"""<model name="tree_{index}"><static>true</static>
         <pose>{tree['x']} {tree['y']} {tree['z']} 0 0 0</pose><link name="tree">
          <collision name="trunk"><pose>0 0 {height/2} 0 0 0</pose><geometry><cylinder><radius>0.3</radius><length>{height}</length></cylinder></geometry></collision>
          <visual name="trunk"><pose>0 0 {height/2} 0 0 0</pose><geometry><cylinder><radius>0.3</radius><length>{height}</length></cylinder></geometry><material><ambient>0.25 0.15 0.08 1</ambient></material></visual>
          <collision name="crown"><pose>0 0 {height-2.5} 0 0 0</pose><geometry><sphere><radius>2.5</radius></sphere></geometry></collision>
          <visual name="crown"><pose>0 0 {height-2.5} 0 0 0</pose><geometry><sphere><radius>2.5</radius></sphere></geometry><material><ambient>0.08 0.3 0.08 1</ambient><diffuse>0.1 0.35 0.1 1</diffuse></material></visual>
         </link></model>""")
    rng = np.random.default_rng(seed)
    truth = []
    negatives = []
    marker_sites, negative_sites = PLACEMENT_LAYOUTS[layout]
    mixed_sites = set(marker_sites) & set(negative_sites)
    placements = ([(site, "marker") for site in marker_sites] if count else [])
    placements += list(zip(negative_sites, ("flower", "clothing", "clothing")))
    for index, (site_index, kind) in enumerate(placements):
        positive = kind == "marker"
        site = sites[site_index]
        jitter = 0.15
        x, y = site["x"] + rng.uniform(-jitter, jitter), site["y"] + rng.uniform(-jitter, jitter)
        if site_index in mixed_sites:
            y += -0.9 if positive else 0.9
        z = terrain.height(x, y) + 0.025
        yaw = rng.uniform(-np.pi, np.pi)
        image = color_board(kind, variant=seed % 7)
        if not cv2.imwrite(str(textures / f"board{index}.png"), image):
            raise OSError("Could not save mountain marker")
        definitions.append(f"""material SOS/Board{index}
{{
 technique
 {{
  pass
  {{
   lighting off
   texture_unit
   {{
    texture board{index}.png
   }}
  }}
 }}
}}""")
        objects.append(f"""<model name="board_{index}"><static>true</static>
         <pose>{x} {y} {z} 0 0 {yaw}</pose><link name="board">
         <visual name="sign"><geometry><plane><normal>0 0 1</normal><size>1.2 0.8</size></plane></geometry>
         <material><script>{uri}<name>SOS/Board{index}</name></script></material></visual>
         </link></model>""")
        if positive:
            truth.append({"id": index+1, "x": x, "y": y, "z": z, "viewpoint": site["id"]})
        else:
            negatives.append({"x": x, "y": y, "z": z, "kind": kind, "viewpoint": site["id"]})
    (materials / "sos.material").write_text("\n".join(definitions), encoding="utf-8")
    world = f"""<?xml version="1.0"?><sdf version="1.6"><world name="search">
      <gravity>0 0 -9.81</gravity><physics type="ode"><max_step_size>0.004</max_step_size>
      <real_time_update_rate>250</real_time_update_rate></physics>
      <scene><ambient>0.7 0.7 0.7 1</ambient><background>0.45 0.65 0.8 1</background></scene>
      <gui><camera name="user_camera"><pose>30 -55 50 0 0.6287962864 1.5707963268</pose>
      <view_controller>orbit</view_controller></camera></gui>
      <light name="sun" type="directional"><cast_shadows>false</cast_shadows>
      <direction>0.1 0.1 -1</direction></light>{''.join(objects)}
      </world></sdf>"""
    (directory / "search.world").write_text(world, encoding="utf-8")
    (directory / "truth.json").write_text(
        json.dumps({"seed": seed, "layout": layout, "mode": "mountain", "markers": truth,
                    "scheme": "color", "negatives": negatives}, indent=2), encoding="utf-8"
    )
    print(f"Created {summit:g}m mountain, {len(trees)} trees, {count} color markers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("assets/color"))
    parser.add_argument("--seed", type=int, default=503)
    parser.add_argument("--count", type=int, choices=(0, 2), default=2)
    parser.add_argument("--layout", type=int, choices=tuple(PLACEMENT_LAYOUTS), default=0)
    args = parser.parse_args()
    generate_mountain(args.directory, args.seed, args.count, args.layout)
