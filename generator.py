import gzip
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import shape


# ============================================================
# CONFIG
# ============================================================

CHUNKS_DIR = Path("data/chunks")
OUTPUT_DIR = Path("output")

BUILDING_HEIGHT_DEFAULT = 6.0
BUILDING_LEVEL_HEIGHT = 3.0

ROAD_WIDTH_DEFAULT = 5.0
ROAD_HEIGHT = 0.04

WATER_WIDTH_DEFAULT = 3.0
WATER_HEIGHT = 0.02

# Terrain resolution used for prototype.
# 121 x 121 = 14,641 vertices.
TERRAIN_RESOLUTION = 121

# Vertical scale.
TERRAIN_VERTICAL_SCALE = 1.0

# Select first available chunk automatically.
CHUNK_ID = os.environ.get("CHUNK_ID", "")


# ============================================================
# HELPERS
# ============================================================

def log(message=""):
    print(message, flush=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def find_chunk():

    if CHUNK_ID:

        candidate = (
            CHUNKS_DIR / CHUNK_ID
        )

        if candidate.is_dir():
            return candidate

        raise RuntimeError(
            f"Requested chunk does not exist: "
            f"{CHUNK_ID}"
        )

    chunks = sorted(
        path
        for path in CHUNKS_DIR.iterdir()
        if path.is_dir()
        and path.name.startswith("PUNE_")
    )

    if not chunks:

        raise RuntimeError(
            "No PUNE_* chunk found."
        )

    return chunks[0]


def read_geojson(path: Path):

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def feature_geometry(feature):

    geometry = feature.get(
        "geometry"
    )

    if not geometry:
        return None

    try:
        return shape(geometry)
    except Exception:
        return None


# ============================================================
# COORDINATE SYSTEM
# ============================================================

def lonlat_to_local(
    lon,
    lat,
    origin_lon,
    origin_lat,
):
    """
    Convert WGS84 coordinates to local
    meter coordinates.

    This is suitable for a 1 km prototype
    chunk. For a large final world we will
    later use a proper projected/local origin.
    """

    earth_radius = 6378137.0

    lat_scale = (
        math.pi
        / 180.0
        * earth_radius
    )

    lon_scale = (
        math.pi
        / 180.0
        * earth_radius
        * math.cos(
            math.radians(origin_lat)
        )
    )

    x = (
        lon - origin_lon
    ) * lon_scale

    y = (
        lat - origin_lat
    ) * lat_scale

    return x, y


# ============================================================
# BUILDING HEIGHT
# ============================================================

def get_building_height(properties):

    # Explicit height.
    height = properties.get("height")

    if height is not None:

        try:
            value = float(
                str(height)
                .replace("m", "")
                .strip()
            )

            if value > 0:
                return value

        except Exception:
            pass

    # Building levels.
    levels = properties.get(
        "building:levels"
    )

    if levels is None:
        levels = properties.get(
            "levels"
        )

    if levels is not None:

        try:

            value = float(
                str(levels)
                .replace(",", ".")
                .strip()
            )

            if value > 0:

                return (
                    value
                    * BUILDING_LEVEL_HEIGHT
                )

        except Exception:
            pass

    return BUILDING_HEIGHT_DEFAULT


# ============================================================
# BUILDING MESH
# ============================================================

def polygon_to_building(
    polygon,
    properties,
    origin_lon,
    origin_lat,
):

    if polygon.is_empty:
        return None

    if polygon.geom_type != "Polygon":
        return None

    height = get_building_height(
        properties
    )

    exterior = list(
        polygon.exterior.coords
    )

    if len(exterior) < 4:
        return None

    points = []

    for lon, lat in exterior[:-1]:

        x, y = lonlat_to_local(
            lon,
            lat,
            origin_lon,
            origin_lat,
        )

        points.append(
            [x, y]
        )

    if len(points) < 3:
        return None

    # Remove duplicate consecutive points.
    clean = []

    for point in points:

        if not clean or point != clean[-1]:
            clean.append(point)

    points = clean

    if len(points) < 3:
        return None

    vertices = []

    # Bottom vertices.
    for x, y in points:

        vertices.append(
            [x, y, 0.0]
        )

    # Top vertices.
    for x, y in points:

        vertices.append(
            [
                x,
                y,
                height,
            ]
        )

    n = len(points)

    faces = []

    # Top / bottom using triangle fan.
    for i in range(1, n - 1):

        faces.append(
            [0, i, i + 1]
        )

        faces.append(
            [
                n,
                n + i + 1,
                n + i,
            ]
        )

    # Walls.
    for i in range(n):

        j = (
            i + 1
        ) % n

        faces.append(
            [
                i,
                j,
                n + j,
            ]
        )

        faces.append(
            [
                i,
                n + j,
                n + i,
            ]
        )

    try:

        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                vertices,
                dtype=np.float64,
            ),
            faces=np.asarray(
                faces,
                dtype=np.int64,
            ),
            process=False,
        )

        return mesh

    except Exception:

        return None


# ============================================================
# BUILDINGS
# ============================================================

def generate_buildings(
    chunk_dir,
    origin_lon,
    origin_lat,
):

    path = (
        chunk_dir
        / "buildings.geojson"
    )

    data = read_geojson(path)

    if not data:
        log(
            "No buildings.geojson found."
        )
        return []

    meshes = []

    features = data.get(
        "features",
        [],
    )

    log(
        f"Building features: "
        f"{len(features)}"
    )

    for feature in features:

        geometry = feature_geometry(
            feature
        )

        if geometry is None:
            continue

        properties = feature.get(
            "properties"
        ) or {}

        if geometry.geom_type == "Polygon":

            mesh = polygon_to_building(
                geometry,
                properties,
                origin_lon,
                origin_lat,
            )

            if mesh is not None:
                meshes.append(mesh)

        elif geometry.geom_type == "MultiPolygon":

            for polygon in geometry.geoms:

                mesh = polygon_to_building(
                    polygon,
                    properties,
                    origin_lon,
                    origin_lat,
                )

                if mesh is not None:
                    meshes.append(mesh)

    log(
        f"Building meshes: "
        f"{len(meshes)}"
    )

    return meshes


# ============================================================
# LINE MESH
# ============================================================

def line_to_strip(
    line,
    width,
    height,
    origin_lon,
    origin_lat,
):

    if line.is_empty:
        return None

    if line.geom_type != "LineString":
        return None

    coords = list(
        line.coords
    )

    if len(coords) < 2:
        return None

    half = width / 2.0

    vertices = []
    faces = []

    previous_left = None
    previous_right = None

    for index, coordinate in enumerate(
        coords
    ):

        lon, lat = coordinate[:2]

        x, y = lonlat_to_local(
            lon,
            lat,
            origin_lon,
            origin_lat,
        )

        if index == 0:

            lon2, lat2 = coords[
                index + 1
            ][:2]

            x2, y2 = lonlat_to_local(
                lon2,
                lat2,
                origin_lon,
                origin_lat,
            )

        elif index == len(coords) - 1:

            lon2, lat2 = coords[
                index - 1
            ][:2]

            x2, y2 = lonlat_to_local(
                lon2,
                lat2,
                origin_lon,
                origin_lat,
            )

        else:

            lon1, lat1 = coords[
                index - 1
            ][:2]

            lon2, lat2 = coords[
                index + 1
            ][:2]

            x1, y1 = lonlat_to_local(
                lon1,
                lat1,
                origin_lon,
                origin_lat,
            )

            x2, y2 = lonlat_to_local(
                lon2,
                lat2,
                origin_lon,
                origin_lat,
            )

            dx = x2 - x1
            dy = y2 - y1

            length = math.hypot(
                dx,
                dy,
            )

            if length < 0.000001:
                continue

            nx = -dy / length
            ny = dx / length

            left = (
                x + nx * half,
                y + ny * half,
            )

            right = (
                x - nx * half,
                y - ny * half,
            )

            vertices.extend(
                [
                    [
                        left[0],
                        left[1],
                        height,
                    ],
                    [
                        right[0],
                        right[1],
                        height,
                    ],
                ]
            )

            if previous_left is not None:

                current_left = (
                    len(vertices) - 2
                )

                current_right = (
                    len(vertices) - 1
                )

                faces.append(
                    [
                        previous_left,
                        previous_right,
                        current_right,
                    ]
                )

                faces.append(
                    [
                        previous_left,
                        current_right,
                        current_left,
                    ]
                )

            previous_left = (
                len(vertices) - 2
            )

            previous_right = (
                len(vertices) - 1
            )

            continue

        dx = x2 - x
        dy = y2 - y

        length = math.hypot(
            dx,
            dy,
        )

        if length < 0.000001:
            continue

        nx = -dy / length
        ny = dx / length

        left = (
            x + nx * half,
            y + ny * half,
        )

        right = (
            x - nx * half,
            y - ny * half,
        )

        vertices.extend(
            [
                [
                    left[0],
                    left[1],
                    height,
                ],
                [
                    right[0],
                    right[1],
                    height,
                ],
            ]
        )

        if previous_left is not None:

            current_left = (
                len(vertices) - 2
            )

            current_right = (
                len(vertices) - 1
            )

            faces.append(
                [
                    previous_left,
                    previous_right,
                    current_right,
                ]
            )

            faces.append(
                [
                    previous_left,
                    current_right,
                    current_left,
                ]
            )

        previous_left = (
            len(vertices) - 2
        )

        previous_right = (
            len(vertices) - 1
        )

    if not faces:
        return None

    return trimesh.Trimesh(
        vertices=np.asarray(
            vertices,
            dtype=np.float64,
        ),
        faces=np.asarray(
            faces,
            dtype=np.int64,
        ),
        process=False,
    )


# ============================================================
# ROADS
# ============================================================

def generate_roads(
    chunk_dir,
    origin_lon,
    origin_lat,
):

    path = (
        chunk_dir
        / "roads.geojson"
    )

    data = read_geojson(path)

    if not data:
        log(
            "No roads.geojson found."
        )
        return []

    meshes = []

    features = data.get(
        "features",
        [],
    )

    log(
        f"Road features: "
        f"{len(features)}"
    )

    for feature in features:

        geometry = feature_geometry(
            feature
        )

        if geometry is None:
            continue

        properties = feature.get(
            "properties"
        ) or {}

        highway = properties.get(
            "highway",
            "",
        )

        # Basic width approximation.
        width = ROAD_WIDTH_DEFAULT

        if highway in {
            "motorway",
            "trunk",
        }:
            width = 10.0

        elif highway in {
            "primary",
        }:
            width = 8.0

        elif highway in {
            "secondary",
        }:
            width = 7.0

        elif highway in {
            "tertiary",
        }:
            width = 6.0

        elif highway in {
            "residential",
            "living_street",
        }:
            width = 5.0

        elif highway in {
            "service",
            "track",
        }:
            width = 3.5

        lines = []

        if geometry.geom_type == "LineString":

            lines = [geometry]

        elif geometry.geom_type == "MultiLineString":

            lines = list(
                geometry.geoms
            )

        for line in lines:

            mesh = line_to_strip(
                line,
                width,
                ROAD_HEIGHT,
                origin_lon,
                origin_lat,
            )

            if mesh is not None:
                meshes.append(mesh)

    log(
        f"Road meshes: "
        f"{len(meshes)}"
    )

    return meshes


# ============================================================
# WATER
# ============================================================

def generate_water(
    chunk_dir,
    origin_lon,
    origin_lat,
):

    path = (
        chunk_dir
        / "waterways.geojson"
    )

    data = read_geojson(path)

    if not data:
        log(
            "No waterways.geojson found."
        )
        return []

    meshes = []

    features = data.get(
        "features",
        [],
    )

    for feature in features:

        geometry = feature_geometry(
            feature
        )

        if geometry is None:
            continue

        if geometry.geom_type == "LineString":

            mesh = line_to_strip(
                geometry,
                WATER_WIDTH_DEFAULT,
                WATER_HEIGHT,
                origin_lon,
                origin_lat,
            )

            if mesh is not None:
                meshes.append(mesh)

        elif geometry.geom_type == "MultiLineString":

            for line in geometry.geoms:

                mesh = line_to_strip(
                    line,
                    WATER_WIDTH_DEFAULT,
                    WATER_HEIGHT,
                    origin_lon,
                    origin_lat,
                )

                if mesh is not None:
                    meshes.append(mesh)

    log(
        f"Water meshes: "
        f"{len(meshes)}"
    )

    return meshes


# ============================================================
# DEM
# ============================================================

def find_dem_for_chunk(
    min_lat,
    min_lon,
):

    latitude = math.floor(
        min_lat
    )

    longitude = math.floor(
        min_lon
    )

    ns = (
        "N"
        if latitude >= 0
        else "S"
    )

    ew = (
        "E"
        if longitude >= 0
        else "W"
    )

    tile_name = (
        f"{ns}"
        f"{abs(latitude):02d}"
        f"{ew}"
        f"{abs(longitude):03d}"
    )

    path = (
        Path("data/raw/dem")
        / f"{tile_name}.hgt.gz"
    )

    return path, tile_name


def read_hgt(path):

    with gzip.open(
        path,
        "rb",
    ) as file:

        raw = file.read()

    values = np.frombuffer(
        raw,
        dtype=">i2",
    )

    count = values.size

    side = int(
        math.sqrt(count)
    )

    if side * side != count:
        raise RuntimeError(
            f"Invalid HGT dimensions: "
            f"{path}"
        )

    return values.reshape(
        side,
        side,
    )


def create_terrain(
    metadata,
    origin_lon,
    origin_lat,
):

    bounds = metadata.get(
        "bounds_wgs84"
    )

    if not bounds:
        log(
            "No bounds in metadata."
        )
        return None

    min_lon, min_lat, max_lon, max_lat = (
        bounds
    )

    dem_path, tile_name = find_dem_for_chunk(
        min_lat,
        min_lon,
    )

    if not dem_path.exists():

        log(
            f"DEM not found: "
            f"{dem_path}"
        )

        return None

    log(
        f"Reading DEM: "
        f"{tile_name}"
    )

    elevation = read_hgt(
        dem_path
    )

    resolution = TERRAIN_RESOLUTION

    # HGT is normally 3601 x 3601 or 1201 x 1201.
    # Downsample to prototype resolution.
    rows = np.linspace(
        0,
        elevation.shape[0] - 1,
        resolution,
    ).astype(int)

    cols = np.linspace(
        0,
        elevation.shape[1] - 1,
        resolution,
    ).astype(int)

    sampled = elevation[
        np.ix_(rows, cols)
    ].astype(
        np.float64
    )

    # Local origin elevation.
    base_elevation = float(
        np.nanmean(sampled)
    )

    vertices = []

    for r in range(resolution):

        # HGT starts north and goes south.
        fraction_y = (
            r
            / (resolution - 1)
        )

        lat = (
            max_lat
            - (
                max_lat
                - min_lat
            )
            * fraction_y
        )

        for c in range(resolution):

            fraction_x = (
                c
                / (resolution - 1)
            )

            lon = (
                min_lon
                + (
                    max_lon
                    - min_lon
                )
                * fraction_x
            )

            x, y = lonlat_to_local(
                lon,
                lat,
                origin_lon,
                origin_lat,
            )

            z = (
                sampled[r, c]
                - base_elevation
            ) * TERRAIN_VERTICAL_SCALE

            vertices.append(
                [
                    x,
                    y,
                    z,
                ]
            )

    faces = []

    for r in range(
        resolution - 1
    ):

        for c in range(
            resolution - 1
        ):

            a = (
                r * resolution
                + c
            )

            b = a + 1

            d = (
                (r + 1)
                * resolution
                + c
            )

            e = d + 1

            faces.append(
                [a, d, b]
            )

            faces.append(
                [b, d, e]
            )

    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            vertices,
            dtype=np.float64,
        ),
        faces=np.asarray(
            faces,
            dtype=np.int64,
        ),
        process=False,
    )

    log(
        f"Terrain vertices: "
        f"{len(vertices)}"
    )

    log(
        f"Terrain triangles: "
        f"{len(faces)}"
    )

    return mesh


# ============================================================
# MATERIALS
# ============================================================

def make_material(
    name,
    color,
):

    try:

        return trimesh.visual.material.PBRMaterial(
            name=name,
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.8,
        )

    except Exception:

        return None


def apply_material(
    mesh,
    material,
):

    if material is None:
        return

    try:
        mesh.visual.material = material
    except Exception:
        pass


# ============================================================
# EXPORT
# ============================================================

def export_scene(
    terrain,
    buildings,
    roads,
    water,
    output_path,
):

    scene = trimesh.Scene()

    if terrain is not None:

        apply_material(
            terrain,
            make_material(
                "Terrain",
                [
                    110,
                    140,
                    90,
                    255,
                ],
            ),
        )

        scene.add_geometry(
            terrain,
            node_name="terrain",
        )

    building_material = make_material(
        "Buildings",
        [
            190,
            190,
            190,
            255,
        ],
    )

    for index, mesh in enumerate(
        buildings
    ):

        apply_material(
            mesh,
            building_material,
        )

        scene.add_geometry(
            mesh,
            node_name=f"building_{index}",
        )

    road_material = make_material(
        "Roads",
        [
            60,
            60,
            60,
            255,
        ],
    )

    for index, mesh in enumerate(
        roads
    ):

        apply_material(
            mesh,
            road_material,
        )

        scene.add_geometry(
            mesh,
            node_name=f"road_{index}",
        )

    water_material = make_material(
        "Water",
        [
            50,
            100,
            180,
            255,
        ],
    )

    for index, mesh in enumerate(
        water
    ):

        apply_material(
            mesh,
            water_material,
        )

        scene.add_geometry(
            mesh,
            node_name=f"water_{index}",
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log()
    log(
        "Exporting GLB..."
    )

    scene.export(
        str(output_path),
        file_type="glb",
    )

    log()
    log(
        f"GLB created: "
        f"{output_path}"
    )

    log(
        f"Size: "
        f"{output_path.stat().st_size / 1024 / 1024:.2f} MiB"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log()
    log("=" * 70)
    log("PUNE 3D GENERATOR")
    log("=" * 70)

    chunk_dir = find_chunk()

    log()
    log(
        f"Selected chunk: "
        f"{chunk_dir.name}"
    )

    metadata_path = (
        chunk_dir
        / "metadata.json"
    )

    if not metadata_path.exists():

        raise RuntimeError(
            f"metadata.json not found: "
            f"{metadata_path}"
        )

    metadata = load_json(
        metadata_path
    )

    bounds = metadata.get(
        "bounds_wgs84"
    )

    if not bounds:

        raise RuntimeError(
            "Chunk metadata does not "
            "contain bounds_wgs84."
        )

    min_lon, min_lat, max_lon, max_lat = (
        bounds
    )

    # Use south-west corner as local origin.
    origin_lon = min_lon
    origin_lat = min_lat

    log()
    log(
        "Chunk bounds:"
    )

    log(
        f"Longitude: "
        f"{min_lon} → {max_lon}"
    )

    log(
        f"Latitude: "
        f"{min_lat} → {max_lat}"
    )

    log()
    log(
        "Local origin:"
    )

    log(
        f"{origin_lon}, "
        f"{origin_lat}"
    )

    # --------------------------------------------------------
    # Terrain
    # --------------------------------------------------------

    log()
    log("=" * 70)
    log("TERRAIN")
    log("=" * 70)

    terrain = create_terrain(
        metadata,
        origin_lon,
        origin_lat,
    )

    # --------------------------------------------------------
    # Buildings
    # --------------------------------------------------------

    log()
    log("=" * 70)
    log("BUILDINGS")
    log("=" * 70)

    buildings = generate_buildings(
        chunk_dir,
        origin_lon,
        origin_lat,
    )

    # --------------------------------------------------------
    # Roads
    # --------------------------------------------------------

    log()
    log("=" * 70)
    log("ROADS")
    log("=" * 70)

    roads = generate_roads(
        chunk_dir,
        origin_lon,
        origin_lat,
    )

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    log()
    log("=" * 70)
    log("WATER")
    log("=" * 70)

    water = generate_water(
        chunk_dir,
        origin_lon,
        origin_lat,
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_DIR
        / chunk_dir.name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    glb_path = (
        output_dir
        / "pune_chunk.glb"
    )

    export_scene(
        terrain,
        buildings,
        roads,
        water,
        glb_path,
    )

    # --------------------------------------------------------
    # Generator metadata
    # --------------------------------------------------------

    result = {
        "chunk_id": chunk_dir.name,
        "bounds_wgs84": bounds,
        "origin_wgs84": [
            origin_lon,
            origin_lat,
        ],
        "terrain": terrain is not None,
        "building_count": len(
            buildings
        ),
        "road_count": len(
            roads
        ),
        "water_count": len(
            water
        ),
        "glb": str(
            glb_path
        ),
    }

    save_json(
        output_dir
        / "generator_metadata.json",
        result,
    )

    log()
    log("=" * 70)
    log("GENERATION COMPLETE")
    log("=" * 70)

    log()
    log(
        f"GLB: {glb_path}"
    )

    log(
        f"Buildings: "
        f"{len(buildings)}"
    )

    log(
        f"Roads: "
        f"{len(roads)}"
    )

    log(
        f"Water: "
        f"{len(water)}"
    )


if __name__ == "__main__":
    main()
