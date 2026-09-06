//! Placeholder city layout + triangle soup builder.
//!
//! One ground quad, a thin-quad survey grid, and a handful of flat-shaded
//! boxes. Everything is axis-aligned, so picking is a 2D footprint test
//! (see [`pick`]). Geometry is rebuilt on the CPU per frame and drawn
//! back-to-front (painter's algorithm) — no depth buffer yet; switch to
//! GPU depth + instancing once the scene outgrows a few thousand tris.

use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::camera::OrbitCamera;

/// Linear-space RGB triplets (authored flat, output raw).
pub type Rgb = [f32; 3];

pub const GRASS: Rgb = [0.62, 0.72, 0.45];
pub const GRID_LINE: Rgb = [0.55, 0.65, 0.39];
pub const GROUND_SIZE: f32 = 240.0;
pub const GRID_STEP: f32 = 10.0;
pub const GRID_HALF: f32 = 120.0;

#[derive(Clone, Copy, Debug)]
pub struct BlockDef {
    pub id: &'static str,
    pub cx: f32,
    pub cz: f32,
    pub w: f32,
    pub h: f32,
    pub d: f32,
    pub color: Rgb,
}

/// Starter skyline near the origin. Footprints do not overlap, which keeps
/// footprint picking unambiguous.
pub const BLOCKS: [BlockDef; 7] = [    BlockDef { id: "bld:1", cx: -14.0, cz: -8.0, w: 12.0, h: 7.0, d: 8.0, color: [0.93, 0.93, 0.93] },
    BlockDef { id: "bld:2", cx: 10.0, cz: -14.0, w: 9.0, h: 12.0, d: 9.0, color: [0.88, 0.82, 0.72] },
    BlockDef { id: "bld:3", cx: 12.0, cz: 10.0, w: 8.0, h: 5.0, d: 8.0, color: [0.72, 0.45, 0.35] },
    BlockDef { id: "bld:4", cx: -12.0, cz: 12.0, w: 10.0, h: 9.0, d: 7.0, color: [0.82, 0.82, 0.84] },
    BlockDef { id: "bld:5", cx: -2.0, cz: -26.0, w: 14.0, h: 6.0, d: 6.0, color: [0.75, 0.68, 0.45] },
    BlockDef { id: "bld:6", cx: 28.0, cz: 2.0, w: 7.0, h: 15.0, d: 7.0, color: [0.55, 0.62, 0.72] },
    BlockDef { id: "bld:7", cx: -30.0, cz: -2.0, w: 8.0, h: 4.0, d: 10.0, color: [0.65, 0.72, 0.60] },
];

/// Dynamic agent avatar from the sim: outfit body + skin head + a
/// floating state badge (activity reads at a glance, identity is stable).
#[derive(Clone, Copy, Debug)]
pub struct AgentMarker {
    pub x: f32,
    pub z: f32,
    pub outfit: Rgb,
    pub skin: Rgb,
    pub accent: Rgb,
}

/// Flat planning marker (diamond quad on the ground).
#[derive(Clone, Copy, Debug)]
pub struct GroundMarker {
    pub x: f32,
    pub z: f32,
    pub color: Rgb,
    pub size: f32,
}
/// Planning path (road centre-line); rendered as a flat ribbon.
#[derive(Clone, Debug)]
pub struct PathLine {
    pub points: Vec<[f32; 2]>,
    pub width: f32,
    pub color: Rgb,
}

/// Zone fill: flat polygon (triangle fan — assumes roughly convex zones).
#[derive(Clone, Debug)]
pub struct PathPoly {
    pub points: Vec<[f32; 2]>,
    pub color: Rgb,
}

/// Built wall: vertical box strip along a segment (6 tris: 2 sides + top).
#[derive(Clone, Debug)]
pub struct WallSeg {
    pub ax: f32,
    pub az: f32,
    pub bx: f32,
    pub bz: f32,
    pub height: f32,
    pub color: Rgb,
}

/// Furniture prop: small shaded box (fridge/bed/sofa by size+color).
#[derive(Clone, Debug)]
pub struct PropBox {
    pub cx: f32,
    pub cz: f32,
    pub w: f32,
    pub h: f32,
    pub d: f32,
    pub color: Rgb,
}
/// Painter layers (drawn first -> last). Flat overlays hover micrometers
/// above the ground (grid 0.02, marker 0.06, poly 0.07, path 0.09), so
/// pure centroid-depth sorting flips their order against the giant ground
/// halves at grazing angles and the ground paints over them. Layers pin
/// the physical stacking; `depth` still sorts within a layer.
const LAYER_GROUND: u8 = 0;
const LAYER_GRID: u8 = 1;
const LAYER_POLY: u8 = 2;
const LAYER_PATH: u8 = 3;
const LAYER_MARKER: u8 = 4;
/// True 3D volumes (boxes, walls, props, agents): centroid depth orders
/// them correctly, so they share one layer.
const LAYER_SOLID: u8 = 5;

/// Raw triangle with its painter layer (world space, pre-projection).
type RawTri = ([Vec3; 3], Rgb, u8);

fn quad_tris(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    color: Rgb,
    shade: f32,
    layer: u8,
) -> [RawTri; 2] {
    let col = [color[0] * shade, color[1] * shade, color[2] * shade];
    [([a, b, c], col, layer), ([a, c, d], col, layer)]
}

fn push_box(tris: &mut Vec<RawTri>, b: &BlockDef, eye: Vec3) {
    push_box_at(tris, b.cx, b.cz, b.w, b.h, b.d, b.color, eye);
}

/// Prop (furniture) box: same shading as buildings.
fn push_prop(tris: &mut Vec<RawTri>, p: &PropBox, eye: Vec3) {
    push_box_at(tris, p.cx, p.cz, p.w, p.h, p.d, p.color, eye);
}

/// Analytic front-face test: does the plane (unit `dir` through `pt`)
/// face the eye? Winding-independent (unlike GPU culling), with ~1° of
/// slop so near-edge-on slivers are dropped while still invisible.
/// Opposing faces of thin geometry tie in mean depth and flicker while
/// orbiting; culling the far one removes the tie structurally. Safe for
/// convex volumes viewed from outside: culled faces are always occluded.
fn faces_eye(dir: Vec3, pt: Vec3, eye: Vec3) -> bool {
    let to = eye - pt;
    dir.dot(to) > 0.02 * to.length().max(1e-6)
}

fn push_box_at(
    tris: &mut Vec<RawTri>,
    cx: f32,
    cz: f32,
    w: f32,
    h: f32,
    d: f32,
    c: Rgb,
    eye: Vec3,
) {
    let x0 = cx - w / 2.0;
    let x1 = cx + w / 2.0;
    let z0 = cz - d / 2.0;
    let z1 = cz + d / 2.0;
    // top (+Y)
    if faces_eye(Vec3::Y, Vec3::new(cx, h, cz), eye) {
        tris.extend(quad_tris(
            Vec3::new(x0, h, z0),
            Vec3::new(x0, h, z1),
            Vec3::new(x1, h, z1),
            Vec3::new(x1, h, z0),
            c, 1.0,
            LAYER_SOLID,
        ));
    }
    // +X
    if faces_eye(Vec3::X, Vec3::new(x1, h / 2.0, cz), eye) {
        tris.extend(quad_tris(
            Vec3::new(x1, 0.0, z0),
            Vec3::new(x1, 0.0, z1),
            Vec3::new(x1, h, z1),
            Vec3::new(x1, h, z0),
            c, 0.82,
            LAYER_SOLID,
        ));
    }
    // -X
    if faces_eye(Vec3::NEG_X, Vec3::new(x0, h / 2.0, cz), eye) {
        tris.extend(quad_tris(
            Vec3::new(x0, 0.0, z1),
            Vec3::new(x0, 0.0, z0),
            Vec3::new(x0, h, z0),
            Vec3::new(x0, h, z1),
            c, 0.72,
            LAYER_SOLID,
        ));
    }
    // +Z
    if faces_eye(Vec3::Z, Vec3::new(cx, h / 2.0, z1), eye) {
        tris.extend(quad_tris(
            Vec3::new(x1, 0.0, z1),
            Vec3::new(x0, 0.0, z1),
            Vec3::new(x0, h, z1),
            Vec3::new(x1, h, z1),
            c, 0.88,
            LAYER_SOLID,
        ));
    }
    // -Z
    if faces_eye(Vec3::NEG_Z, Vec3::new(cx, h / 2.0, z0), eye) {
        tris.extend(quad_tris(
            Vec3::new(x0, 0.0, z0),
            Vec3::new(x1, 0.0, z0),
            Vec3::new(x1, h, z0),
            Vec3::new(x0, h, z0),
            c, 0.66,
            LAYER_SOLID,
        ));
    }
}

/// Agent avatar: outfit octahedron body, skin octahedron head, flat
/// state badge floating above (18 tris, all solid layer).
fn push_agent(tris: &mut Vec<RawTri>, a: &AgentMarker) {
    let body_top = Vec3::new(a.x, 1.9, a.z);
    let body_bottom = Vec3::new(a.x, 0.3, a.z);
    let ring = [
        Vec3::new(a.x + 0.9, 1.1, a.z),
        Vec3::new(a.x, 1.1, a.z + 0.9),
        Vec3::new(a.x - 0.9, 1.1, a.z),
        Vec3::new(a.x, 1.1, a.z - 0.9),
    ];
    let bright = [a.outfit[0], a.outfit[1], a.outfit[2]];
    let dim = [a.outfit[0] * 0.7, a.outfit[1] * 0.7, a.outfit[2] * 0.7];
    for i in 0..4 {
        let p = ring[i];
        let q = ring[(i + 1) % 4];
        tris.push(([body_top, p, q], bright, LAYER_SOLID));
        tris.push(([body_bottom, q, p], dim, LAYER_SOLID));
    }
    // Head: smaller octahedron on top.
    let head_top = Vec3::new(a.x, 3.0, a.z);
    let head_bottom = Vec3::new(a.x, 1.9, a.z);
    let hring = [
        Vec3::new(a.x + 0.45, 2.45, a.z),
        Vec3::new(a.x, 2.45, a.z + 0.45),
        Vec3::new(a.x - 0.45, 2.45, a.z),
        Vec3::new(a.x, 2.45, a.z - 0.45),
    ];
    let hbright = [a.skin[0], a.skin[1], a.skin[2]];
    let hdim = [a.skin[0] * 0.75, a.skin[1] * 0.75, a.skin[2] * 0.75];
    for i in 0..4 {
        let p = hring[i];
        let q = hring[(i + 1) % 4];
        tris.push(([head_top, p, q], hbright, LAYER_SOLID));
        tris.push(([head_bottom, q, p], hdim, LAYER_SOLID));
    }
    // Badge: flat state-coloured diamond floating above the head.
    let s = 0.45;
    let y = 3.3;
    tris.extend(quad_tris(
        Vec3::new(a.x - s, y, a.z),
        Vec3::new(a.x, y, a.z + s),
        Vec3::new(a.x + s, y, a.z),
        Vec3::new(a.x, y, a.z - s),
        a.accent, 1.0,
        LAYER_SOLID,
    ));
}

/// Planning marker: flat diamond quad just above the ground.
fn push_marker(tris: &mut Vec<RawTri>, m: &GroundMarker) {
    let s = m.size;
    let y = 0.06;
    tris.extend(quad_tris(
        Vec3::new(m.x - s, y, m.z),
        Vec3::new(m.x, y, m.z + s),
        Vec3::new(m.x + s, y, m.z),
        Vec3::new(m.x, y, m.z - s),
        m.color, 1.0,
        LAYER_MARKER,
    ));
}
/// Road ribbon: flat quads along each segment (y just above ground).
/// Joints are butt caps; overlap is harmless under painter sorting.
fn push_path(tris: &mut Vec<RawTri>, p: &PathLine) {
    if p.points.len() < 2 {
        return;
    }
    let y = 0.09;
    let hw = p.width / 2.0;
    for w in p.points.windows(2) {
        let (ax, az) = (w[0][0], w[0][1]);
        let (bx, bz) = (w[1][0], w[1][1]);
        let dx = bx - ax;
        let dz = bz - az;
        let len = (dx * dx + dz * dz).sqrt().max(1e-6);
        let (nx, nz) = (-dz / len * hw, dx / len * hw);
        tris.extend(quad_tris(
            Vec3::new(ax - nx, y, az - nz),
            Vec3::new(ax + nx, y, az + nz),
            Vec3::new(bx + nx, y, bz + nz),
            Vec3::new(bx - nx, y, bz - nz),
            p.color,
            1.0,
            LAYER_PATH,
        ));
    }
}

/// Zone fill: triangle fan around the first point. Correct for convex
/// (or roughly convex) zones; concave outlines will need ear clipping.
fn push_poly(tris: &mut Vec<RawTri>, p: &PathPoly) {
    if p.points.len() < 3 {
        return;
    }
    let y = 0.07;
    let a = Vec3::new(p.points[0][0], y, p.points[0][1]);
    for w in p.points[1..].windows(2) {
        let b = Vec3::new(w[0][0], y, w[0][1]);
        let c = Vec3::new(w[1][0], y, w[1][1]);
        tris.push(([a, b, c], p.color, LAYER_POLY));
    }
}

/// Built wall: thin vertical box along the segment (front side + top cap
/// + facing end caps), split into <=2u chunks. Only camera-facing planes
/// are emitted (analytic, winding-independent): opposing sides of thin
/// geometry tie in mean depth and flicker while orbiting, so the far one
/// is culled structurally. Long single faces get mean depths far from
/// their pixels at grazing angles, hence the chunking.
fn push_wall(tris: &mut Vec<RawTri>, w: &WallSeg, eye: Vec3) {
    let dx = w.bx - w.ax;
    let dz = w.bz - w.az;
    let len = (dx * dx + dz * dz).sqrt().max(1e-6);
    let (ux, uz) = (-dz / len, dx / len); // unit left normal
    let (nx, nz) = (ux * 0.2, uz * 0.2);
    let h = w.height;
    let n = (len / 2.0).ceil().max(1.0) as usize;
    for i in 0..n {
        let t0 = i as f32 / n as f32;
        let t1 = (i + 1) as f32 / n as f32;
        let (ax, az) = (w.ax + dx * t0, w.az + dz * t0);
        let (bx, bz) = (w.ax + dx * t1, w.az + dz * t1);
        let (mx, mz) = ((ax + bx) / 2.0, (az + bz) / 2.0);
        // +n side
        if faces_eye(Vec3::new(ux, 0.0, uz), Vec3::new(mx + nx, h / 2.0, mz + nz), eye) {
            tris.extend(quad_tris(
                Vec3::new(ax + nx, 0.0, az + nz),
                Vec3::new(bx + nx, 0.0, bz + nz),
                Vec3::new(bx + nx, h, bz + nz),
                Vec3::new(ax + nx, h, az + nz),
                w.color, 0.82,
                LAYER_SOLID,
            ));
        }
        // -n side
        if faces_eye(Vec3::new(-ux, 0.0, -uz), Vec3::new(mx - nx, h / 2.0, mz - nz), eye) {
            tris.extend(quad_tris(
                Vec3::new(bx - nx, 0.0, bz - nz),
                Vec3::new(ax - nx, 0.0, az - nz),
                Vec3::new(ax - nx, h, az - nz),
                Vec3::new(bx - nx, h, bz - nz),
                w.color, 0.66,
                LAYER_SOLID,
            ));
        }
        // top cap
        if faces_eye(Vec3::Y, Vec3::new(mx, h, mz), eye) {
            tris.extend(quad_tris(
                Vec3::new(ax - nx, h, az - nz),
                Vec3::new(ax + nx, h, az + nz),
                Vec3::new(bx + nx, h, bz + nz),
                Vec3::new(bx - nx, h, bz - nz),
                w.color, 1.0,
                LAYER_SOLID,
            ));
        }
    }
    // End caps (camera-facing only: same tie argument as the sides).
    let (tx, tz) = (dx / len, dz / len);
    for (ex, ez, s) in [(w.ax, w.az, -1.0), (w.bx, w.bz, 1.0)] {
        if faces_eye(
            Vec3::new(tx * s, 0.0, tz * s),
            Vec3::new(ex, h / 2.0, ez),
            eye,
        ) {
            tris.extend(quad_tris(
                Vec3::new(ex - nx * s, 0.0, ez - nz * s),
                Vec3::new(ex + nx * s, 0.0, ez + nz * s),
                Vec3::new(ex + nx * s, h, ez + nz * s),
                Vec3::new(ex - nx * s, h, ez - nz * s),
                w.color, 0.74,
                LAYER_SOLID,
            ));
        }
    }
}

/// Near-plane distance used for CPU-side clipping. Must match the `near`
/// passed to `perspective_gl` in [`OrbitCamera::proj_matrix`]: anything
/// closer is outside the projection's valid range (`w <= 0` would make the
/// perspective divide explode, which is why whole triangles used to pop
/// out of existence whenever one corner swung behind the camera).
pub const NEAR: f32 = 0.5;

/// Clip a triangle against the near plane in view space (Sutherland–Hodgman
/// against `view_z <= -NEAR`). Returns 0 (fully behind), 1 (inside or
/// clipped to a smaller triangle), or 2 (clipped to a quad, fanned)
/// triangles with view-space `z <= -NEAR` guaranteed, so the perspective
/// divide after projection is always safe.
fn clip_near(v: &[Vec3; 3], view: &Mat4) -> Vec<[Vec4; 3]> {
    let mut poly: Vec<Vec4> = v.iter().map(|p| *view * p.extend(1.0)).collect();
    let mut out: Vec<Vec4> = Vec::with_capacity(4);
    for i in 0..poly.len() {
        let cur = poly[i];
        let prev = poly[(i + poly.len() - 1) % poly.len()];
        let cur_in = cur.z <= -NEAR;
        let prev_in = prev.z <= -NEAR;
        if cur_in {
            if !prev_in {
                out.push(intersect_near(prev, cur));
            }
            out.push(cur);
        } else if prev_in {
            out.push(intersect_near(prev, cur));
        }
    }
    poly = out;
    if poly.len() < 3 {
        return Vec::new();
    }
    // Fan-triangulate the surviving polygon (triangle or quad).
    (1..poly.len() - 1)
        .map(|i| [poly[0], poly[i], poly[i + 1]])
        .collect()
}

/// Point where segment `a -> b` crosses the near plane (`z = -NEAR`).
/// View transform is affine so linear interpolation is exact.
fn intersect_near(a: Vec4, b: Vec4) -> Vec4 {
    let t = (-NEAR - a.z) / (b.z - a.z);
    a + (b - a) * t
}

/// Interleaved position+color float buffer (3+3 per vertex) in NDC space,
/// back-to-front. Returns `(floats, tri_count)`.
pub fn build_sorted_tris(
    cam: &OrbitCamera,
    aspect: f32,
    agents: &[AgentMarker],
    markers: &[GroundMarker],
    paths: &[PathLine],
    polys: &[PathPoly],
    walls: &[WallSeg],
    props: &[PropBox],
) -> (Vec<f32>, usize) {
    // `clip_near` outputs VIEW-space points, so they go through the
    // projection alone — `view_proj` would apply the view twice and
    // pivot the whole scene around the wrong center.
    let proj = cam.proj_matrix(aspect);
    let view = cam.view_matrix();
    let eye = cam.eye();

    let mut raw: Vec<RawTri> = Vec::with_capacity(512);
    // Ground.
    let g = GROUND_SIZE / 2.0;
    raw.extend(quad_tris(
        Vec3::new(-g, 0.0, -g),
        Vec3::new(-g, 0.0, g),
        Vec3::new(g, 0.0, g),
        Vec3::new(g, 0.0, -g),
        GRASS, 1.0,
        LAYER_GROUND,
    ));
    // Survey grid as thin quads (0.25u wide) so it survives without lines.
    let mut k = -GRID_HALF;
    while k <= GRID_HALF {
        let t = 0.125;
        raw.extend(quad_tris(
            Vec3::new(k - t, 0.02, -GRID_HALF),
            Vec3::new(k - t, 0.02, GRID_HALF),
            Vec3::new(k + t, 0.02, GRID_HALF),
            Vec3::new(k + t, 0.02, -GRID_HALF),
            GRID_LINE, 1.0,
            LAYER_GRID,
        ));
        raw.extend(quad_tris(
            Vec3::new(-GRID_HALF, 0.02, k - t),
            Vec3::new(-GRID_HALF, 0.02, k + t),
            Vec3::new(GRID_HALF, 0.02, k + t),
            Vec3::new(GRID_HALF, 0.02, k - t),
            GRID_LINE, 1.0,
            LAYER_GRID,
        ));
        k += GRID_STEP;
    }
    for b in BLOCKS {
        push_box(&mut raw, &b, eye);
    }
    for a in agents {
        push_agent(&mut raw, a);
    }
    for m in markers {
        push_marker(&mut raw, m);
    }
    for p in paths {
        push_path(&mut raw, p);
    }
    for p in polys {
        push_poly(&mut raw, p);
    }
    for w in walls {
        push_wall(&mut raw, w, eye);
    }
    for p in props {
        push_prop(&mut raw, p, eye);
    }

    // Transform with near-plane clipping, then painter-sort: layers pin
    // the physical stacking (ground < grid < zone < road < marker <
    // solids), view-depth orders within a layer, far -> near.
    // (Older code dropped the whole triangle when any vertex had clip
    // `w <= 0.05`; giant tris like the ground halves and full-map grid
    // strips constantly lost a corner behind the camera while orbiting,
    // so visible geometry popped in and out by angle.)
    let tris = layout_tris(raw, &proj, &view);

    let mut out = Vec::with_capacity(tris.len() * 18);
    for t in &tris {
        for i in 0..3 {
            out.extend_from_slice(&t.v[i].to_array());
            out.extend_from_slice(&t.c[i]);
        }
    }
    let n = tris.len();
    (out, n)
}

/// Shared clip + project + painter-sort path (renderer and tests use the
/// same layout, so the hunt below checks what is actually drawn).
/// `vd` keeps per-vertex view depths for the painter oracle.
#[derive(Clone)]
struct LaidTri {
    v: [Vec3; 3],
    c: [Rgb; 3],
    depth: f32,
    /// Per-vertex view depths: only the painter-oracle test reads these
    /// (per-pixel depth), the renderer uses the mean `depth`.
    #[allow(dead_code)]
    vd: [f32; 3],
    layer: u8,
}

fn layout_tris(raw: Vec<RawTri>, proj: &Mat4, view: &Mat4) -> Vec<LaidTri> {
    let mut tris: Vec<LaidTri> = Vec::with_capacity(raw.len());
    for (v, col, layer) in raw {
        for clipped in clip_near(&v, view) {
            let c0 = *proj * clipped[0];
            let c1 = *proj * clipped[1];
            let c2 = *proj * clipped[2];
            if c0.w <= 0.0 || c1.w <= 0.0 || c2.w <= 0.0 {
                continue; // safety net: clipping guarantees w > 0
            }
            let ndc = [
                [c0.x / c0.w, c0.y / c0.w, c0.z / c0.w],
                [c1.x / c1.w, c1.y / c1.w, c1.z / c1.w],
                [c2.x / c2.w, c2.y / c2.w, c2.z / c2.w],
            ];
            let vd = [-clipped[0].z, -clipped[1].z, -clipped[2].z];
            tris.push(LaidTri {
                v: [Vec3::from(ndc[0]), Vec3::from(ndc[1]), Vec3::from(ndc[2])],
                c: [col, col, col],
                depth: (vd[0] + vd[1] + vd[2]) / 3.0,
                vd,
                layer,
            });
        }
    }
    tris.sort_by(|a, b| {
        a.layer
            .cmp(&b.layer)
            .then(b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal))
    });
    tris
}

/// Block definition by id (stable for the starter skyline).
pub fn block_by_id(id: &str) -> Option<&'static BlockDef> {
    BLOCKS.iter().find(|b| b.id == id)
}

/// Block id under a viewport pixel, if any.
pub fn pick(cam: &OrbitCamera, aspect: f32, viewport_px: Vec2, px: Vec2) -> Option<&'static str> {
    let g = cam.ground_point(aspect, viewport_px, px)?;
    // Nearest containing footprint wins (footprints do not overlap).
    let mut best: Option<(f32, &'static str)> = None;
    for b in BLOCKS {
        if g.x >= b.cx - b.w / 2.0
            && g.x <= b.cx + b.w / 2.0
            && g.y >= b.cz - b.d / 2.0
            && g.y <= b.cz + b.d / 2.0
        {
            let d2 = (g.x - b.cx).powi(2) + (g.y - b.cz).powi(2);
            if best.is_none_or(|(bd2, _)| d2 < bd2) {
                best = Some((d2, b.id));
            }
        }
    }
    best.map(|(_, id)| id)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASPECT: f32 = 16.0 / 9.0;
    const VIEWPORT: Vec2 = Vec2::new(1600.0, 900.0);

    /// Project a world point to viewport px with the given camera.
    fn to_screen(cam: &OrbitCamera, p: Vec3) -> Vec2 {
        let clip = cam.view_proj(ASPECT) * p.extend(1.0);
        let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
        Vec2::new(
            (ndc.x + 1.0) * 0.5 * VIEWPORT.x,
            (1.0 - ndc.y) * 0.5 * VIEWPORT.y,
        )
    }

    #[test]
    fn ground_never_covers_flat_overlays() {
        use crate::camera::OrbitCamera;
        // Coplanar overlays (grid 0.02, marker 0.06 above ground) must
        // draw AFTER the ground at every angle: pure centroid-depth
        // sorting flips their order at grazing angles, so the ground
        // paints over grid lines/markers and they "disappear".
        let markers = [GroundMarker { x: 10.0, z: 10.0, color: [1.0, 0.0, 1.0], size: 1.2 }];
        for (dist, pitch, yaw) in [
            (25.0, 0.12, 1.0),
            (25.0, 0.12, 2.6),
            (60.0, 0.25, 4.2),
            (110.0, 0.85, 0.7),
        ] {
            let mut cam = OrbitCamera::default();
            cam.dist = dist;
            cam.pitch = pitch;
            cam.yaw = yaw;
            let (floats, _) =
                build_sorted_tris(&cam, ASPECT, &[], &markers, &[], &[], &[], &[]);
            let mut last_ground = 0usize;
            let mut first_overlay = usize::MAX;
            for (i, t) in floats.chunks_exact(18).enumerate() {
                for v in 0..3 {
                    let c = [t[v * 6 + 3], t[v * 6 + 4], t[v * 6 + 5]];
                    if c == GRASS || c == GRID_LINE {
                        last_ground = last_ground.max(i);
                    }
                    if c == [1.0, 0.0, 1.0] {
                        first_overlay = first_overlay.min(i);
                    }
                }
            }
            assert!(first_overlay != usize::MAX, "marker missing at dist {dist} yaw {yaw}");
            assert!(
                last_ground < first_overlay,
                "ground tri {last_ground} drawn over marker (first at {first_overlay}), dist {dist} yaw {yaw}"
            );
        }
    }


    #[test]
    fn pick_roundtrip_hits_known_block() {
        let cam = OrbitCamera::default();
        // bld:1 ground center must pick back to bld:1.
        let b = &BLOCKS[0];
        let px = to_screen(&cam, Vec3::new(b.cx, 0.0, b.cz));
        assert_eq!(pick(&cam, ASPECT, VIEWPORT, px), Some(b.id));
    }

    #[test]
    fn pick_empty_ground_returns_none() {
        let cam = OrbitCamera::default();
        // Far corner of the viewport looks at empty grass.
        let px = Vec2::new(VIEWPORT.x - 4.0, VIEWPORT.y - 4.0);
        assert_eq!(pick(&cam, ASPECT, VIEWPORT, px), None);
    }

    /// Screen-space barycentric of `p` in triangle `t` (2D, NDC x/y).
    /// Returns `None` for degenerate triangles.
    fn bary2(t: &[[f32; 3]; 3], p: [f32; 2]) -> Option<[f32; 3]> {
        let (ax, ay) = (t[0][0], t[0][1]);
        let (bx, by) = (t[1][0], t[1][1]);
        let (cx, cy) = (t[2][0], t[2][1]);
        let d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
        if d.abs() < 1e-12 {
            return None;
        }
        let l0 = ((by - cy) * (p[0] - cx) + (cx - bx) * (p[1] - cy)) / d;
        let l1 = ((cy - ay) * (p[0] - cx) + (ax - cx) * (p[1] - cy)) / d;
        Some([l0, l1, 1.0 - l0 - l1])
    }

    /// Painter oracle: same-layer pairs whose draw order contradicts
    /// per-pixel depth at mutually covered pixels flicker while orbiting.
    /// Uses the real [`layout_tris`] path, so this guards what is drawn.
    /// Prints offenders, fails on any visible (different-color, on-screen)
    /// inversion. Agents are kept apart: interpenetrating octahedra cannot
    /// be depth-sorted by any painter (crowd contact may shimmer slightly).
    #[test]
    fn painter_order_matches_per_pixel_depth() {
        use crate::camera::OrbitCamera;
        let agents = [
            AgentMarker { x: 2.0, z: 2.0, outfit: [1.0, 0.0, 0.0], skin: [0.9, 0.8, 0.7], accent: [1.0, 1.0, 0.0] },
            AgentMarker { x: 30.0, z: -20.0, outfit: [0.0, 1.0, 1.0], skin: [0.9, 0.8, 0.7], accent: [1.0, 1.0, 0.0] },
        ];
        let markers = [GroundMarker { x: 10.0, z: 10.0, color: [1.0, 0.0, 1.0], size: 1.2 }];
        let paths = [PathLine {
            points: vec![[-8.0, 0.0], [0.0, 0.0], [8.0, 8.0]],
            width: 2.0,
            color: [1.0, 1.0, 1.0],
        }];
        let polys = [PathPoly {
            points: vec![[20.0, 0.0], [28.0, 0.0], [28.0, 8.0], [20.0, 8.0]],
            color: [1.0, 1.0, 0.0],
        }];
        let walls = [WallSeg { ax: -4.0, az: -4.0, bx: 4.0, bz: -4.0, height: 3.0, color: [0.9, 0.9, 0.9] }];
        let props = [PropBox { cx: 0.0, cz: 20.0, w: 1.2, h: 2.2, d: 1.2, color: [0.92, 0.92, 0.94] }];
        let mut bad = 0;
        for dist in [20.0, 60.0, 150.0, 400.0] {
            for pitch in [0.12, 0.5, 0.85, 1.2] {
                for yaw in [0.0, 0.4, 0.8, 1.2, 1.6, 2.4, 3.1, 4.0, 5.0, 5.8] {
                    let mut cam = OrbitCamera::default();
                    cam.dist = dist;
                    cam.pitch = pitch;
                    cam.yaw = yaw;
                    let proj = cam.proj_matrix(ASPECT);
                    let view = cam.view_matrix();
                    let eye = cam.eye();
                    let mut raw: Vec<RawTri> = Vec::new();
                    let g = GROUND_SIZE / 2.0;
                    raw.extend(quad_tris(
                        Vec3::new(-g, 0.0, -g), Vec3::new(-g, 0.0, g),
                        Vec3::new(g, 0.0, g), Vec3::new(g, 0.0, -g),
                        GRASS, 1.0, LAYER_GROUND,
                    ));
                    for b in BLOCKS {
                        push_box(&mut raw, &b, eye);
                    }
                    for a in &agents {
                        push_agent(&mut raw, a);
                    }
                    for m in &markers {
                        push_marker(&mut raw, m);
                    }
                    for p in &paths {
                        push_path(&mut raw, p);
                    }
                    for p in &polys {
                        push_poly(&mut raw, p);
                    }
                    for w in &walls {
                        push_wall(&mut raw, w, eye);
                    }
                    for p in &props {
                        push_prop(&mut raw, p, eye);
                    }
                    // Real layout path: already layer + far->near sorted.
                    let tris = layout_tris(raw, &proj, &view);
                    let ndc_of = |t: &LaidTri| {
                        [
                            [t.v[0].x, t.v[0].y, t.v[0].z],
                            [t.v[1].x, t.v[1].y, t.v[1].z],
                            [t.v[2].x, t.v[2].y, t.v[2].z],
                        ]
                    };
                    // Depth at pixel via perspective-correct 1/z interpolation.
                    // Pixels near a tri edge are boundary ties (sub-pixel
                    // shimmer); only interior-interior contradictions are
                    // visible pops, so edges are skipped.
                    let z_at = |t: &LaidTri, p: [f32; 2]| -> Option<f32> {
                        let l = bary2(&ndc_of(t), p)?;
                        if l.iter().any(|x| *x < 0.02 || *x > 0.98) {
                            return None;
                        }
                        let inv = l[0] / t.vd[0] + l[1] / t.vd[1] + l[2] / t.vd[2];
                        if inv <= 0.0 {
                            return None;
                        }
                        Some(1.0 / inv)
                    };
                    for (oi, a) in tris.iter().enumerate() {
                        for b in &tris[oi + 1..] {
                            if a.layer != b.layer || a.c == b.c {
                                continue; // layers decide; same color is invisible
                            }
                            // Candidate pixels: tri centroids inside the other tri.
                            // Centroids are interior by construction, so any
                            // contradiction here is a visible pop, not edge
                            // shimmer. Off-screen overlap is GPU-clipped.
                            let centroid = |t: &LaidTri| {
                                [
                                    (t.v[0].x + t.v[1].x + t.v[2].x) / 3.0,
                                    (t.v[0].y + t.v[1].y + t.v[2].y) / 3.0,
                                ]
                            };
                            let onscreen = |p: [f32; 2]| p[0].abs() <= 1.0 && p[1].abs() <= 1.0;
                            let mut pts = Vec::new();
                            let (ca, cb) = (centroid(a), centroid(b));
                            if onscreen(ca) && z_at(b, ca).is_some() {
                                pts.push(ca);
                            }
                            if onscreen(cb) && z_at(a, cb).is_some() {
                                pts.push(cb);
                            }
                            for p in pts {
                                let (Some(za), Some(zb)) = (z_at(a, p), z_at(b, p)) else {
                                    continue; // edge pixel: boundary tie, not a pop
                                };
                                // `a` drawn first => must be farther at every shared pixel.
                                if za < zb - (zb * 0.001 + 1e-4) {
                                    println!(
                                        "INVERSION dist={dist} pitch={pitch:.2} yaw={yaw:.2} \
                                         a(col={:?},mean={:.1}) over b(col={:?},mean={:.1}) \
                                         pixel=({:.3},{:.3}) za={:.2} zb={:.2}",
                                        a.c[0], a.depth, b.c[0], b.depth, p[0], p[1], za, zb
                                    );
                                    bad += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(bad, 0, "{bad} painter inversions found (see output above)");
    }

    #[test]
    fn agents_and_markers_add_tris() {
        use crate::camera::OrbitCamera;
        let cam = OrbitCamera::default();
        let (_, base) = build_sorted_tris(&cam, ASPECT, &[], &[], &[], &[], &[], &[]);
        let agents = [AgentMarker { x: 0.0, z: 0.0, outfit: [1.0, 0.0, 0.0], skin: [0.9, 0.8, 0.7], accent: [1.0, 1.0, 0.0] }];
        let markers = [GroundMarker { x: 5.0, z: 5.0, color: [0.0, 0.0, 1.0], size: 1.2 }];
        let paths = [PathLine { points: vec![[-8.0, 0.0], [0.0, 0.0], [8.0, 8.0]], width: 2.0, color: [1.0, 1.0, 1.0] }];
        let polys = [PathPoly { points: vec![[20.0, 0.0], [28.0, 0.0], [28.0, 8.0], [20.0, 8.0]], color: [1.0, 1.0, 0.0] }];
        let walls = [WallSeg { ax: -4.0, az: -4.0, bx: 4.0, bz: -4.0, height: 3.0, color: [0.9, 0.9, 0.9] }];
        let props = [PropBox { cx: 0.0, cz: 20.0, w: 1.2, h: 2.2, d: 1.2, color: [0.92, 0.92, 0.94] }];
        let (floats, n) = build_sorted_tris(&cam, ASPECT, &agents, &markers, &paths, &polys, &walls, &props);
        // avatar (8 body + 8 head + 2 badge) + diamond (2) + ribbon (2 segments x 2) + fan (2)
        // + wall (8-long -> 4 chunks x (side + cap) + 1 facing end = 18)
        // + prop box (top + 2 facing sides = 6).
        assert_eq!(n, base + 50);
        assert_eq!(floats.len(), n * 18);
        assert!(floats.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn built_tris_are_sane_and_sorted() {
        use crate::camera::OrbitCamera;
        let cam = OrbitCamera::default();
        let (floats, n) = build_sorted_tris(&cam, ASPECT, &[], &[], &[], &[], &[], &[]);
        assert!(n > 100, "ground + grid + boxes, got {n} tris");
        assert_eq!(floats.len(), n * 18);
        assert!(floats.iter().all(|f| f.is_finite()));
        for t in floats.chunks_exact(18) {
            for v in [0, 1, 2] {
                assert!(
                    (0.0..=1.0).contains(&t[v * 6 + 2]),
                    "depth {}",
                    t[v * 6 + 2]
                );
            }
        }
        // All starter boxes visible at the default camera.
        let vp = cam.view_proj(ASPECT);
        for b in BLOCKS {
            let clip = vp * glam::Vec3::new(b.cx, b.h, b.cz).extend(1.0);
            assert!(clip.w > 0.0, "{} behind camera", b.id);
            let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);
            assert!(
                ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0,
                "{} off screen: {ndc:?}",
                b.id
            );
        }
    }

    #[test]
    fn straddling_triangle_clips_instead_of_vanishing() {
        use crate::camera::OrbitCamera;
        let cam = OrbitCamera::default();
        let view = cam.view_matrix();
        let eye = cam.eye();
        let fwd = (cam.target - eye).normalize();
        // One vertex behind the camera, two far ahead: the old
        // whole-triangle drop would delete this entirely.
        let behind = eye - fwd * 5.0;
        let ahead_l = cam.target + Vec3::new(-30.0, 0.0, 0.0);
        let ahead_r = cam.target + Vec3::new(30.0, 0.0, 0.0);
        let clipped = clip_near(&[behind, ahead_l, ahead_r], &view);
        assert!(!clipped.is_empty(), "visible part must survive");
        assert!(clipped.len() <= 2);
        for tri in &clipped {
            for v in tri {
                assert!(v.z <= -NEAR, "clipped vert in front of near plane: {v:?}");
            }
        }
    }

    #[test]
    fn fully_behind_triangle_clips_to_nothing() {
        use crate::camera::OrbitCamera;
        let cam = OrbitCamera::default();
        let view = cam.view_matrix();
        let eye = cam.eye();
        let fwd = (cam.target - eye).normalize();
        let tri = [eye - fwd * 5.0, eye - fwd * 10.0, eye - fwd * 8.0 + Vec3::X * 3.0];
        assert!(clip_near(&tri, &view).is_empty());
    }

    #[test]
    fn whole_triangle_drop_would_delete_visible_ground() {
        use crate::camera::OrbitCamera;
        // Mechanism proof for the flicker bug: a ground-half-sized
        // triangle at a zoomed-close, near-horizontal camera straddles
        // the camera plane. The old predicate (drop when ANY vertex has
        // clip `w <= 0.05`) deletes it wholesale; near-plane clipping
        // keeps the visible part.
        let mut cam = OrbitCamera::default();
        cam.dist = 15.0;
        cam.pitch = 0.12;
        cam.yaw = 0.0;
        let vp = cam.view_proj(ASPECT);
        let view = cam.view_matrix();
        let g = GROUND_SIZE / 2.0;
        let tri = [
            Vec3::new(-g, 0.0, -g),
            Vec3::new(-g, 0.0, g),
            Vec3::new(g, 0.0, g),
        ];
        let ws: Vec<f32> = tri
            .iter()
            .map(|p| (vp * p.extend(1.0)).w)
            .collect();
        assert!(
            ws.iter().any(|w| *w <= 0.05) && ws.iter().any(|w| *w > 0.05),
            "test setup must straddle: {ws:?}"
        );
        // ... so the old code dropped this tri ...
        assert!(ws.iter().any(|w| *w <= 0.05));
        // ... while clipping preserves its visible part.
        let kept = clip_near(&tri, &view);
        assert!(!kept.is_empty(), "visible ground must survive, ws={ws:?}");
        assert!(kept.len() <= 2);
    }

    #[test]
    fn low_grazing_camera_keeps_all_ground_tris() {
        use crate::camera::OrbitCamera;
        // Flicker configs: zoomed close, near-horizontal view — ground
        // corners and grid-strip ends swing behind the camera here.
        // Counts pinned post-fix (whole-triangle dropping gave far fewer);
        // all output must stay finite so the divide never explodes.
        for (dist, pitch, yaw, pinned) in [
            (15.0, 0.12, 0.0, 129),
            (15.0, 0.12, 2.1, 153),
            (30.0, 0.12, 4.0, 169),
            (25.0, 0.3, 1.0, 170),
        ] {
            let mut cam = OrbitCamera::default();
            cam.dist = dist;
            cam.pitch = pitch;
            cam.yaw = yaw;
            let (floats, n) = build_sorted_tris(&cam, ASPECT, &[], &[], &[], &[], &[], &[]);
            assert!(floats.iter().all(|f| f.is_finite()), "dist {dist} yaw {yaw}: non-finite NDC");
            assert_eq!(n, pinned, "dist {dist} yaw {yaw}: clipping changed");
        }
    }
}
