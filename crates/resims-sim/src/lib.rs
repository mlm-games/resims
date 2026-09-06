//! resims headless simulation.
//!
//! Small agents with decaying needs, points of interest with motive
//! advertisements, and a tiny state machine (idle / walk / seek-goal),
//! implemented as plain [`bevy_ecs`] components and systems so the sim
//! stays renderer-free. Idle agents score every goal — advertised
//! restoration weighted by low need, attenuated by distance — and head
//! for the best one above [`MIN_SCORE`].

use bevy_ecs::prelude::*;
use geo::{Coord, Intersects, LineString, Rect};
use pathfinding::directed::dijkstra::dijkstra;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Re-exported so UI crates can hold agent identities without depending
/// on `bevy_ecs` directly.
pub use bevy_ecs::prelude::Entity;

mod save;
pub use save::{SaveFile, SAVE_VERSION};

/// Where something is on the ground plane (x right, z towards viewer).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Position {
    pub x: f32,
    pub z: f32,
}

/// What an agent is currently doing.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentState {
    Idle,
    Walk,
    SeekGoal,
    Socialize,
    Working,
    Sleeping,
}

/// A job site: agents employed here commute during the shift and earn
/// `pay_per_hour` into household [`Funds`] while on site.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Workplace {
    pub x: f32,
    pub z: f32,
    pub pay_per_hour: f32,
    /// Shift start (inclusive) and end (exclusive), hours 0..23.
    /// Overnight shifts (start > end) wrap past midnight.
    pub shift_start: u8,
    pub shift_end: u8,
}

/// Employment link. Survives orders (player directs, work resumes
/// after); removed by [`Sim::dismiss`].
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Employment {
    pub workplace: Entity,
}

/// Work-directed commute marker (commute = SeekGoal + this). Removed by
/// player orders and shift-end cleanup, so it always means work.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Commute(pub Entity);

/// A home with beds, spawned by residential stamps. Agents move in;
/// nights are spent here (full doze) instead of rough-sleeping outside.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Dwelling {
    pub x: f32,
    pub z: f32,
    pub capacity: u32,
    pub taken: u32,
}

/// Home link. Dropped on load (entity refs don't survive snapshots).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Home(pub Entity);

/// Night commute marker (going home = SeekGoal + this). Removed by
/// player orders and on arrival, so it always means night rest.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct GoingHome(pub Entity);

/// Integer-cents money with a sub-cent accumulator. Per-tick earnings
/// are fractions of a cent (e.g. $36/h at 10Hz = 0.1¢/tick), so the
/// fraction carries until it makes a whole cent — no float drift, no
/// lost pennies.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Money {
    pub cents: i64,
    frac: f32,
}

impl Money {
    pub fn accrue(&mut self, dollars: f32) {
        let total = self.frac + dollars * 100.0;
        let whole = total.floor();
        self.cents += whole as i64;
        self.frac = total - whole;
    }
}

/// "$d.cc" for integer cents (handles negatives via euclid ops).
pub fn fmt_cents(cents: i64) -> String {
    format!("${}.{:02}", cents.div_euclid(100), cents.rem_euclid(100))
}

/// Placeable furniture: behaves as a goal with kind-fixed ads (a
/// fridge IS a hunger goal). Room-containment rules come later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FurnitureKind {
    Fridge,
    Bed,
    Sofa,
    Toilet,
    Tub,
    TV,
}

pub fn furniture_ads(kind: FurnitureKind) -> Vec<(NeedKind, f32)> {
    match kind {
        FurnitureKind::Fridge => vec![(NeedKind::Hunger, 0.7)],
        FurnitureKind::Bed => vec![(NeedKind::Energy, 0.8)],
        FurnitureKind::Sofa => vec![(NeedKind::Sociability, 0.6)],
        FurnitureKind::Toilet => vec![(NeedKind::Bladder, 0.8)],
        FurnitureKind::Tub => vec![(NeedKind::Hygiene, 0.8)],
        FurnitureKind::TV => vec![(NeedKind::Fun, 0.7)],
    }
}

/// Furniture marker on the goal entity (lets UI/render distinguish
/// furniture from building-attached goals).
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Furniture(pub FurnitureKind);

/// Room function voted from furniture contents (bedroom/kitchen/
/// living); empty or tied rooms stay unassigned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoomFunction {
    Bedroom,
    Kitchen,
    Living,
    Bathroom,
}

fn room_function_for(kind: FurnitureKind) -> RoomFunction {
    match kind {
        FurnitureKind::Bed => RoomFunction::Bedroom,
        FurnitureKind::Fridge => RoomFunction::Kitchen,
        FurnitureKind::Sofa | FurnitureKind::TV => RoomFunction::Living,
        FurnitureKind::Toilet | FurnitureKind::Tub => RoomFunction::Bathroom,
    }
}

/// Household money, earned by [`AgentState::Working`] agents.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Funds(pub Money);
/// City treasury, filled by the nightly tax tick (Micropolis budget in
/// miniature: rate x capacity x development per building per day).
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Treasury(pub Money);

/// Day counter, incremented when the clock wraps past midnight.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DayCount(pub u32);

/// Tax funds per capacity per day at full development.
pub const TAX_PER_CAP_DAY: f32 = 5.0;

/// Zoning function stamped onto a building by Implement (Micropolis
/// zones in miniature: function + development level, no tile grid).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZoneFunction {
    Residential,
    Commercial,
    Industrial,
    Agricultural,
    Recreational,
    Administrative,
}

/// A city building with a stamped zone function. Residential = future
/// homes (capacity), Commercial/Industrial = job sites, the rest =
/// amenity goal sites. Behaviour flows through spawned [`Goal`] /
/// [`Workplace`] entities; this holds the city-level state (function,
/// development, capacity) for stats and growth.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct CityBuilding {
    /// Links back to the UI block id (e.g. "bld:3").
    pub label: String,
    pub function: ZoneFunction,
    pub min_x: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_z: f32,
    /// Development 0..=1: grows with road access, decays without.
    pub development: f32,
    /// Residents or jobs at full development (from footprint area).
    pub capacity: u32,
}

impl CityBuilding {
    pub fn new(
        label: String,
        function: ZoneFunction,
        min_x: f32,
        min_z: f32,
        max_x: f32,
        max_z: f32,
    ) -> Self {
        let area = (max_x - min_x).max(0.0) * (max_z - min_z).max(0.0);
        let capacity = (area / 25.0).floor().max(1.0) as u32;
        Self {
            label,
            function,
            min_x,
            min_z,
            max_x,
            max_z,
            development: 0.0,
            capacity,
        }
    }

    pub fn center(&self) -> [f32; 2] {
        [(self.min_x + self.max_x) / 2.0, (self.min_z + self.max_z) / 2.0]
    }
}

/// A wall segment (build mode): thin impassable barrier. Grid cells
/// whose rect a segment truly crosses are blocked for routing.
/// Door gaps need one full free cell row (2u at default cell size):
/// wall endpoints pull 1mm inside so shared corners don't bleed, but a
/// gap straddling two crossed rows stays sealed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Wall {
    pub ax: f32,
    pub az: f32,
    pub bx: f32,
    pub bz: f32,
}

/// Stamped/built wall network.
#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Walls(pub Vec<Wall>);

/// Stamped road network (polylines). Buildings within
/// [`ROAD_ACCESS_DIST`] of any segment develop; isolated ones decay.
/// Road cells are also ~3x cheaper to walk, so agents prefer roads.
#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Roads(pub Vec<Vec<[f32; 2]>>);

/// Road-access radius, development growth and vacancy decay per sim-second.
pub const ROAD_ACCESS_DIST: f32 = 6.0;
pub const DEVELOP_RATE: f32 = 0.02;
pub const NEGLECT_RATE: f32 = 0.005;

/// Distance from point to segment (for road access).
fn point_seg_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let abx = b[0] - a[0];
    let abz = b[1] - a[1];
    let len2 = abx * abx + abz * abz;
    let t = if len2 <= 0.0 {
        0.0
    } else {
        (((p[0] - a[0]) * abx + (p[1] - a[1]) * abz) / len2).clamp(0.0, 1.0)
    };
    let dx = p[0] - (a[0] + abx * t);
    let dz = p[1] - (a[1] + abz * t);
    (dx * dx + dz * dz).sqrt()
}

/// Zone development (Micropolis doRes/doCom/doInd in miniature):
/// buildings with road access develop toward 1, isolated ones decay.
fn develop_buildings(world: &mut World, dt: f32) {
    let roads = world.get_resource::<Roads>().map(|r| r.0.clone()).unwrap_or_default();
    let mut query = world.query::<&mut CityBuilding>();
    for mut b in query.iter_mut(world) {
        let probes = [
            b.center(),
            [b.min_x, b.min_z],
            [b.max_x, b.min_z],
            [b.min_x, b.max_z],
            [b.max_x, b.max_z],
        ];
        let mut access = false;
        'outer: for p in probes {
            for line in &roads {
                for w in line.windows(2) {
                    if point_seg_dist(p, w[0], w[1]) <= ROAD_ACCESS_DIST {
                        access = true;
                        break 'outer;
                    }
                }
            }
        }
        if access {
            b.development = (b.development + DEVELOP_RATE * dt).min(1.0);
        } else {
            b.development = (b.development - NEGLECT_RATE * dt).max(0.0);
        }
    }
}

/// True while `hour` is inside [start, end) (wraps past midnight).
fn in_shift(hour: u8, start: u8, end: u8) -> bool {
    if start <= end {
        hour >= start && hour < end
    } else {
        hour >= start || hour < end
    }
}

/// Simple decaying needs in 0..=1. Autonomy scores the lowest ones.
#[derive(Component, Clone, Debug)]
pub struct Needs {
    pub hunger: f32,
    pub energy: f32,
    pub sociability: f32,
    pub comfort: f32,
    pub hygiene: f32,
    pub bladder: f32,
    pub fun: f32,
}

impl Needs {
    pub fn value(&self, kind: NeedKind) -> f32 {
        match kind {
            NeedKind::Hunger => self.hunger,
            NeedKind::Energy => self.energy,
            NeedKind::Sociability => self.sociability,
            NeedKind::Comfort => self.comfort,
            NeedKind::Hygiene => self.hygiene,
            NeedKind::Bladder => self.bladder,
            NeedKind::Fun => self.fun,
        }
    }

    /// Restore a need by `amount`, capped at 1.0.
    pub fn restore(&mut self, kind: NeedKind, amount: f32) {
        let slot = match kind {
            NeedKind::Hunger => &mut self.hunger,
            NeedKind::Energy => &mut self.energy,
            NeedKind::Sociability => &mut self.sociability,
            NeedKind::Comfort => &mut self.comfort,
            NeedKind::Hygiene => &mut self.hygiene,
            NeedKind::Bladder => &mut self.bladder,
            NeedKind::Fun => &mut self.fun,
        };
        *slot = (*slot + amount).min(1.0);
    }

    /// Lowest need (autonomy/chat urgency) and mean level (happiness).
    pub fn lowest(&self) -> f32 {
        self.hunger
            .min(self.energy)
            .min(self.sociability)
            .min(self.comfort)
            .min(self.hygiene)
            .min(self.bladder)
            .min(self.fun)
    }

    pub fn mean(&self) -> f32 {
        (self.hunger
            + self.energy
            + self.sociability
            + self.comfort
            + self.hygiene
            + self.bladder
            + self.fun)
            / 7.0
    }
}

impl Default for Needs {
    fn default() -> Self {
        Self {
            hunger: 0.8,
            energy: 0.8,
            sociability: 0.8,
            comfort: 0.8,
            hygiene: 0.8,
            bladder: 0.8,
            fun: 0.8,
        }
    }
}

/// Personality traits 0..=1, randomized at spawn. Each scales the
/// autonomy gain of one need (free-will multipliers in miniature):
/// playful->Fun, outgoing->Sociability, active->Energy, as 0.5+t.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Personality {
    pub playful: f32,
    pub outgoing: f32,
    pub active: f32,
}

impl Personality {
    /// Multiplier for a need's scored gain.
    pub fn factor(&self, kind: NeedKind) -> f32 {
        match kind {
            NeedKind::Fun => 0.5 + self.playful,
            NeedKind::Sociability => 0.5 + self.outgoing,
            NeedKind::Energy => 0.5 + self.active,
            _ => 1.0,
        }
    }
}

/// Learnable skills 0..=1, all start at 0 and grow +0.02 per relevant
/// arrival (cap 1). Cooking scales hunger restoration, charisma scales
/// sociability restoration.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct Skills {
    pub cooking: f32,
    pub charisma: f32,
}

impl Skills {
    /// Restoration multiplier for a need.
    pub fn factor(&self, kind: NeedKind) -> f32 {
        match kind {
            NeedKind::Hunger => 1.0 + self.cooking,
            NeedKind::Sociability => 1.0 + self.charisma,
            _ => 1.0,
        }
    }

    /// Learn from a used ad (called on arrival per ad kind).
    pub fn learn(&mut self, kind: NeedKind) {
        match kind {
            NeedKind::Hunger => self.cooking = (self.cooking + 0.02).min(1.0),
            NeedKind::Sociability => self.charisma = (self.charisma + 0.02).min(1.0),
            _ => {}
        }
    }
}

/// A point of interest agents can use (food stall, bench, home...).
///
/// `ads` are motive advertisements: (need, amount restored) pairs scored
/// by the autonomy system. `attenuation` discounts distant goals: the
/// score is divided by `1 + attenuation * distance`.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Goal {
    pub x: f32,
    pub z: f32,
    pub ads: Vec<(NeedKind, f32)>,
    pub attenuation: f32,
    /// Agent currently walking to this goal (one agent per goal).
    pub claimed_by: Option<Entity>,
}

impl Goal {
    /// Dominant advertised need (for markers); Hunger if ad-less.
    pub fn primary(&self) -> NeedKind {
        let mut best = NeedKind::Hunger;
        let mut best_amt = f32::NEG_INFINITY;
        for (kind, amount) in &self.ads {
            if *amount > best_amt {
                best_amt = *amount;
                best = *kind;
            }
        }
        best
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeedKind {
    Hunger,
    Energy,
    Sociability,
    Comfort,
    Hygiene,
    Bladder,
    Fun,
}

/// Walk target assigned by the behaviour system or the player.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct WalkTarget {
    pub x: f32,
    pub z: f32,
    /// Goal object claim held while walking (one agent per goal).
    pub claim: Option<Entity>,
}

/// A player-issued order. Player actions preempt autonomy and are
/// worked through front to back; autonomy only fills idle agents with
/// an empty queue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActionKind {
    /// Walk to a ground point.
    Goto { x: f32, z: f32 },
    /// Walk to a goal object and use it.
    Use { goal: Entity },
}

#[derive(Component, Clone, Debug, Default)]
pub struct ActionQueue(pub VecDeque<ActionKind>);

/// Axis-aligned blocked rect on the ground plane (buildings).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Obstacle {
    pub min_x: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_z: f32,
}

/// Routing grid + obstacles. Agents BFS around `rects`; unreachable
/// targets fail (agent stays idle) instead of walking through walls.
#[derive(Resource, Clone, Debug)]
pub struct Obstacles {
    pub rects: Vec<Obstacle>,
    /// World half-extent covered by the grid.
    pub half_extent: f32,
    /// Cell size in world units.
    pub cell: f32,
}

impl Default for Obstacles {
    fn default() -> Self {
        Self {
            rects: Vec::new(),
            half_extent: 64.0,
            cell: 2.0,
        }
    }
}

/// Remaining waypoints to the [`WalkTarget`] (empty = straight walk).
#[derive(Component, Clone, Debug, PartialEq, Default)]
pub struct Route(pub Vec<[f32; 2]>);

/// Snap radius (cells) for targets inside obstacles.
const SNAP_RADIUS_CELLS: i32 = 4;
/// Routing costs x10 (integer Dijkstra): road cells are ~3x cheaper, so
/// agents detour onto stamped roads instead of cutting across grass.
const GRASS_COST: u32 = 10;
const ROAD_COST: u32 = 3;
/// Cell centers within this of a road segment count as road.
const ROAD_HALF_WIDTH: f32 = 2.0;

/// Snapshot format version. Bump when [`SimSnapshot`] changes shape;
/// [`Sim::load_json`] rejects anything else.
pub const SNAPSHOT_VERSION: u32 = 1;

/// Serializable sim snapshot. Entity identities are NOT preserved
/// (claims/queues/targets reference live entities); see [`Sim::restore`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SimSnapshot {
    pub version: u32,
    pub agents: Vec<AgentSnapshot>,
    pub goals: Vec<GoalSnapshot>,
    pub workplaces: Vec<WorkplaceSnapshot>,
    pub buildings: Vec<BuildingSnapshot>,
    pub dwellings: Vec<DwellingSnapshot>,
    pub obstacles: Vec<Obstacle>,
    pub roads: Vec<Vec<[f32; 2]>>,
    pub walls: Vec<Wall>,
    pub hour: u8,
    pub day: u32,
    pub funds: i64,
    pub treasury: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentSnapshot {
    pub x: f32,
    pub z: f32,
    pub hunger: f32,
    pub energy: f32,
    pub sociability: f32,
    pub comfort: f32,
    pub hygiene: f32,
    pub bladder: f32,
    pub fun: f32,
    pub playful: f32,
    pub outgoing: f32,
    pub active: f32,
    pub cooking: f32,
    pub charisma: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GoalSnapshot {
    pub x: f32,
    pub z: f32,
    pub ads: Vec<(NeedKind, f32)>,
    pub attenuation: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkplaceSnapshot {
    pub x: f32,
    pub z: f32,
    pub pay_per_hour: f32,
    pub shift_start: u8,
    pub shift_end: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BuildingSnapshot {
    pub label: String,
    pub function: ZoneFunction,
    pub min_x: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_z: f32,
    pub development: f32,
    pub capacity: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DwellingSnapshot {
    pub x: f32,
    pub z: f32,
    pub capacity: u32,
    pub taken: u32,
}

/// A detected room: an enclosed walkable region (walled area, courtyard
/// against a building, ...). Centroid + cell count + bounds; function
/// assignment (bedroom/kitchen/...) comes with furniture.
#[derive(Clone, Debug, PartialEq)]
pub struct Room {
    pub cx: f32,
    pub cz: f32,
    pub cells: u32,
    pub min_x: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_z: f32,
}

/// Detected rooms, rebuilt by [`Sim::rebuild_rooms`] (call after wall
/// edits; also rebuilt on snapshot restore). Derived from walls, so
/// snapshots don't store it.
#[derive(Resource, Clone, Debug, Default)]
pub struct Rooms(pub Vec<Room>);

/// Grid size shared by routing and room detection.
fn grid_n(obs: &Obstacles) -> i32 {
    (obs.half_extent * 2.0 / obs.cell).ceil() as i32
}

/// Blocked-cell grid: obstacle-rect centers plus any cell whose rect a
/// wall segment truly crosses. Segment endpoints pull a millimetre
/// inside so shared corners and door jambs don't bleed collision into
/// neighbour cells (plain geo intersect otherwise). Returns grid + side.
fn build_blocked_grid(obs: &Obstacles, walls: &Walls) -> (Vec<bool>, i32) {
    let (cell, half) = (obs.cell, obs.half_extent);
    let n = grid_n(obs);
    let idx = |cx: i32, cy: i32| (cy * n + cx) as usize;
    const WALL_EPS: f32 = 1e-3;
    let segs: Vec<LineString<f32>> = walls
        .0
        .iter()
        .map(|w| {
            let dx = w.bx - w.ax;
            let dz = w.bz - w.az;
            let len = (dx * dx + dz * dz).sqrt().max(1e-6);
            let t = (WALL_EPS / len).min(0.5);
            LineString::from(vec![
                Coord { x: w.ax + dx * t, y: w.az + dz * t },
                Coord { x: w.bx - dx * t, y: w.bz - dz * t },
            ])
        })
        .collect();
    let mut grid = vec![false; (n * n) as usize];
    for cy in 0..n {
        for cx in 0..n {
            let wx = cx as f32 * cell + cell * 0.5 - half;
            let wz = cy as f32 * cell + cell * 0.5 - half;
            if obs.rects.iter().any(|r| {
                wx >= r.min_x && wx <= r.max_x && wz >= r.min_z && wz <= r.max_z
            }) {
                grid[idx(cx, cy)] = true;
                continue;
            }
            let cell_rect = Rect::new(
                Coord { x: wx - cell / 2.0, y: wz - cell / 2.0 },
                Coord { x: wx + cell / 2.0, y: wz + cell / 2.0 },
            );
            if segs.iter().any(|s| s.intersects(&cell_rect)) {
                grid[idx(cx, cy)] = true;
            }
        }
    }
    (grid, n)
}

/// Flood-fill room detection on the blocked grid: border-reachable
/// walkable cells are "outside"; remaining walkable components are
/// rooms (walled interiors, courtyards against buildings, ...).
fn detect_rooms(obs: &Obstacles, walls: &Walls) -> Vec<Room> {
    let (cell, half) = (obs.cell, obs.half_extent);
    let (blocked, n) = build_blocked_grid(obs, walls);
    let idx = |cx: i32, cy: i32| (cy * n + cx) as usize;
    let center = |cx: i32, cy: i32| {
        [
            cx as f32 * cell + cell * 0.5 - half,
            cy as f32 * cell + cell * 0.5 - half,
        ]
    };
    // Outside: flood walkable cells from every border cell.
    let mut outside = vec![false; (n * n) as usize];
    let mut stack: Vec<(i32, i32)> = Vec::new();
    for c in 0..n {
        for (cx, cy) in [(c, 0), (c, n - 1), (0, c), (n - 1, c)] {
            if !blocked[idx(cx, cy)] && !outside[idx(cx, cy)] {
                outside[idx(cx, cy)] = true;
                stack.push((cx, cy));
            }
        }
    }
    while let Some((cx, cy)) = stack.pop() {
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (cx + dx, cy + dy);
            if nx < 0 || ny < 0 || nx >= n || ny >= n {
                continue;
            }
            if blocked[idx(nx, ny)] || outside[idx(nx, ny)] {
                continue;
            }
            outside[idx(nx, ny)] = true;
            stack.push((nx, ny));
        }
    }
    // Interior components become rooms.
    let mut seen = vec![false; (n * n) as usize];
    let mut rooms = Vec::new();
    for cy in 0..n {
        for cx in 0..n {
            if blocked[idx(cx, cy)] || outside[idx(cx, cy)] || seen[idx(cx, cy)] {
                continue;
            }
            let (mut sx, mut sz, mut count) = (0.0, 0.0, 0u32);
            let (mut min_x, mut min_z) = (f32::INFINITY, f32::INFINITY);
            let (mut max_x, mut max_z) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            let mut work = vec![(cx, cy)];
            seen[idx(cx, cy)] = true;
            while let Some((qx, qy)) = work.pop() {
                let [wx, wz] = center(qx, qy);
                sx += wx;
                sz += wz;
                count += 1;
                min_x = min_x.min(wx - cell / 2.0);
                min_z = min_z.min(wz - cell / 2.0);
                max_x = max_x.max(wx + cell / 2.0);
                max_z = max_z.max(wz + cell / 2.0);
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (qx + dx, qy + dy);
                    if nx < 0 || ny < 0 || nx >= n || ny >= n {
                        continue;
                    }
                    if blocked[idx(nx, ny)] || outside[idx(nx, ny)] || seen[idx(nx, ny)] {
                        continue;
                    }
                    seen[idx(nx, ny)] = true;
                    work.push((nx, ny));
                }
            }
            rooms.push(Room {
                cx: sx / count as f32,
                cz: sz / count as f32,
                cells: count,
                min_x,
                min_z,
                max_x,
                max_z,
            });
        }
    }
    rooms
}

/// Dijkstra route on the obstacle grid with a road cost map. `None` =
/// unreachable (fail like `VMRouteFailCode`, caller leaves the agent
/// idle). Empty vec = walk straight (no obstacles and no roads, or
/// start and target share a cell).
fn plan_route(
    obs: &Obstacles,
    roads: &Roads,
    walls: &Walls,
    from: [f32; 2],
    to: [f32; 2],
) -> Option<Vec<[f32; 2]>> {
    if obs.rects.is_empty() && roads.0.is_empty() && walls.0.is_empty() {
        return Some(Vec::new());
    }
    let (cell, half) = (obs.cell, obs.half_extent);
    let n = (half * 2.0 / cell).ceil() as i32;
    let to_world = |cx: i32, cy: i32| {
        [
            cx as f32 * cell + cell * 0.5 - half,
            cy as f32 * cell + cell * 0.5 - half,
        ]
    };
    let to_cell = |p: [f32; 2]| {
        [
            ((p[0] + half) / cell).floor() as i32,
            ((p[1] + half) / cell).floor() as i32,
        ]
    };
    let in_bounds = |cx: i32, cy: i32| cx >= 0 && cy >= 0 && cx < n && cy < n;
    let idx = |cx: i32, cy: i32| (cy * n + cx) as usize;
    // Blocked grid shared with room detection (obstacles + walls).
    let (blocked_grid, _) = build_blocked_grid(obs, walls);
    let blocked = |cx: i32, cy: i32| blocked_grid[idx(cx, cy)];
    let start = to_cell(from);
    if !in_bounds(start[0], start[1]) {
        return Some(Vec::new()); // off-grid: straight fallback
    }
    // Target inside an obstacle: snap to nearest walkable cell.
    let mut target = to_cell(to);
    if !in_bounds(target[0], target[1]) || blocked(target[0], target[1]) {
        let mut snapped = None;
        'snap: for r in 1..=SNAP_RADIUS_CELLS {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs() != r && dy.abs() != r {
                        continue;
                    }
                    let (cx, cy) = (target[0] + dx, target[1] + dy);
                    if in_bounds(cx, cy) && !blocked(cx, cy) {
                        snapped = Some([cx, cy]);
                        break 'snap;
                    }
                }
            }
        }
        target = match snapped {
            Some(c) => c,
            None => return None,
        };
    }
    if start == target || blocked(start[0], start[1]) {
        return Some(Vec::new()); // same cell, or start inside: straight
    }
    // Road mask: cell centers near any stamped segment ride cheap.
    let mut road_mask = vec![false; (n * n) as usize];
    for cy in 0..n {
        for cx in 0..n {
            let [wx, wz] = to_world(cx, cy);
            'segs: for line in &roads.0 {
                for w in line.windows(2) {
                    let dx = wx - w[0][0];
                    let dz = wz - w[0][1];
                    // Segment distance, squared-side first for the cheap out.
                    let abx = w[1][0] - w[0][0];
                    let abz = w[1][1] - w[0][1];
                    let len2 = abx * abx + abz * abz;
                    let t = if len2 <= 0.0 {
                        0.0
                    } else {
                        ((dx * abx + dz * abz) / len2).clamp(0.0, 1.0)
                    };
                    let ex = wx - (w[0][0] + abx * t);
                    let ez = wz - (w[0][1] + abz * t);
                    if ex * ex + ez * ez <= ROAD_HALF_WIDTH * ROAD_HALF_WIDTH {
                        road_mask[idx(cx, cy)] = true;
                        break 'segs;
                    }
                }
            }
        }
    }
    let cells = dijkstra(
        &start,
        |[cx, cy]| {
            [(1, 0), (-1, 0), (0, 1), (0, -1)]
                .into_iter()
                .filter_map(|(dx, dy)| {
                    let (nx, ny) = (cx + dx, cy + dy);
                    if !in_bounds(nx, ny) || blocked(nx, ny) {
                        return None;
                    }
                    let step = if road_mask[idx(nx, ny)] {
                        ROAD_COST
                    } else {
                        GRASS_COST
                    };
                    Some(([nx, ny], step))
                })
                .collect::<Vec<_>>()
        },
        |c| *c == target,
    )
    .map(|(cells, _)| cells);
    let cells = match cells {
        Some(c) => c,
        None => return None,
    };
    let mut points: Vec<[f32; 2]> = cells.into_iter().map(|[cx, cy]| to_world(cx, cy)).collect();
    points.remove(0); // skip start cell
    points.push(to); // exact target last
    Some(points)
}

/// Needs at or above this are in the flat part of the curve: ads for
/// them score zero, so satisfied agents stay idle.
pub const SATISFIED: f32 = 0.5;
/// Scores at or below this are ignored (float dust, far-away weak wants).
pub const MIN_SCORE: f32 = 1e-4;
/// Default distance discount for [`Goal`]s.
pub const DEFAULT_ATTENUATION: f32 = 0.05;
/// Default amount a single-need goal restores.
pub const DEFAULT_AD_AMOUNT: f32 = 0.5;

/// Contribution curve: low needs weigh quadratically more, so urgency
/// grows as a need drops (0 above [`SATISFIED`]).
fn motive_weight(v: f32) -> f32 {
    let d = ((SATISFIED - v).max(0.0)) / SATISFIED;
    d * d
}

/// Seeded RNG for autonomy decisions (tied-score tie-breaks). Same
/// seed + same ops = same city: the sim stays deterministic.
#[derive(Resource, Clone, Debug)]
pub struct SimRng(pub ChaCha8Rng);

impl Default for SimRng {
    fn default() -> Self {
        Self(ChaCha8Rng::seed_from_u64(0xC17B07))
    }
}

/// Hour of day 0..23, fed by the UI clock each frame. Drives day/night
/// effects (dozing). Sims-style nights run 22:00-06:00.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HourOfDay(pub u8);

pub fn is_night(hour: u8) -> bool {
    hour >= 22 || hour < 6
}

/// Simulation clock + tuning. Fixed-step friendly: call [`step`] with dt.
///
/// Decay is per-need: different needs drain at different rates instead
/// of one global knob.
#[derive(Resource, Clone, Debug)]
pub struct SimConfig {
    /// Hunger points lost per second.
    pub hunger_decay: f32,
    /// Energy points lost per second.
    pub energy_decay: f32,
    /// Sociability points lost per second.
    pub sociability_decay: f32,
    /// Comfort points lost per second.
    pub comfort_decay: f32,
    /// Hygiene points lost per second.
    pub hygiene_decay: f32,
    /// Bladder points lost per second.
    pub bladder_decay: f32,
    /// Fun points lost per second.
    pub fun_decay: f32,
    /// Ground units walked per second.
    pub walk_speed: f32,
    /// Distance at which a goal counts as reached.
    pub arrive_dist: f32,
    /// Energy per second regained by idle agents at night (dozing).
    pub night_restore: f32,
    /// Radius (ground units) within which idle agents chat.
    pub chat_radius: f32,
    /// Sociability per second regained while chatting.
    pub chat_restore: f32,
    /// Chatting stops when any need drops below this (free will wins).
    pub chat_break: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            hunger_decay: 0.01,
            energy_decay: 0.01,
            sociability_decay: 0.01,
            comfort_decay: 0.01,
            hygiene_decay: 0.012,
            bladder_decay: 0.015,
            fun_decay: 0.008,
            walk_speed: 4.0,
            arrive_dist: 1.5,
            night_restore: 0.02,
            chat_radius: 3.0,
            chat_restore: 0.08,
            chat_break: 0.25,
        }
    }
}

/// Minimal world: agents plus the goals they can use.
#[derive(Default)]
pub struct Sim {
    pub world: World,
}

impl Sim {
    pub fn new() -> Self {
        let mut world = World::new();
        world.init_resource::<SimConfig>();
        world.init_resource::<Obstacles>();
        world.insert_resource(HourOfDay(8)); // start in the morning
        world.insert_resource(Funds(Money::default()));
        world.insert_resource(Treasury(Money::default()));
        world.insert_resource(DayCount(0));
        world.init_resource::<SimRng>();
        world.init_resource::<Roads>();
        world.init_resource::<Walls>();
        world.init_resource::<Rooms>();
        Self { world }
    }

    pub fn spawn_agent(&mut self, x: f32, z: f32) -> Entity {
        let personality = {
            let mut rng = self.world.get_resource_mut::<SimRng>().expect("SimRng");
            Personality {
                playful: rng.0.random_range(0.0..=1.0),
                outgoing: rng.0.random_range(0.0..=1.0),
                active: rng.0.random_range(0.0..=1.0),
            }
        };
        self.world
            .spawn((
                Position { x, z },
                Needs::default(),
                AgentState::Idle,
                ActionQueue::default(),
                personality,
                Skills::default(),
            ))
            .id()
    }

    pub fn spawn_goal(&mut self, x: f32, z: f32, restores: NeedKind) -> Entity {
        self.spawn_goal_ads(x, z, vec![(restores, DEFAULT_AD_AMOUNT)], DEFAULT_ATTENUATION)
    }

    pub fn spawn_goal_ads(
        &mut self,
        x: f32,
        z: f32,
        ads: Vec<(NeedKind, f32)>,
        attenuation: f32,
    ) -> Entity {
        self.world
            .spawn((
                Goal {
                    x,
                    z,
                    ads,
                    attenuation,
                    claimed_by: None,
                },
            ))
            .id()
    }

    /// Stamp a zoned building: records function/capacity, starts at
    /// zero development (road access grows it).
    pub fn spawn_building(
        &mut self,
        label: String,
        function: ZoneFunction,
        min_x: f32,
        min_z: f32,
        max_x: f32,
        max_z: f32,
    ) -> Entity {
        self.world
            .spawn((CityBuilding::new(label, function, min_x, min_z, max_x, max_z),))
            .id()
    }

    /// Replace the stamped road network (for development access).
    pub fn set_roads(&mut self, roads: Vec<Vec<[f32; 2]>>) {
        self.world.insert_resource(Roads(roads));
    }

    /// Add a wall segment (build mode).
    pub fn add_wall(&mut self, ax: f32, az: f32, bx: f32, bz: f32) {
        if let Some(mut walls) = self.world.get_resource_mut::<Walls>() {
            walls.0.push(Wall { ax, az, bx, bz });
        }
    }

    /// Replace the whole wall network.
    pub fn set_walls(&mut self, walls: Vec<Wall>) {
        self.world.insert_resource(Walls(walls));
    }

    /// Rebuild room detection from the current walls + obstacles. Call
    /// after wall edits (wall stamping, restore).
    pub fn rebuild_rooms(&mut self) {
        let obs = self.world.resource::<Obstacles>().clone();
        let walls = self.world.resource::<Walls>().clone();
        self.world.insert_resource(Rooms(detect_rooms(&obs, &walls)));
    }

    /// Detected rooms (enclosed walkable regions).
    pub fn rooms(&self) -> Vec<Room> {
        self.world.get_resource::<Rooms>().map(|r| r.0.clone()).unwrap_or_default()
    }

    /// Room functions voted from furniture inside each room's bounds:
    /// most beds -> bedroom, most fridges -> kitchen, most sofas ->
    /// living; empty rooms and ties stay unassigned. Aligned with
    /// [`Sim::rooms`].
    pub fn room_functions(&mut self) -> Vec<Option<RoomFunction>> {
        let rooms = self.rooms();
        if rooms.is_empty() {
            return Vec::new();
        }
        let pieces: Vec<(f32, f32, RoomFunction)> = {
            let mut q = self.world.query::<(&Goal, &Furniture)>();
            q.iter(&self.world)
                .map(|(g, f)| (g.x, g.z, room_function_for(f.0)))
                .collect()
        };
        rooms
            .iter()
            .map(|r| {
                let (mut bed, mut kitchen, mut living, mut bath) = (0u32, 0u32, 0u32, 0u32);
                for (x, z, f) in &pieces {
                    if *x >= r.min_x && *x <= r.max_x && *z >= r.min_z && *z <= r.max_z {
                        match f {
                            RoomFunction::Bedroom => bed += 1,
                            RoomFunction::Kitchen => kitchen += 1,
                            RoomFunction::Living => living += 1,
                            RoomFunction::Bathroom => bath += 1,
                        }
                    }
                }
                // Strict majority wins; ties and empties unassigned.
                if bed > kitchen && bed > living && bed > bath {
                    Some(RoomFunction::Bedroom)
                } else if kitchen > bed && kitchen > living && kitchen > bath {
                    Some(RoomFunction::Kitchen)
                } else if living > bed && living > kitchen && living > bath {
                    Some(RoomFunction::Living)
                } else if bath > bed && bath > kitchen && bath > living {
                    Some(RoomFunction::Bathroom)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Clear all walls.
    pub fn clear_walls(&mut self) {
        self.set_walls(Vec::new());
    }

    pub fn spawn_workplace(
        &mut self,
        x: f32,
        z: f32,
        pay_per_hour: f32,
        shift_start: u8,
        shift_end: u8,
    ) -> Entity {
        self.world
            .spawn((Workplace { x, z, pay_per_hour, shift_start, shift_end },))
            .id()
    }

    /// Employ an agent (they finish their current errand, then commute
    /// on shift). Re-employing elsewhere resets the commute.
    pub fn employ(&mut self, agent: Entity, workplace: Entity) {
        self.release_claim(agent);
        self.world.entity_mut(agent).remove::<Commute>();
        self.world.entity_mut(agent).remove::<GoingHome>();
        self.world.entity_mut(agent).remove::<WalkTarget>();
        self.world.entity_mut(agent).remove::<Route>();
        self.world.entity_mut(agent).insert(AgentState::Idle);
        self.world.entity_mut(agent).insert(Employment { workplace });
    }

    /// Dismiss: employment ends immediately, activity cleared.
    pub fn dismiss(&mut self, agent: Entity) {
        self.world.entity_mut(agent).remove::<Employment>();
        self.world.entity_mut(agent).remove::<Commute>();
        self.world.entity_mut(agent).remove::<GoingHome>();
        self.world.entity_mut(agent).remove::<WalkTarget>();
        self.world.entity_mut(agent).remove::<Route>();
        self.world.entity_mut(agent).insert(AgentState::Idle);
    }

    pub fn spawn_dwelling(&mut self, x: f32, z: f32, capacity: u32) -> Entity {
        self.world.spawn((Dwelling { x, z, capacity, taken: 0 },)).id()
    }

    /// Place furniture: a goal with kind-fixed ads plus a marker.
    pub fn spawn_furniture(&mut self, x: f32, z: f32, kind: FurnitureKind) -> Entity {
        let e = self.spawn_goal_ads(x, z, furniture_ads(kind), DEFAULT_ATTENUATION);
        self.world.entity_mut(e).insert(Furniture(kind));
        e
    }

    /// Move in: releases the old bed, claims one here. False when full
    /// (or the dwelling is gone) — the old home is kept then.
    pub fn move_in(&mut self, agent: Entity, dwelling: Entity) -> bool {
        if let Some(home) = self.world.get::<Home>(agent).map(|h| h.0) {
            if home == dwelling {
                return true;
            }
        }
        let free = self.world.get::<Dwelling>(dwelling).is_some_and(|d| d.taken < d.capacity);
        if !free {
            return false;
        }
        self.move_out(agent);
        if let Some(mut d) = self.world.get_mut::<Dwelling>(dwelling) {
            d.taken += 1;
        }
        self.world.entity_mut(agent).insert(Home(dwelling));
        true
    }

    /// Move out: frees the bed, clears night commuting.
    pub fn move_out(&mut self, agent: Entity) {
        if let Some(home) = self.world.get::<Home>(agent).map(|h| h.0) {
            if let Some(mut d) = self.world.get_mut::<Dwelling>(home) {
                d.taken = d.taken.saturating_sub(1);
            }
            self.world.entity_mut(agent).remove::<Home>();
        }
        self.world.entity_mut(agent).remove::<GoingHome>();
    }

    pub fn funds(&self) -> i64 {
        self.world.get_resource::<Funds>().map(|f| f.0.cents).unwrap_or(0)
    }

    /// Drop any goal claim held by `agent` (preempt / arrival).
    fn release_claim(&mut self, agent: Entity) {
        let claim = self.world.get::<WalkTarget>(agent).and_then(|t| t.claim);
        if let Some(goal) = claim {
            if let Some(mut g) = self.world.get_mut::<Goal>(goal) {
                if g.claimed_by == Some(agent) {
                    g.claimed_by = None;
                }
            }
        }
    }

    /// Player order: drop current activity (releasing any goal claim,
    /// taking them off-shift and off-to-bed for now — employment and
    /// home persist) and queue the action at the front.
    pub fn order(&mut self, agent: Entity, action: ActionKind) {
        self.release_claim(agent);
        self.world.entity_mut(agent).remove::<Commute>();
        self.world.entity_mut(agent).remove::<GoingHome>();
        self.world.entity_mut(agent).remove::<WalkTarget>();
        self.world.entity_mut(agent).remove::<Route>();
        self.world.entity_mut(agent).insert(AgentState::Idle);
        if let Some(mut q) = self.world.get_mut::<ActionQueue>(agent) {
            q.0.push_front(action);
        }
    }

    /// Reseed autonomy decisions (deterministic replays, map variants).
    pub fn set_seed(&mut self, seed: u64) {
        self.world.insert_resource(SimRng(ChaCha8Rng::seed_from_u64(seed)));
    }

    /// Set the hour of day (wraps into 0..23). The UI clock feeds this
    /// every frame; night hours let idle agents doze (regain energy).
    /// A wrap past midnight advances the day and collects taxes.
    pub fn set_hour(&mut self, hour: u8) {
        let hour = hour % 24;
        let old = self.world.get_resource::<HourOfDay>().map(|h| h.0).unwrap_or(hour);
        self.world.insert_resource(HourOfDay(hour));
        if hour < old {
            if let Some(mut d) = self.world.get_resource_mut::<DayCount>() {
                d.0 += 1;
            }
            self.collect_taxes();
        }
    }

    /// Nightly tax tick: every developed building pays rate x capacity
    /// x development into the treasury.
    fn collect_taxes(&mut self) {
        let mut due = 0.0;
        {
            let mut q = self.world.query::<&CityBuilding>();
            for b in q.iter(&self.world) {
                due += TAX_PER_CAP_DAY * b.capacity as f32 * b.development;
            }
        }
        if due != 0.0 {
            if let Some(mut t) = self.world.get_resource_mut::<Treasury>() {
                t.0.accrue(due);
            }
        }
    }

    pub fn treasury(&self) -> i64 {
        self.world.get_resource::<Treasury>().map(|t| t.0.cents).unwrap_or(0)
    }

    pub fn day(&self) -> u32 {
        self.world.get_resource::<DayCount>().map(|d| d.0).unwrap_or(0)
    }

    /// Replace the routing obstacles (building footprints).
    pub fn set_obstacles(&mut self, rects: Vec<Obstacle>) {
        if let Some(mut o) = self.world.get_resource_mut::<Obstacles>() {
            o.rects = rects;
        }
    }

    /// Serializable snapshot (Marshals analogue): agents, goals,
    /// workplaces, buildings, dwellings, obstacles, roads, clock, day,
    /// funds, treasury. In-flight state (walk targets, routes, claims,
    /// commutes, home links, queues, employments) is intentionally
    /// dropped — everything loads Idle, unemployed, and homeless;
    /// dwelling beds reload empty.
    pub fn snapshot(&mut self) -> SimSnapshot {
        let mut agents = Vec::new();
        {
            let mut q = self.world.query::<(&Position, &Needs, &Personality, &Skills)>();
            for (p, n, personality, skills) in q.iter(&self.world) {
                agents.push(AgentSnapshot {
                    x: p.x,
                    z: p.z,
                    hunger: n.hunger,
                    energy: n.energy,
                    sociability: n.sociability,
                    comfort: n.comfort,
                    hygiene: n.hygiene,
                    bladder: n.bladder,
                    fun: n.fun,
                    playful: personality.playful,
                    outgoing: personality.outgoing,
                    active: personality.active,
                    cooking: skills.cooking,
                    charisma: skills.charisma,
                });
            }
        }
        let mut goals = Vec::new();
        {
            let mut q = self.world.query::<&Goal>();
            for g in q.iter(&self.world) {
                goals.push(GoalSnapshot {
                    x: g.x,
                    z: g.z,
                    ads: g.ads.clone(),
                    attenuation: g.attenuation,
                });
            }
        }
        let mut workplaces = Vec::new();
        {
            let mut q = self.world.query::<&Workplace>();
            for w in q.iter(&self.world) {
                workplaces.push(WorkplaceSnapshot {
                    x: w.x,
                    z: w.z,
                    pay_per_hour: w.pay_per_hour,
                    shift_start: w.shift_start,
                    shift_end: w.shift_end,
                });
            }
        }
        let obstacles = self
            .world
            .get_resource::<Obstacles>()
            .map(|o| o.rects.clone())
            .unwrap_or_default();
        let roads = self.world.get_resource::<Roads>().map(|r| r.0.clone()).unwrap_or_default();
        let hour = self.world.get_resource::<HourOfDay>().map(|h| h.0).unwrap_or(8);
        let mut buildings = Vec::new();
        {
            let mut q = self.world.query::<&CityBuilding>();
            for b in q.iter(&self.world) {
                buildings.push(BuildingSnapshot {
                    label: b.label.clone(),
                    function: b.function,
                    min_x: b.min_x,
                    min_z: b.min_z,
                    max_x: b.max_x,
                    max_z: b.max_z,
                    development: b.development,
                    capacity: b.capacity,
                });
            }
        }
        let mut dwellings = Vec::new();
        {
            let mut q = self.world.query::<&Dwelling>();
            for d in q.iter(&self.world) {
                dwellings.push(DwellingSnapshot {
                    x: d.x,
                    z: d.z,
                    capacity: d.capacity,
                    taken: d.taken,
                });
            }
        }
        SimSnapshot { version: SNAPSHOT_VERSION, agents, goals, workplaces, buildings, dwellings, obstacles, roads, walls: self.world.get_resource::<Walls>().map(|w| w.0.clone()).unwrap_or_default(), hour, day: self.day(), funds: self.funds(), treasury: self.treasury() }
    }

    /// Restore a snapshot: swaps in a fresh world, respawns agents
    /// (Idle) and goals (unclaimed), restores obstacles and clock.
    /// Tuning (`SimConfig`) carries over untouched.
    pub fn restore(&mut self, snap: &SimSnapshot) {
        let cfg = self
            .world
            .get_resource::<SimConfig>()
            .cloned()
            .unwrap_or_default();
        self.world = World::new();
        self.world.insert_resource(cfg);
        self.world.init_resource::<Obstacles>();
        for a in &snap.agents {
            self.world.spawn((
                Position { x: a.x, z: a.z },
                Needs {
                    hunger: a.hunger,
                    energy: a.energy,
                    sociability: a.sociability,
                    comfort: a.comfort,
                    hygiene: a.hygiene,
                    bladder: a.bladder,
                    fun: a.fun,
                },
                AgentState::Idle,
                ActionQueue::default(),
                Personality {
                    playful: a.playful,
                    outgoing: a.outgoing,
                    active: a.active,
                },
                Skills { cooking: a.cooking, charisma: a.charisma },
            ));
        }
        for g in &snap.goals {
            self.world.spawn((Goal {
                x: g.x,
                z: g.z,
                ads: g.ads.clone(),
                attenuation: g.attenuation,
                claimed_by: None,
            },));
        }
        for w in &snap.workplaces {
            self.world.spawn((Workplace {
                x: w.x,
                z: w.z,
                pay_per_hour: w.pay_per_hour,
                shift_start: w.shift_start,
                shift_end: w.shift_end,
            },));
        }
        for d in &snap.dwellings {
            // Home links don't survive (entity refs); beds reload empty
            // and agents move back in through play.
            self.world.spawn((Dwelling {
                x: d.x,
                z: d.z,
                capacity: d.capacity,
                taken: 0,
            },));
        }
        self.world.insert_resource(Funds(Money { cents: snap.funds, frac: 0.0 }));
        self.world.insert_resource(Treasury(Money { cents: snap.treasury, frac: 0.0 }));
        self.world.insert_resource(DayCount(snap.day));
        self.world.insert_resource(Roads(snap.roads.clone()));
        self.world.insert_resource(Walls(snap.walls.clone()));
        for b in &snap.buildings {
            let mut building = CityBuilding::new(
                b.label.clone(),
                b.function,
                b.min_x,
                b.min_z,
                b.max_x,
                b.max_z,
            );
            building.development = b.development;
            self.world.spawn((building,));
        }
        self.set_obstacles(snap.obstacles.clone());
        self.set_hour(snap.hour);
        self.rebuild_rooms();
    }

    /// Snapshot to JSON string.
    pub fn save_json(&mut self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.snapshot())
    }

    /// Restore from a JSON string produced by [`Sim::save_json`].
    /// Rejects unknown snapshot versions.
    pub fn load_json(&mut self, s: &str) -> Result<(), String> {
        let snap: SimSnapshot = serde_json::from_str(s).map_err(|e| e.to_string())?;
        if snap.version != SNAPSHOT_VERSION {
            return Err(format!(
                "unsupported snapshot version {} (want {})",
                snap.version, SNAPSHOT_VERSION
            ));
        }
        self.restore(&snap);
        Ok(())
    }

    /// Advance the sim by `dt` seconds: decay needs, doze, work shifts,
    /// night rest, develop buildings, run player queue, assign goals to
    /// the neediest free agents, walk them there, then chat.
    pub fn step(&mut self, dt: f32) {
        let cfg = self.world.resource::<SimConfig>().clone();
        decay_needs(&mut self.world, dt, &cfg);
        doze_at_night(&mut self.world, dt, &cfg);
        work_shifts(&mut self.world, dt);
        night_rest(&mut self.world);
        develop_buildings(&mut self.world, dt);
        assign_goals(&mut self.world);
        pump_queue(&mut self.world);
        walk_agents(&mut self.world, dt, &cfg);
        socialize(&mut self.world, dt, &cfg);
    }
}

fn decay_needs(world: &mut World, dt: f32, cfg: &SimConfig) {
    let mut query = world.query::<&mut Needs>();
    for mut needs in query.iter_mut(world) {
        needs.hunger = (needs.hunger - cfg.hunger_decay * dt).max(0.0);
        needs.energy = (needs.energy - cfg.energy_decay * dt).max(0.0);
        needs.sociability = (needs.sociability - cfg.sociability_decay * dt).max(0.0);
        needs.comfort = (needs.comfort - cfg.comfort_decay * dt).max(0.0);
        needs.hygiene = (needs.hygiene - cfg.hygiene_decay * dt).max(0.0);
        needs.bladder = (needs.bladder - cfg.bladder_decay * dt).max(0.0);
        needs.fun = (needs.fun - cfg.fun_decay * dt).max(0.0);
    }
}

/// At night, sleepers doze at full rate; the homeless doze rough at
/// half rate (net zero against base decay, so they tread water).
fn doze_at_night(world: &mut World, dt: f32, cfg: &SimConfig) {
    let night = world.resource::<HourOfDay>().0;
    if !is_night(night) {
        return;
    }
    let mut query = world.query::<(&AgentState, &mut Needs)>();
    for (state, mut needs) in query.iter_mut(world) {
        let rate = match *state {
            AgentState::Sleeping => cfg.night_restore,
            AgentState::Idle => cfg.night_restore / 2.0,
            _ => continue,
        };
        needs.energy = (needs.energy + rate * dt).min(1.0);
    }
}

/// Night rest: sleepers wake at dawn; idle homed agents with an empty
/// queue head home (unreachable homes sleep rough where they stand).
fn night_rest(world: &mut World) {
    let night = world.get_resource::<HourOfDay>().map(|h| h.0).unwrap_or(8);
    if !is_night(night) {
        let mut q = world.query::<(Entity, &AgentState)>();
        let mut wake: Vec<Entity> = Vec::new();
        for (e, st) in q.iter(world) {
            if *st == AgentState::Sleeping {
                wake.push(e);
            }
        }
        for e in wake {
            world.entity_mut(e).insert(AgentState::Idle);
        }
        return;
    }
    let obs = world.resource::<Obstacles>().clone();
    let roads = world.resource::<Roads>().clone();
    let walls = world.resource::<Walls>().clone();
    let mut go_home: Vec<(Entity, Entity, Route, f32, f32)> = Vec::new();
    let mut homeless: Vec<Entity> = Vec::new();
    {
        let mut q = world.query::<(Entity, &Position, &AgentState, &Home, &ActionQueue)>();
        for (agent, pos, state, home, queue) in q.iter(world) {
            if *state != AgentState::Idle || !queue.0.is_empty() {
                continue;
            }
            if world.get::<GoingHome>(agent).is_some()
                || world.get::<WalkTarget>(agent).is_some()
            {
                continue;
            }
            let Some(d) = world.get::<Dwelling>(home.0) else {
                homeless.push(agent); // home demolished: link cleared below
                continue;
            };
            if let Some(points) = plan_route(&obs, &roads, &walls, [pos.x, pos.z], [d.x, d.z]) {
                go_home.push((agent, home.0, Route(points), d.x, d.z));
            }
        }
    }
    for agent in homeless {
        world.entity_mut(agent).remove::<Home>(); // sleep rough tonight
    }
    for (agent, dwelling, route, x, z) in go_home {
        world.entity_mut(agent).insert(GoingHome(dwelling));
        world.entity_mut(agent).insert((
            WalkTarget { x, z, claim: None },
            route,
            AgentState::SeekGoal,
        ));
    }
}

/// Proximity chat: idle agents near each other socialize (restore
/// sociability) until a need gets urgent, then free will wins and they
/// drop back to Idle for autonomy to pick up.
fn socialize(world: &mut World, dt: f32, cfg: &SimConfig) {
    let people: Vec<(Entity, f32, f32, f32)> = {
        let mut q = world.query::<(Entity, &Position, &Needs, &AgentState)>();
        q.iter(world)
            .filter(|(_, _, _, st)| **st == AgentState::Idle || **st == AgentState::Socialize)
            .map(|(e, p, n, _)| (e, p.x, p.z, n.lowest()))
            .collect()
    };
    let mut chatting: std::collections::HashSet<Entity> = std::collections::HashSet::new();
    for (i, (a, ax, az, a_low)) in people.iter().enumerate() {
        if *a_low < cfg.chat_break {
            continue;
        }
        for (b, bx, bz, b_low) in people.iter().skip(i + 1) {
            if *b_low < cfg.chat_break {
                continue;
            }
            let d2 = (bx - ax).powi(2) + (bz - az).powi(2);
            if d2 <= cfg.chat_radius.powi(2) {
                chatting.insert(*a);
                chatting.insert(*b);
                for e in [*a, *b] {
                    if let Some(mut needs) = world.get_mut::<Needs>(e) {
                        needs.sociability =
                            (needs.sociability + cfg.chat_restore * dt).min(1.0);
                    }
                }
            }
        }
    }
    for (e, _, _, _) in &people {
        let want = if chatting.contains(e) {
            AgentState::Socialize
        } else {
            AgentState::Idle
        };
        if let Some(mut st) = world.get_mut::<AgentState>(*e) {
            *st = want;
        }
    }
}

/// Job shifts: clock out commute holders whose job ended, moved, or
/// vanished; send idle employees to work in-shift; pay on-site workers.
fn work_shifts(world: &mut World, dt: f32) {
    let hour = world.get_resource::<HourOfDay>().map(|h| h.0).unwrap_or(8);
    let obs = world.resource::<Obstacles>().clone();
    let roads = world.resource::<Roads>().clone();
    let walls = world.resource::<Walls>().clone();
    let arrive = world.get_resource::<SimConfig>().map(|c| c.arrive_dist).unwrap_or(1.5);
    // Clock out: work-directed agents (Working, or SeekGoal with a
    // Commute) whose shift is over, moved jobs, or lost the site.
    let mut clock_out: Vec<Entity> = Vec::new();
    {
        let mut q = world.query::<(Entity, &AgentState, &Commute)>();
        for (agent, state, commute) in q.iter(world) {
            if *state != AgentState::Working && *state != AgentState::SeekGoal {
                continue;
            }
            let ok = world.get::<Workplace>(commute.0).is_some_and(|w| {
                world
                    .get::<Employment>(agent)
                    .is_some_and(|e| e.workplace == commute.0)
                    && in_shift(hour, w.shift_start, w.shift_end)
            });
            if !ok {
                clock_out.push(agent);
            }
        }
    }
    for agent in clock_out {
        world.entity_mut(agent).remove::<Commute>();
        world.entity_mut(agent).remove::<WalkTarget>();
        world.entity_mut(agent).remove::<Route>();
        world.entity_mut(agent).insert(AgentState::Idle);
    }
    // Commute: idle employees with an empty queue head to work in-shift.
    let mut commutes: Vec<(Entity, Entity, Route, f32, f32)> = Vec::new();
    {
        let mut q =
            world.query::<(Entity, &Position, &AgentState, &Employment, &ActionQueue)>();
        for (agent, pos, state, emp, queue) in q.iter(world) {
            if *state != AgentState::Idle || !queue.0.is_empty() {
                continue;
            }
            if world.get::<Commute>(agent).is_some()
                || world.get::<WalkTarget>(agent).is_some()
            {
                continue;
            }
            let Some(w) = world.get::<Workplace>(emp.workplace).copied() else {
                continue; // site demolished: stays idle until dismissed
            };
            if !in_shift(hour, w.shift_start, w.shift_end) {
                continue;
            }
            if let Some(points) = plan_route(&obs, &roads, &walls, [pos.x, pos.z], [w.x, w.z]) {
                commutes.push((agent, emp.workplace, Route(points), w.x, w.z));
            }
        }
    }
    for (agent, wp, route, x, z) in commutes {
        world.entity_mut(agent).insert(Commute(wp));
        world.entity_mut(agent).insert((
            WalkTarget { x, z, claim: None },
            route,
            AgentState::SeekGoal,
        ));
    }
    // Pay on-site workers.
    let mut earned = 0.0;
    {
        let mut q = world.query::<(&AgentState, &Employment, &Position)>();
        for (state, emp, pos) in q.iter(world) {
            if *state != AgentState::Working {
                continue;
            }
            let Some(w) = world.get::<Workplace>(emp.workplace) else {
                continue;
            };
            if !in_shift(hour, w.shift_start, w.shift_end) {
                continue;
            }
            let d2 = (w.x - pos.x).powi(2) + (w.z - pos.z).powi(2);
            if d2 <= (arrive * 2.0).powi(2) {
                earned += w.pay_per_hour / 3600.0 * dt;
            }
        }
    }
    if earned != 0.0 {
        if let Some(mut funds) = world.get_resource_mut::<Funds>() {
            funds.0.accrue(earned);
        }
    }
}

/// Idle agents score every goal — advertised restoration weighted by low
/// need, attenuated by distance — and head for the best one above
/// [`MIN_SCORE`].
fn assign_goals(world: &mut World) {
    // Snapshot goals so we can mutate agents freely.
    let goals: Vec<(Entity, f32, f32, Vec<(NeedKind, f32)>, f32, Option<Entity>)> = {
        let mut q = world.query::<(Entity, &Goal)>();
        q.iter(world)
            .map(|(e, g)| (e, g.x, g.z, g.ads.clone(), g.attenuation, g.claimed_by))
            .collect()
    };
    if goals.is_empty() {
        return;
    }
    let obs = world.resource::<Obstacles>().clone();
    let roads = world.resource::<Roads>().clone();
    let walls = world.resource::<Walls>().clone();
    // Phase 1: score candidates per free agent (query borrow only).
    // Phase 2: tie-break with the seeded RNG, plan, assign (needs the
    // RNG mutably, which phase 1's shared borrow forbids).
    let mut pending: Vec<(Entity, f32, f32, Personality, Vec<(f32, Entity, f32, f32)>)> =
        Vec::new();
    {
        let mut entity_query = world.query::<(
            Entity,
            &Position,
            &Needs,
            &AgentState,
            &ActionQueue,
            &Personality,
        )>();
        for (entity, pos, needs, state, queue, personality) in entity_query.iter(world) {
            if *state != AgentState::Idle {
                continue;
            }
            if !queue.0.is_empty() {
                continue; // player orders win over autonomy
            }
            // Score all free goals; routing + exclusivity resolve later.
            let mut candidates: Vec<(f32, Entity, f32, f32)> = Vec::new(); // score, goal, x, z
            for (goal, gx, gz, ads, atten, claimed) in &goals {
                if claimed.is_some_and(|c| c != entity) {
                    continue; // object is occupied
                }
                let dist = ((gx - pos.x).powi(2) + (gz - pos.z).powi(2)).sqrt();
                let mut gain = 0.0;
                for (kind, amount) in ads {
                    let v = needs.value(*kind);
                    let eff = amount.min(1.0 - v).max(0.0);
                    gain += eff * motive_weight(v) * personality.factor(*kind);
                }
                let score = gain / (1.0 + atten * dist);
                if score > MIN_SCORE {
                    candidates.push((score, *goal, *gx, *gz));
                }
            }
            if !candidates.is_empty() {
                pending.push((entity, pos.x, pos.z, *personality, candidates));
            }
        }
    }
    // Collect assignments first to keep borrow scopes simple.
    let mut assignments: Vec<(Entity, Entity, Route, f32, f32)> = Vec::new();
    // Goals taken this tick are exclusive: later agents in the same tick
    // see them as occupied (snapshot would otherwise double-book).
    let mut taken: std::collections::HashSet<Entity> = std::collections::HashSet::new();
    let mut rng = world.get_resource_mut::<SimRng>().expect("SimRng");
    for (entity, px, pz, _personality, mut candidates) in pending {
        candidates.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        // Best first, but rotate the tied-best prefix randomly so
        // identical twins diverge instead of marching in lockstep.
        let best = candidates[0].0;
        let tied = candidates
            .iter()
            .take_while(|(s, _, _, _)| best - *s <= 1e-6)
            .count()
            .max(1);
        let off = rng.0.random_range(0..tied);
        let order: Vec<usize> =
            (off..tied).chain(0..off).chain(tied..candidates.len()).collect();
        // The first routable one wins (unreachable targets fail,
        // agent tries the next).
        for i in order {
            let (_, goal, gx, gz) = candidates[i];
            if taken.contains(&goal) {
                continue;
            }
            if let Some(points) = plan_route(&obs, &roads, &walls, [px, pz], [gx, gz]) {
                taken.insert(goal);
                assignments.push((entity, goal, Route(points), gx, gz));
                break;
            }
        }
    }
    drop(rng);
    for (entity, goal, route, gx, gz) in assignments {
        if let Some(mut g) = world.get_mut::<Goal>(goal) {
            g.claimed_by = Some(entity);
        }
        world.entity_mut(entity).insert((
            WalkTarget {
                x: gx,
                z: gz,
                claim: Some(goal),
            },
            route,
            AgentState::SeekGoal,
        ));
    }
}

/// Work the player queue: idle agents with no walk target pop the front
/// action into a WalkTarget. Dead goals are dropped silently.
fn pump_queue(world: &mut World) {
    let mut pops: Vec<(Entity, [f32; 2], Option<[f32; 2]>)> = Vec::new();
    {
        let mut q = world.query::<(Entity, &Position, &AgentState, &ActionQueue)>();
        for (entity, pos, state, queue) in q.iter(world) {
            if *state != AgentState::Idle {
                continue;
            }
            if world.get::<WalkTarget>(entity).is_some() {
                continue;
            }
            if queue.0.front().is_some() {
                let dest = match queue.0.front().copied() {
                    Some(ActionKind::Goto { x, z }) => Some([x, z]),
                    Some(ActionKind::Use { goal }) => {
                        world.get::<Goal>(goal).map(|g| [g.x, g.z])
                    }
                    None => None,
                };
                pops.push((entity, [pos.x, pos.z], dest));
            }
        }
    }
    let obs = world.resource::<Obstacles>().clone();
    let roads = world.resource::<Roads>().clone();
    let walls = world.resource::<Walls>().clone();
    for (entity, from, dest) in pops {
        if let Some(mut q) = world.get_mut::<ActionQueue>(entity) {
            q.0.pop_front();
        }
        let Some(dest) = dest else { continue }; // dead goal: drop silently
        let Some(points) = plan_route(&obs, &roads, &walls, from, dest) else { continue }; // unreachable: drop
        world.entity_mut(entity).insert((
            WalkTarget {
                x: dest[0],
                z: dest[1],
                claim: None,
            },
            Route(points),
            AgentState::SeekGoal,
        ));
    }
}

fn walk_agents(world: &mut World, dt: f32, cfg: &SimConfig) {
    let mut done: Vec<Entity> = Vec::new();
    {
        let mut query = world.query::<(
            Entity,
            &mut Position,
            &AgentState,
            &WalkTarget,
            Option<&mut Route>,
        )>();
        for (entity, mut pos, state, target, route) in query.iter_mut(world) {
            if *state != AgentState::SeekGoal {
                continue;
            }
            // Head for the front waypoint, or the exact target when the
            // route is spent.
            let dest = match route.as_ref() {
                Some(r) if !r.0.is_empty() => r.0[0],
                _ => [target.x, target.z],
            };
            let dx = dest[0] - pos.x;
            let dz = dest[1] - pos.z;
            let dist = (dx * dx + dz * dz).sqrt();
            if dist <= cfg.arrive_dist {
                match route {
                    Some(mut r) if !r.0.is_empty() => {
                        r.0.remove(0);
                        continue;
                    }
                    _ => {
                        done.push(entity);
                        continue;
                    }
                }
            }
            let step = (cfg.walk_speed * dt).min(dist);
            pos.x += dx / dist * step;
            pos.z += dz / dist * step;
        }
    }
    // Apply the reached goal's ads (nearest goal within reach; the
    // object may have been removed while walking).
    for entity in done {
        let (px, pz) = {
            let pos = world.get::<Position>(entity).copied();
            match pos {
                Some(p) => (p.x, p.z),
                None => continue,
            }
        };
        // Work commute: clock in on site, else drop the stale commute.
        if let Some(site) = world.get::<Commute>(entity).map(|c| c.0) {
            let at_work = world.get::<Workplace>(site).is_some_and(|w| {
                (w.x - px).powi(2) + (w.z - pz).powi(2) <= (cfg.arrive_dist * 2.0).powi(2)
            });
            world.entity_mut(entity).remove::<WalkTarget>();
            world.entity_mut(entity).remove::<Route>();
            if at_work {
                world.entity_mut(entity).insert(AgentState::Working);
            } else {
                world.entity_mut(entity).remove::<Commute>();
                world.entity_mut(entity).insert(AgentState::Idle);
            }
            continue;
        }
        // Night commute: turn in (days are for errands, even at home).
        if let Some(home) = world.get::<GoingHome>(entity).map(|h| h.0) {
            let at_home = world.get::<Dwelling>(home).is_some_and(|d| {
                (d.x - px).powi(2) + (d.z - pz).powi(2) <= (cfg.arrive_dist * 2.0).powi(2)
            });
            let night = world.get_resource::<HourOfDay>().map(|h| h.0).unwrap_or(8);
            world.entity_mut(entity).remove::<GoingHome>();
            world.entity_mut(entity).remove::<WalkTarget>();
            world.entity_mut(entity).remove::<Route>();
            if at_home && is_night(night) {
                world.entity_mut(entity).insert(AgentState::Sleeping);
            } else {
                world.entity_mut(entity).insert(AgentState::Idle);
            }
            continue;
        }
        let mut best: Option<(f32, Vec<(NeedKind, f32)>)> = None;
        {
            let mut q = world.query::<&Goal>();
            for g in q.iter(world) {
                let d2 = (g.x - px).powi(2) + (g.z - pz).powi(2);
                if d2 > (cfg.arrive_dist * 2.0).powi(2) {
                    continue;
                }
                if best.as_ref().is_none_or(|(bd2, _)| d2 < *bd2) {
                    best = Some((d2, g.ads.clone()));
                }
            }
        }
        if let Some((_, ads)) = best {
            let skills = world.get::<Skills>(entity).copied().unwrap_or_default();
            if let Some(mut needs) = world.get_mut::<Needs>(entity) {
                for (kind, amount) in &ads {
                    needs.restore(*kind, *amount * skills.factor(*kind));
                }
            }
            // Practice makes better: relevant skills grow per arrival.
            if let Some(mut s) = world.get_mut::<Skills>(entity) {
                for (kind, _) in &ads {
                    s.learn(*kind);
                }
            }
        }
        // Release the goal claim held while walking.
        let claim = world.get::<WalkTarget>(entity).and_then(|t| t.claim);
        if let Some(goal) = claim {
            if let Some(mut g) = world.get_mut::<Goal>(goal) {
                if g.claimed_by == Some(entity) {
                    g.claimed_by = None;
                }
            }
        }
        world.entity_mut(entity).remove::<WalkTarget>();
        world.entity_mut(entity).remove::<Route>();
        world.entity_mut(entity).insert(AgentState::Idle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungry_agent_seeks_food_and_recovers() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        // Starve the agent so it picks the food goal.
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;

        sim.step(0.1);
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::SeekGoal)
        );

        // Walk until arrival (4 u/s over ~10u).
        for _ in 0..60 {
            sim.step(0.1);
        }
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<Needs>(agent).unwrap().hunger > 0.4);
    }

    #[test]
    fn content_agents_stay_idle_and_needs_decay() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.spawn_goal(5.0, 5.0, NeedKind::Hunger);
        let before = sim.world.get::<Needs>(agent).unwrap().hunger;
        sim.step(1.0);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<Needs>(agent).unwrap().hunger < before);
    }

    #[test]
    fn per_need_decay_rates_differ() {
        let mut sim = Sim::new();
        {
            let mut cfg = sim.world.resource_mut::<SimConfig>();
            cfg.hunger_decay = 0.1;
            cfg.energy_decay = 0.0;
            cfg.sociability_decay = 0.0;
        }
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.step(1.0);
        let needs = sim.world.get::<Needs>(agent).unwrap();
        assert!((needs.hunger - 0.7).abs() < 1e-6);
        assert!((needs.energy - 0.8).abs() < 1e-6);
        assert!((needs.sociability - 0.8).abs() < 1e-6);
    }

    #[test]
    fn attenuation_prefers_nearer_goal() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_goal(30.0, 0.0, NeedKind::Hunger);
        sim.spawn_goal(3.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::SeekGoal)
        );
        let t = sim.world.get::<WalkTarget>(agent).unwrap();
        assert_eq!((t.x, t.z), (3.0, 0.0));
    }

    #[test]
    fn weak_far_want_stays_idle() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.49;
        sim.spawn_goal(60.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
    }

    #[test]
    fn multi_ad_goal_restores_all_advertised_needs() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        {
            let mut needs = sim.world.get_mut::<Needs>(agent).unwrap();
            needs.hunger = 0.1;
            needs.energy = 0.1;
        }
        sim.spawn_goal_ads(
            1.0,
            0.0,
            vec![(NeedKind::Hunger, 0.5), (NeedKind::Energy, 0.5)],
            DEFAULT_ATTENUATION,
        );
        sim.step(0.1); // within arrive_dist: arrival applies both ads.
        let needs = sim.world.get::<Needs>(agent).unwrap();
        assert!(needs.hunger > 0.4, "hunger restored: {}", needs.hunger);
        assert!(needs.energy > 0.4, "energy restored: {}", needs.energy);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
    }

    #[test]
    fn player_goto_wins_over_autonomy() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        sim.order(agent, ActionKind::Goto { x: -7.0, z: 4.0 });
        sim.step(0.1);
        let t = sim.world.get::<WalkTarget>(agent).unwrap();
        assert_eq!((t.x, t.z), (-7.0, 4.0));
        assert!(sim.world.get::<ActionQueue>(agent).unwrap().0.is_empty());
    }

    #[test]
    fn player_use_walks_and_restores() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        let food = sim.spawn_goal(8.0, 0.0, NeedKind::Hunger);
        sim.order(agent, ActionKind::Use { goal: food });
        for _ in 0..60 {
            sim.step(0.1);
        }
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<Needs>(agent).unwrap().hunger > 0.4);
    }

    #[test]
    fn player_use_of_dead_goal_is_dropped() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        let food = sim.spawn_goal(8.0, 0.0, NeedKind::Hunger);
        sim.world.despawn(food);
        sim.order(agent, ActionKind::Use { goal: food });
        sim.step(0.1);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<WalkTarget>(agent).is_none());
    }

    #[test]
    fn second_agent_waits_for_occupied_goal() {
        let mut sim = Sim::new();
        let a = sim.spawn_agent(0.0, 0.0);
        let b = sim.spawn_agent(1.0, 0.0);
        sim.world.get_mut::<Needs>(a).unwrap().hunger = 0.1;
        sim.world.get_mut::<Needs>(b).unwrap().hunger = 0.1;
        let food = sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        // Exactly one of them walks; the goal is claimed.
        let seeking = [a, b]
            .into_iter()
            .filter(|e| sim.world.get::<AgentState>(*e) == Some(&AgentState::SeekGoal))
            .count();
        assert_eq!(seeking, 1);
        assert!(sim.world.get::<Goal>(food).unwrap().claimed_by.is_some());
        // A second stall frees the other agent.
        sim.spawn_goal(-10.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        assert_eq!(
            sim.world.get::<AgentState>(a),
            Some(&AgentState::SeekGoal)
        );
        assert_eq!(
            sim.world.get::<AgentState>(b),
            Some(&AgentState::SeekGoal)
        );
    }

    #[test]
    fn claim_released_on_arrival() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        let food = sim.spawn_goal(6.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        assert!(sim.world.get::<Goal>(food).unwrap().claimed_by.is_some());
        for _ in 0..60 {
            sim.step(0.1);
        }
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert_eq!(sim.world.get::<Goal>(food).unwrap().claimed_by, None);
    }

    #[test]
    fn straight_line_when_no_obstacles() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        let route = sim.world.get::<Route>(agent).unwrap();
        assert!(route.0.is_empty());
    }

    #[test]
    fn route_avoids_wall_and_arrives() {
        let mut sim = Sim::new();
        sim.set_obstacles(vec![Obstacle {
            min_x: 4.0,
            min_z: -10.0,
            max_x: 6.0,
            max_z: 10.0,
        }]);
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::SeekGoal)
        );
        assert!(!sim.world.get::<Route>(agent).unwrap().0.is_empty());
        for _ in 0..400 {
            sim.step(0.1);
        }
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<Needs>(agent).unwrap().hunger > 0.4);
    }

    #[test]
    fn route_prefers_roads_over_grass() {
        let mut sim = Sim::new();
        // Road along z=4: detouring onto it (~14 cost) beats 20u of
        // grass (~20 cost), so the route should ride the road row.
        sim.set_roads(vec![vec![[-4.0, 4.0], [24.0, 4.0]]]);
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_goal(20.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        let route = sim.world.get::<Route>(agent).unwrap();
        assert!(!route.0.is_empty());
        assert!(
            route.0.iter().any(|p| p[1] > 2.0),
            "rides the road: {:?}",
            route.0
        );
        for _ in 0..400 {
            sim.step(0.1);
        }
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<Needs>(agent).unwrap().hunger > 0.4);
    }

    #[test]
    fn sealed_goal_fails_and_agent_stays_idle() {
        let mut sim = Sim::new();
        // 20x20 sealed box around the goal (walls 2 thick).
        sim.set_obstacles(vec![
            Obstacle { min_x: 4.0, min_z: -6.0, max_x: 6.0, max_z: 6.0 },
            Obstacle { min_x: 14.0, min_z: -6.0, max_x: 16.0, max_z: 6.0 },
            Obstacle { min_x: 6.0, min_z: -6.0, max_x: 14.0, max_z: -4.0 },
            Obstacle { min_x: 6.0, min_z: 4.0, max_x: 14.0, max_z: 6.0 },
        ]);
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<WalkTarget>(agent).is_none());
    }

    #[test]
    fn night_idle_dozing_holds_energy() {
        let mut sim = Sim::new();
        sim.set_hour(23);
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().energy = 0.3;
        sim.step(1.0);
        // Rough sleep (half rate) cancels base decay: no collapse.
        assert!(sim.world.get::<Needs>(agent).unwrap().energy >= 0.3 - 1e-4);
    }

    #[test]
    fn day_idle_does_not_restore_energy() {
        let mut sim = Sim::new();
        sim.set_hour(12);
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().energy = 0.3;
        sim.step(1.0);
        assert!(sim.world.get::<Needs>(agent).unwrap().energy < 0.3);
    }

    #[test]
    fn set_hour_wraps_into_day() {
        let mut sim = Sim::new();
        sim.set_hour(25);
        assert_eq!(sim.world.resource::<HourOfDay>(), &HourOfDay(1));
        assert!(is_night(23) && is_night(3) && !is_night(8) && !is_night(21));
    }

    #[test]
    fn adjacent_idle_agents_chat() {
        let mut sim = Sim::new();
        let a = sim.spawn_agent(0.0, 0.0);
        let b = sim.spawn_agent(1.0, 0.0);
        for e in [a, b] {
            sim.world.get_mut::<Needs>(e).unwrap().sociability = 0.3;
        }
        sim.step(1.0);
        assert_eq!(
            sim.world.get::<AgentState>(a),
            Some(&AgentState::Socialize)
        );
        assert_eq!(
            sim.world.get::<AgentState>(b),
            Some(&AgentState::Socialize)
        );
        assert!(sim.world.get::<Needs>(a).unwrap().sociability > 0.3);
    }

    #[test]
    fn distant_agents_do_not_chat() {
        let mut sim = Sim::new();
        let a = sim.spawn_agent(0.0, 0.0);
        let b = sim.spawn_agent(50.0, 0.0);
        for e in [a, b] {
            sim.world.get_mut::<Needs>(e).unwrap().sociability = 0.3;
        }
        sim.step(1.0);
        assert_eq!(sim.world.get::<AgentState>(a), Some(&AgentState::Idle));
        assert_eq!(sim.world.get::<AgentState>(b), Some(&AgentState::Idle));
    }

    #[test]
    fn urgent_need_breaks_chat() {
        let mut sim = Sim::new();
        let a = sim.spawn_agent(0.0, 0.0);
        let b = sim.spawn_agent(1.0, 0.0);
        sim.world.get_mut::<Needs>(a).unwrap().hunger = 0.1;
        sim.step(0.1);
        // Too hungry to chat: back to Idle for autonomy.
        assert_eq!(sim.world.get::<AgentState>(a), Some(&AgentState::Idle));
        assert_eq!(sim.world.get::<AgentState>(b), Some(&AgentState::Idle));
    }

    fn sample_world() -> Sim {
        let mut sim = Sim::new();
        sim.set_hour(23);
        sim.set_obstacles(vec![Obstacle {
            min_x: 4.0,
            min_z: -10.0,
            max_x: 6.0,
            max_z: 10.0,
        }]);
        let a = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(a).unwrap().hunger = 0.2;
        sim.spawn_agent(30.0, -30.0);
        sim.spawn_goal_ads(
            10.0,
            0.0,
            vec![(NeedKind::Hunger, 0.5), (NeedKind::Energy, 0.3)],
            DEFAULT_ATTENUATION,
        );
        sim.step(0.1); // let claims/queues form
        sim
    }

    #[test]
    fn snapshot_roundtrip_preserves_world() {
        let mut sim = sample_world();
        let snap = sim.snapshot();
        assert_eq!(snap.agents.len(), 2);
        assert_eq!(snap.goals.len(), 1);
        assert_eq!(snap.hour, 23);
        let mut sim2 = Sim::new();
        sim2.restore(&snap);
        assert_eq!(snap, sim2.snapshot());
    }

    #[test]
    fn json_roundtrip_and_garbage() {
        let mut sim = sample_world();
        let json = sim.save_json().expect("serializes");
        let mut sim2 = Sim::new();
        sim2.load_json(&json).expect("deserializes");
        assert_eq!(sim.snapshot(), sim2.snapshot());
        assert!(sim2.load_json("{not json").is_err());
        // Wrong version rejected even when the shape parses.
        let mut v9: serde_json::Value = serde_json::from_str(&json).unwrap();
        v9["version"] = serde_json::json!(9);
        assert!(sim2.load_json(&v9.to_string()).is_err());
    }

    #[test]
    fn restore_drops_inflight_state() {
        let mut sim = sample_world();
        let snap = sim.snapshot();
        let mut sim2 = Sim::new();
        sim2.restore(&snap);
        // Everything loads Idle with no targets, routes, or claims.
        let mut q = sim2.world.query::<(&AgentState, Option<&WalkTarget>, Option<&Route>)>();
        for (st, target, route) in q.iter(&sim2.world) {
            assert_eq!(*st, AgentState::Idle);
            assert!(target.is_none());
            assert!(route.is_none());
        }
        let mut gq = sim2.world.query::<&Goal>();
        for g in gq.iter(&sim2.world) {
            assert_eq!(g.claimed_by, None);
        }
    }

    #[test]
    fn in_shift_handles_overnight() {
        assert!(in_shift(10, 9, 17));
        assert!(!in_shift(8, 9, 17));
        assert!(!in_shift(17, 9, 17));
        assert!(in_shift(23, 22, 6));
        assert!(in_shift(3, 22, 6));
        assert!(!in_shift(12, 22, 6));
        assert!(!in_shift(10, 9, 9)); // empty shift
    }

    #[test]
    fn employed_agent_commutes_and_earns() {
        let mut sim = Sim::new();
        sim.set_hour(10);
        let agent = sim.spawn_agent(0.0, 0.0);
        let office = sim.spawn_workplace(8.0, 0.0, 36.0, 9, 17);
        sim.employ(agent, office);
        sim.step(0.1);
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::SeekGoal)
        );
        for _ in 0..100 {
            sim.step(0.1);
        }
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::Working)
        );
        // $36/h over ~10 sim-seconds on site earns cents, not zero.
        for _ in 0..100 {
            sim.step(1.0);
        }
        assert!(sim.funds() > 0, "funds: {}", sim.funds());
    }

    #[test]
    fn off_shift_employee_stays_idle() {
        let mut sim = Sim::new();
        sim.set_hour(20);
        let agent = sim.spawn_agent(0.0, 0.0);
        let office = sim.spawn_workplace(8.0, 0.0, 36.0, 9, 17);
        sim.employ(agent, office);
        sim.step(0.1);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<WalkTarget>(agent).is_none());
    }

    #[test]
    fn shift_end_clocks_out() {
        let mut sim = Sim::new();
        sim.set_hour(10);
        let agent = sim.spawn_agent(0.0, 0.0);
        let office = sim.spawn_workplace(2.0, 0.0, 36.0, 9, 17);
        sim.employ(agent, office);
        for _ in 0..30 {
            sim.step(0.1);
        }
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::Working)
        );
        sim.set_hour(18);
        sim.step(0.1);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<Commute>(agent).is_none());
    }

    #[test]
    fn player_order_defers_work_until_done() {
        let mut sim = Sim::new();
        sim.set_hour(10);
        let agent = sim.spawn_agent(0.0, 0.0);
        let office = sim.spawn_workplace(8.0, 0.0, 36.0, 9, 17);
        sim.employ(agent, office);
        sim.order(agent, ActionKind::Goto { x: -30.0, z: 0.0 });
        sim.step(0.1);
        // Player order wins over the commute.
        let t = sim.world.get::<WalkTarget>(agent).unwrap();
        assert_eq!((t.x, t.z), (-30.0, 0.0));
    }

    #[test]
    fn dismiss_ends_employment() {
        let mut sim = Sim::new();
        sim.set_hour(10);
        let agent = sim.spawn_agent(0.0, 0.0);
        let office = sim.spawn_workplace(8.0, 0.0, 36.0, 9, 17);
        sim.employ(agent, office);
        sim.step(0.1);
        sim.dismiss(agent);
        sim.step(0.1);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<Employment>(agent).is_none());
    }

    #[test]
    fn building_capacity_comes_from_area() {
        let b = CityBuilding::new(
            "t".to_string(),
            ZoneFunction::Residential,
            0.0,
            0.0,
            10.0,
            10.0,
        );
        assert_eq!(b.capacity, 4); // 100u² / 25
        assert_eq!((b.center()), ([5.0, 5.0]));
    }

    #[test]
    fn building_develops_with_road_access() {
        let mut sim = Sim::new();
        sim.set_roads(vec![vec![[-10.0, 2.0], [10.0, 2.0]]]);
        let b = sim.spawn_building(
            "t".to_string(),
            ZoneFunction::Commercial,
            -4.0,
            -4.0,
            4.0,
            4.0,
        );
        sim.step(10.0);
        let dev = sim.world.get::<CityBuilding>(b).unwrap().development;
        assert!(dev > 0.0, "developed: {dev}");
    }

    #[test]
    fn isolated_building_decays() {
        let mut sim = Sim::new();
        let b = sim.spawn_building(
            "t".to_string(),
            ZoneFunction::Industrial,
            -4.0,
            -4.0,
            4.0,
            4.0,
        );
        sim.world.get_mut::<CityBuilding>(b).unwrap().development = 0.5;
        sim.step(10.0);
        let dev = sim.world.get::<CityBuilding>(b).unwrap().development;
        assert!(dev < 0.5, "decayed: {dev}");
    }

    #[test]
    fn move_in_claims_bed_and_full_rejects() {
        let mut sim = Sim::new();
        let a = sim.spawn_agent(0.0, 0.0);
        let b = sim.spawn_agent(1.0, 0.0);
        let home = sim.spawn_dwelling(10.0, 0.0, 1);
        assert!(sim.move_in(a, home));
        assert_eq!(sim.world.get::<Dwelling>(home).unwrap().taken, 1);
        assert!(!sim.move_in(b, home)); // full: keeps old (no) home
        assert!(sim.world.get::<Home>(b).is_none());
        sim.move_out(a);
        assert_eq!(sim.world.get::<Dwelling>(home).unwrap().taken, 0);
        assert!(sim.move_in(b, home));
    }

    #[test]
    fn night_goes_home_and_sleeps() {
        let mut sim = Sim::new();
        sim.set_hour(23);
        let agent = sim.spawn_agent(0.0, 0.0);
        let home = sim.spawn_dwelling(6.0, 0.0, 2);
        sim.move_in(agent, home);
        sim.world.get_mut::<Needs>(agent).unwrap().energy = 0.3;
        sim.step(0.1);
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::SeekGoal)
        );
        for _ in 0..60 {
            sim.step(0.1);
        }
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::Sleeping)
        );
        let before = sim.world.get::<Needs>(agent).unwrap().energy;
        sim.step(10.0);
        assert!(sim.world.get::<Needs>(agent).unwrap().energy > before);
    }

    #[test]
    fn morning_wakes_sleepers() {
        let mut sim = Sim::new();
        sim.set_hour(23);
        let agent = sim.spawn_agent(0.0, 0.0);
        let home = sim.spawn_dwelling(1.0, 0.0, 2);
        sim.move_in(agent, home);
        sim.step(0.1); // arrival: within arrive_dist
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::Sleeping)
        );
        sim.set_hour(7);
        sim.step(0.1);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
    }

    #[test]
    fn rough_sleep_treads_water() {
        let mut sim = Sim::new();
        sim.set_hour(23);
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().energy = 0.3;
        for _ in 0..10 {
            sim.step(1.0);
        }
        // Half-rate doze exactly cancels base decay: homeless hold steady.
        let e = sim.world.get::<Needs>(agent).unwrap().energy;
        assert!((e - 0.3).abs() < 1e-4, "rough sleep: {e}");
    }

    #[test]
    fn demolished_home_sleeps_rough_and_unlinks() {
        let mut sim = Sim::new();
        sim.set_hour(23);
        let agent = sim.spawn_agent(0.0, 0.0);
        let home = sim.spawn_dwelling(6.0, 0.0, 2);
        sim.move_in(agent, home);
        sim.world.despawn(home);
        sim.step(0.1);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<Home>(agent).is_none());
    }

    #[test]
    fn restore_empties_beds_and_unlinks_homes() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        let home = sim.spawn_dwelling(6.0, 0.0, 2);
        sim.move_in(agent, home);
        let snap = sim.snapshot();
        let mut sim2 = Sim::new();
        sim2.restore(&snap);
        let mut q = sim2.world.query::<&Dwelling>();
        for d in q.iter(&sim2.world) {
            assert_eq!(d.taken, 0);
        }
        let mut hq = sim2.world.query::<&Home>();
        assert_eq!(hq.iter(&sim2.world).count(), 0);
    }

    #[test]
    fn midnight_tick_collects_taxes_and_advances_day() {
        let mut sim = Sim::new();
        let b = sim.spawn_building(
            "t".to_string(),
            ZoneFunction::Commercial,
            0.0,
            0.0,
            10.0,
            10.0,
        ); // capacity 4
        sim.world.get_mut::<CityBuilding>(b).unwrap().development = 1.0;
        assert_eq!(sim.day(), 0);
        sim.set_hour(23);
        assert_eq!(sim.day(), 0); // no wrap yet
        sim.set_hour(5);
        assert_eq!(sim.day(), 1);
        assert_eq!(sim.treasury(), (TAX_PER_CAP_DAY * 4.0 * 100.0) as i64);
        // Same-day hours don't double-collect.
        sim.set_hour(6);
        assert_eq!(sim.day(), 1);
        assert_eq!(sim.treasury(), (TAX_PER_CAP_DAY * 4.0 * 100.0) as i64);
    }

    #[test]
    fn undeveloped_buildings_pay_nothing() {
        let mut sim = Sim::new();
        sim.spawn_building(
            "t".to_string(),
            ZoneFunction::Industrial,
            0.0,
            0.0,
            10.0,
            10.0,
        );
        sim.set_hour(23);
        sim.set_hour(0);
        assert_eq!(sim.day(), 1);
        assert_eq!(sim.treasury(), 0);
    }

    #[test]
    fn snapshot_preserves_day_funds_treasury() {
        let mut sim = Sim::new();
        sim.set_hour(23);
        sim.set_hour(1);
        assert_eq!(sim.day(), 1);
        let snap = sim.snapshot();
        let mut sim2 = Sim::new();
        sim2.restore(&snap);
        assert_eq!(sim2.day(), 1);
        assert_eq!(sim.snapshot(), sim2.snapshot());
    }

    /// Twin hungry agents, twin identical food stalls at mirrored
    /// positions: scores tie exactly, so the RNG decides.
    fn twin_cities(seed: u64) -> Vec<Option<(f32, f32)>> {
        let mut sim = Sim::new();
        sim.set_seed(seed);
        let a = sim.spawn_agent(0.0, 0.0);
        let b = sim.spawn_agent(0.0, 0.0);
        for e in [a, b] {
            sim.world.get_mut::<Needs>(e).unwrap().hunger = 0.1;
        }
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        sim.spawn_goal(-10.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        [a, b]
            .into_iter()
            .map(|e| sim.world.get::<WalkTarget>(e).map(|t| (t.x, t.z)))
            .collect()
    }

    #[test]
    fn same_seed_same_city() {
        assert_eq!(twin_cities(7), twin_cities(7));
    }

    #[test]
    fn both_twins_find_valid_targets() {
        // Whatever the seed picks, both agents must hold exactly one
        // of the two stalls (exclusive claims, no double-booking).
        for seed in [1, 7, 42, 1234] {
            let targets = twin_cities(seed);
            for t in &targets {
                let t = t.expect("twin seeks a stall");
                assert!(t == (10.0, 0.0) || t == (-10.0, 0.0));
            }
            assert_ne!(targets[0], targets[1]);
        }
    }

    #[test]
    fn money_accrues_sub_cent_fractions() {
        let mut m = Money::default();
        // $36/h at 10Hz sim ticks = 0.1¢/tick: ten ticks make a cent.
        for _ in 0..10 {
            m.accrue(36.0 / 3600.0 * 0.1);
        }
        assert_eq!(m.cents, 1);
        assert_eq!(fmt_cents(205), "$2.05");
        assert_eq!(fmt_cents(2000), "$20.00");
    }

    #[test]
    fn wall_forces_detour_but_arrives() {
        let mut sim = Sim::new();
        sim.add_wall(5.0, -10.0, 5.0, 10.0);
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        let route = sim.world.get::<Route>(agent).unwrap();
        assert!(!route.0.is_empty());
        // Detour leaves the wall's z-range on one side.
        assert!(route.0.iter().any(|p| p[1].abs() > 10.0));
        for _ in 0..600 {
            sim.step(0.1);
        }
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<Needs>(agent).unwrap().hunger > 0.4);
    }

    #[test]
    fn wall_gap_is_a_door() {
        let mut sim = Sim::new();
        // 4u gap (one free cell row): segments end on cell boundaries,
        // the middle row stays walkable.
        sim.add_wall(5.0, -10.0, 5.0, -2.0);
        sim.add_wall(5.0, 2.0, 5.0, 10.0);
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        let route = sim.world.get::<Route>(agent).unwrap();
        assert!(route.0.iter().any(|p| p[1].abs() < 2.0));
        for _ in 0..60 {
            sim.step(0.1);
        }
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
    }

    #[test]
    fn walled_box_is_unreachable() {
        let mut sim = Sim::new();
        sim.set_walls(vec![
            Wall { ax: 4.0, az: -6.0, bx: 4.0, bz: 6.0 },
            Wall { ax: 16.0, az: -6.0, bx: 16.0, bz: 6.0 },
            Wall { ax: 4.0, az: -6.0, bx: 16.0, bz: -6.0 },
            Wall { ax: 4.0, az: 6.0, bx: 16.0, bz: 6.0 },
        ]);
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        sim.step(0.1);
        assert_eq!(sim.world.get::<AgentState>(agent), Some(&AgentState::Idle));
        assert!(sim.world.get::<WalkTarget>(agent).is_none());
    }

    #[test]
    fn snapshot_preserves_walls() {
        let mut sim = Sim::new();
        sim.add_wall(1.0, 2.0, 3.0, 4.0);
        let snap = sim.snapshot();
        assert_eq!(snap.walls.len(), 1);
        let mut sim2 = Sim::new();
        sim2.restore(&snap);
        assert_eq!(sim.snapshot(), sim2.snapshot());
    }

    fn box_room() -> Sim {
        let mut sim = Sim::new();
        sim.set_walls(vec![
            Wall { ax: 0.0, az: 0.0, bx: 20.0, bz: 0.0 },
            Wall { ax: 20.0, az: 0.0, bx: 20.0, bz: 20.0 },
            Wall { ax: 20.0, az: 20.0, bx: 0.0, bz: 20.0 },
            Wall { ax: 0.0, az: 20.0, bx: 0.0, bz: 0.0 },
        ]);
        sim.rebuild_rooms();
        sim
    }

    #[test]
    fn closed_walls_detect_one_room() {
        let sim = box_room();
        let rooms = sim.rooms();
        assert_eq!(rooms.len(), 1);
        let r = &rooms[0];
        assert!((r.cx - 10.0).abs() < 1.0 && (r.cz - 10.0).abs() < 1.0);
        assert!(r.cells > 40, "cells: {}", r.cells);
    }

    #[test]
    fn open_u_shape_detects_no_room() {
        let mut sim = Sim::new();
        sim.set_walls(vec![
            Wall { ax: 0.0, az: 0.0, bx: 0.0, bz: 20.0 },
            Wall { ax: 0.0, az: 0.0, bx: 20.0, bz: 0.0 },
            Wall { ax: 20.0, az: 0.0, bx: 20.0, bz: 20.0 },
        ]);
        sim.rebuild_rooms();
        assert!(sim.rooms().is_empty());
    }

    #[test]
    fn two_boxes_detect_two_rooms() {
        let mut sim = box_room();
        let mut walls = sim.world.resource::<Walls>().clone().0;
        walls.extend([
            Wall { ax: 40.0, az: 40.0, bx: 50.0, bz: 40.0 },
            Wall { ax: 50.0, az: 40.0, bx: 50.0, bz: 50.0 },
            Wall { ax: 50.0, az: 50.0, bx: 40.0, bz: 50.0 },
            Wall { ax: 40.0, az: 50.0, bx: 40.0, bz: 40.0 },
        ]);
        sim.set_walls(walls);
        sim.rebuild_rooms();
        assert_eq!(sim.rooms().len(), 2);
    }

    #[test]
    fn furniture_advertises_by_kind() {
        let mut sim = Sim::new();
        let fridge = sim.spawn_furniture(0.0, 0.0, FurnitureKind::Fridge);
        let bed = sim.spawn_furniture(5.0, 0.0, FurnitureKind::Bed);
        let sofa = sim.spawn_furniture(10.0, 0.0, FurnitureKind::Sofa);
        assert_eq!(sim.world.get::<Goal>(fridge).unwrap().primary(), NeedKind::Hunger);
        assert_eq!(sim.world.get::<Goal>(bed).unwrap().primary(), NeedKind::Energy);
        assert_eq!(sim.world.get::<Goal>(sofa).unwrap().primary(), NeedKind::Sociability);
        assert_eq!(sim.world.get::<Furniture>(sofa), Some(&Furniture(FurnitureKind::Sofa)));
    }

    #[test]
    fn hungry_agent_uses_fridge() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_furniture(8.0, 0.0, FurnitureKind::Fridge);
        sim.step(0.1);
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::SeekGoal)
        );
        for _ in 0..60 {
            sim.step(0.1);
        }
        assert!(sim.world.get::<Needs>(agent).unwrap().hunger > 0.4);
    }

    fn furnished_box() -> Sim {
        let mut sim = Sim::new();
        sim.set_walls(vec![
            Wall { ax: 0.0, az: 0.0, bx: 20.0, bz: 0.0 },
            Wall { ax: 20.0, az: 0.0, bx: 20.0, bz: 20.0 },
            Wall { ax: 20.0, az: 20.0, bx: 0.0, bz: 20.0 },
            Wall { ax: 0.0, az: 20.0, bx: 0.0, bz: 0.0 },
        ]);
        sim.rebuild_rooms();
        sim.spawn_furniture(8.0, 10.0, FurnitureKind::Bed);
        sim.spawn_furniture(12.0, 10.0, FurnitureKind::Bed);
        sim.spawn_furniture(10.0, 14.0, FurnitureKind::Fridge);
        sim
    }

    #[test]
    fn room_function_voted_from_furniture() {
        let mut sim = furnished_box();
        assert_eq!(sim.rooms().len(), 1);
        assert_eq!(sim.room_functions(), vec![Some(RoomFunction::Bedroom)]);
    }

    #[test]
    fn empty_room_stays_unassigned() {
        let mut sim = Sim::new();
        sim.set_walls(vec![
            Wall { ax: 0.0, az: 0.0, bx: 20.0, bz: 0.0 },
            Wall { ax: 20.0, az: 0.0, bx: 20.0, bz: 20.0 },
            Wall { ax: 20.0, az: 20.0, bx: 0.0, bz: 20.0 },
            Wall { ax: 0.0, az: 20.0, bx: 0.0, bz: 0.0 },
        ]);
        sim.rebuild_rooms();
        assert_eq!(sim.room_functions(), vec![None]);
    }

    #[test]
    fn tied_room_stays_unassigned() {
        let mut sim = Sim::new();
        sim.set_walls(vec![
            Wall { ax: 0.0, az: 0.0, bx: 20.0, bz: 0.0 },
            Wall { ax: 20.0, az: 0.0, bx: 20.0, bz: 20.0 },
            Wall { ax: 20.0, az: 20.0, bx: 0.0, bz: 20.0 },
            Wall { ax: 0.0, az: 20.0, bx: 0.0, bz: 0.0 },
        ]);
        sim.rebuild_rooms();
        sim.spawn_furniture(8.0, 10.0, FurnitureKind::Bed);
        sim.spawn_furniture(12.0, 10.0, FurnitureKind::Sofa);
        assert_eq!(sim.room_functions(), vec![None]);
    }

    #[test]
    fn new_needs_decay_at_own_rates() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.step(1.0);
        let needs = sim.world.get::<Needs>(agent).unwrap();
        assert!((needs.comfort - 0.79).abs() < 1e-6);
        assert!((needs.bladder - (0.8 - 0.015)).abs() < 1e-6);
        assert!((needs.fun - (0.8 - 0.008)).abs() < 1e-6);
    }

    #[test]
    fn desperate_agent_uses_toilet() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().bladder = 0.05;
        sim.spawn_furniture(8.0, 0.0, FurnitureKind::Toilet);
        sim.step(0.1);
        assert_eq!(
            sim.world.get::<AgentState>(agent),
            Some(&AgentState::SeekGoal)
        );
        for _ in 0..60 {
            sim.step(0.1);
        }
        assert!(sim.world.get::<Needs>(agent).unwrap().bladder > 0.4);
    }

    #[test]
    fn bathroom_voted_from_plumbing() {
        let mut sim = Sim::new();
        sim.set_walls(vec![
            Wall { ax: 0.0, az: 0.0, bx: 20.0, bz: 0.0 },
            Wall { ax: 20.0, az: 0.0, bx: 20.0, bz: 20.0 },
            Wall { ax: 20.0, az: 20.0, bx: 0.0, bz: 20.0 },
            Wall { ax: 0.0, az: 20.0, bx: 0.0, bz: 0.0 },
        ]);
        sim.rebuild_rooms();
        sim.spawn_furniture(8.0, 10.0, FurnitureKind::Toilet);
        sim.spawn_furniture(12.0, 10.0, FurnitureKind::Tub);
        sim.spawn_furniture(10.0, 14.0, FurnitureKind::Bed);
        assert_eq!(sim.room_functions(), vec![Some(RoomFunction::Bathroom)]);
    }

    #[test]
    fn tv_counts_as_living() {
        let mut sim = Sim::new();
        sim.set_walls(vec![
            Wall { ax: 0.0, az: 0.0, bx: 20.0, bz: 0.0 },
            Wall { ax: 20.0, az: 0.0, bx: 20.0, bz: 20.0 },
            Wall { ax: 20.0, az: 20.0, bx: 0.0, bz: 20.0 },
            Wall { ax: 0.0, az: 20.0, bx: 0.0, bz: 0.0 },
        ]);
        sim.rebuild_rooms();
        sim.spawn_furniture(10.0, 10.0, FurnitureKind::TV);
        assert_eq!(sim.room_functions(), vec![Some(RoomFunction::Living)]);
    }

    #[test]
    fn playful_agent_prefers_fun_over_equal_hunger() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        // Craft the personality: max playful, min everything else.
        *sim.world.get_mut::<Personality>(agent).unwrap() = Personality {
            playful: 1.0,
            outgoing: 0.0,
            active: 0.0,
        };
        {
            let mut needs = sim.world.get_mut::<Needs>(agent).unwrap();
            needs.hunger = 0.3;
            needs.fun = 0.3;
        }
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        sim.spawn_goal(-10.0, 0.0, NeedKind::Fun);
        sim.step(0.1);
        let t = sim.world.get::<WalkTarget>(agent).unwrap();
        assert_eq!((t.x, t.z), (-10.0, 0.0));
    }

    #[test]
    fn cooking_skill_grows_and_boosts_meals() {
        let mut sim = Sim::new();
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.spawn_goal(1.0, 0.0, NeedKind::Hunger); // within reach
        sim.step(0.1);
        assert!(sim.world.get::<Skills>(agent).unwrap().cooking > 0.0);
        // Skilled second meal restores more than the unskilled first.
        sim.world.get_mut::<Needs>(agent).unwrap().hunger = 0.1;
        sim.world.get_mut::<Skills>(agent).unwrap().cooking = 1.0;
        sim.step(0.1);
        assert!(sim.world.get::<Needs>(agent).unwrap().hunger > 0.9);
    }

    #[test]
    fn spawn_personalities_vary_but_replay() {
        // Same seed: same personalities (deterministic DNA).
        let mut a = Sim::new();
        a.set_seed(99);
        let mut b = Sim::new();
        b.set_seed(99);
        let ea = a.spawn_agent(0.0, 0.0);
        let eb = b.spawn_agent(0.0, 0.0);
        assert_eq!(
            a.world.get::<Personality>(ea),
            b.world.get::<Personality>(eb)
        );
        // Fresh spawns actually differ across a population.
        let mut sim = Sim::new();
        sim.set_seed(3);
        let mut playful = 0u32;
        for _ in 0..6 {
            let e = sim.spawn_agent(0.0, 0.0);
            if sim.world.get::<Personality>(e).unwrap().playful > 0.5 {
                playful += 1;
            }
        }
        assert!((1..6).contains(&playful), "playful count: {playful}");
    }
}
