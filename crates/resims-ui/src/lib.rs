#![allow(non_snake_case)]

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use repose_core::{
    Color, FocusRequester, FontWeight, Modifier, PaddingValues, Rect, Scheduler, View, remember,
    remember_mutable_with_key, remember_state_with_key,
};
use repose_core::input::{Key, KeyEvent, KeyEventType};
use repose_core::locals::{px_to_dp, with_content_color};
use repose_platform::RenderContext;
use repose_ui::{Box as ReposeBox, Column, Row, Spacer, Text, TextStyle, ViewExt, ZStack};
use repose_ui::scroll::{ScrollArea, remember_scroll_state};
use repose_core::request_frame;
use resims_sim::{AgentState as SimAgentState, ActionKind, CityBuilding, Demand, Dwelling, Employment, Entity, Furniture, FurnitureKind, Goal, Home, Ledger, NeedKind, Needs, Obstacle, Personality, Position, Sim, Wall as SimWall, Workplace, ZoneFunction, DEFAULT_ATTENUATION, SAVE_VERSION, fmt_cents, outfit_for};
use game_utils::Storage;
use game_utils::save::SaveManager;
use game_utils::save_store::LoadStatus;
use resims_audio::Audio;
use resims_view3d::{AgentMarker, GroundMarker, PathLine, PathPoly, PickEvent, PropBox, Viewport3d, ViewportInput, WallSeg, BLOCKS, block_by_id};

type SessionRef = Rc<RefCell<UiState>>;

#[derive(Clone)]
struct UiState {
    show_menu: bool,
    menu_tab: usize,
    show_debug: bool,
    debug_sec_actions: bool,
    debug_sec_net: bool,
    debug_sec_log: bool,
    debug_sec_sim: bool,
    debug_grid_n: i32,
    debug_grid_lanes: i32,
    debug_grid_spacing: i32,
    debug_spawn_tries: i32,
    rendering_enabled: bool,
    sim_hour: u32,
    sim_min: u32,
    sim_speed_log: f32, // 0 = paused, else speed = 2^(log-1); marks || 1x 4x 32x
    ui_mode: UiMode,
    planning_mode: Option<PlanningMode>,
    current_project: Option<String>,
    projects: Vec<String>,
    project_seq: u32,
    has_redo: bool,
    selected_land_use: Option<LandUse>,
    selected_furniture: Option<FurnitureKind>,
    inspected_building: Option<String>,
    building_pinned: bool,
    /// Last pick cursor position (viewport px) for window anchoring.
    building_anchor: Option<[f32; 2]>,
    hovered_window: Option<HoverWindow>,
    // --- live sim (headless bevy_ecs, stepped in the frame loop) ---
    sim: Rc<RefCell<Sim>>,
    sim_seeded: bool,
    sim_secs: f32,
    last_frame: Option<Instant>,
    /// Ground points of the current planning project (road/zone clicks).
    project_points: Vec<[f32; 2]>,
    /// Agent selected in the roster for direct orders (click-to-direct).
    selected_agent: Option<Entity>,
    /// Stamped roads: survive Implement, rendered in asphalt grey.
    built_roads: Vec<Vec<[f32; 2]>>,
    /// Built walls: survive Implement, rendered as boxes + block routing.
    built_walls: Vec<[[f32; 2]; 2]>,
    /// Zoned buildings: block id + function + generated sim entities
    /// (re-stamping despawns the old set first).
    stamped: Vec<StampedBuilding>,
    /// Last save/load message for the Game tab.
    last_save_msg: String,
    /// UI blips. `None` on platforms without output / when init fails.
    audio: Rc<Option<Audio>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UiMode {
    None,
    Inspection,
    Planning,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlanningMode {
    Roads,
    Zoning,
    Walls,
    Furniture,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LandUse {
    Residential,
    Commercial,
    Industrial,
    Agricultural,
    Recreational,
    Administrative,
}

/// Land-use choice maps 1:1 onto the sim zone function.
fn land_use_function(lu: LandUse) -> ZoneFunction {
    match lu {
        LandUse::Residential => ZoneFunction::Residential,
        LandUse::Commercial => ZoneFunction::Commercial,
        LandUse::Industrial => ZoneFunction::Industrial,
        LandUse::Agricultural => ZoneFunction::Agricultural,
        LandUse::Recreational => ZoneFunction::Recreational,
        LandUse::Administrative => ZoneFunction::Administrative,
    }
}

/// A stamped building: block id + the sim entities the stamp generated
/// (despawned first on re-stamp). Function lives on the sim
/// [`CityBuilding`]; this is the UI-side ownership record.
#[derive(Clone, Debug)]
struct StampedBuilding {
    id: String,
    generated: Vec<Entity>,
}

/// Ray-cast point-in-polygon (zoning drafts assign blocks by center).
fn point_in_polygon(p: [f32; 2], poly: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, zi) = (poly[i][0], poly[i][1]);
        let (xj, zj) = (poly[j][0], poly[j][1]);
        if (zi > p[1]) != (zj > p[1]) {
            let x_int = xi + (p[1] - zi) * (xj - xi) / (zj - zi);
            if p[0] < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HoverWindow {
    Menu,
    Debug,
    Building,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            show_menu: true,
            menu_tab: 0,
            show_debug: false,
            debug_sec_actions: false,
            debug_sec_net: false,
            debug_sec_log: false,
            debug_sec_sim: false,
            debug_grid_n: 10,
            debug_grid_lanes: 2,
            debug_grid_spacing: 200,
            debug_spawn_tries: 50,
            rendering_enabled: true,
            sim_hour: 0,
            sim_min: 0,
            sim_speed_log: 1.0, // 1x, like Time.initialState speed: 1
            ui_mode: UiMode::None,
            planning_mode: None,
            current_project: None,
            projects: Vec::new(),
            project_seq: 0,
            has_redo: false,
            selected_land_use: None,
            selected_furniture: None,
            inspected_building: None,
            building_pinned: false,
            building_anchor: None,
            hovered_window: None,
            sim: Rc::new(RefCell::new(Sim::new())),
            sim_seeded: false,
            sim_secs: 0.0,
            last_frame: None,
            project_points: Vec::new(),
            selected_agent: None,
            built_roads: Vec::new(),
            built_walls: Vec::new(),
            stamped: Vec::new(),
            last_save_msg: String::new(),
            audio: Rc::new(Audio::try_init().ok()),
        }
    }
}

// ---------------------------------------------------------------------------
// Live sim: stepped once per composed frame while unpaused. The speed
// slider doubles as the time control (0 = paused, else 2^(log-1)x).
// ---------------------------------------------------------------------------

/// Sim-seconds per real second for a slider position (marks || 1x 4x 32x).
fn sim_speed_multiplier(log: f32) -> f32 {
    if log <= 0.0 {
        0.0
    } else {
        2f32.powf(log - 1.0)
    }
}

fn clock_hm(total_secs: f32) -> (u32, u32) {
    let mins = (total_secs / 60.0).floor().max(0.0) as u32;
    ((mins / 60) % 24, mins % 60)
}

impl UiState {
    /// Confirm blip; silent when audio is unavailable.
    fn blip(&self, freq_hz: f32) {
        if let Some(audio) = self.audio.as_ref() {
            audio.blip(freq_hz);
        }
    }
}

/// repose `Color` (sRGB bytes) to unit-triplet for the 3D pass, which
/// outputs authored values raw.
fn rgb_to_unit(c: Color) -> [f32; 3] {
    [c.0 as f32 / 255.0, c.1 as f32 / 255.0, c.2 as f32 / 255.0]
}

/// Click radius (ground units) for snapping a direct order to a goal.
const USE_RADIUS: f32 = 3.0;

/// Direct the selected agent: a click near a goal sends Use, anywhere
/// else Goto. No-op without a selection. Returns true when sent.
fn direct_order(st: &mut UiState, x: f32, z: f32) -> bool {
    let agent = match st.selected_agent {
        Some(e) => e,
        None => return false,
    };
    let goal = {
        let mut sim = st.sim.borrow_mut();
        let mut q = sim.world.query::<(Entity, &Goal)>();
        q.iter_mut(&mut sim.world)
            .map(|(e, g)| (e, (g.x - x).powi(2) + (g.z - z).powi(2)))
            .filter(|(_, d2)| *d2 <= USE_RADIUS * USE_RADIUS)
            .min_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(e, _)| e)
    };
    {
        let mut sim = st.sim.borrow_mut();
        match goal {
            Some(g) => sim.order(agent, ActionKind::Use { goal: g }),
            None => sim.order(agent, ActionKind::Goto { x, z }),
        }
    }
    st.blip(660.0);
    true
}

/// Move in/out via the roster: clicking near a dwelling toggles the
/// selected agent's home there (full houses refuse with a low blip).
/// Returns true when it handled the click.
fn maybe_move_in(st: &mut UiState, x: f32, z: f32) -> bool {
    let agent = match st.selected_agent {
        Some(e) => e,
        None => return false,
    };
    let home = {
        let mut sim = st.sim.borrow_mut();
        let mut q = sim.world.query::<(Entity, &Dwelling)>();
        q.iter_mut(&mut sim.world)
            .map(|(e, d)| (e, (d.x - x).powi(2) + (d.z - z).powi(2)))
            .filter(|(_, d2)| *d2 <= USE_RADIUS * USE_RADIUS)
            .min_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(e, _)| e)
    };
    let home = match home {
        Some(h) => h,
        None => return false,
    };
    {
        let mut sim = st.sim.borrow_mut();
        let already = sim.world.get::<Home>(agent).is_some_and(|h| h.0 == home);
        if already {
            sim.move_out(agent);
            st.blip(392.0);
        } else if sim.move_in(agent, home) {
            st.blip(740.0);
        } else {
            st.blip(220.0); // full house
        }
    }
    true
}

/// Employ/dismiss via the roster: clicking near a workplace toggles the
/// selected agent's job there. Returns true when it handled the click.
fn maybe_employ(st: &mut UiState, x: f32, z: f32) -> bool {
    let agent = match st.selected_agent {
        Some(e) => e,
        None => return false,
    };
    let site = {
        let mut sim = st.sim.borrow_mut();
        let mut q = sim.world.query::<(Entity, &Workplace)>();
        q.iter_mut(&mut sim.world)
            .map(|(e, w)| (e, (w.x - x).powi(2) + (w.z - z).powi(2)))
            .filter(|(_, d2)| *d2 <= USE_RADIUS * USE_RADIUS)
            .min_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(e, _)| e)
    };
    let site = match site {
        Some(s) => s,
        None => return false,
    };
    {
        let mut sim = st.sim.borrow_mut();
        let employed_here = sim
            .world
            .get::<Employment>(agent)
            .is_some_and(|e| e.workplace == site);
        if employed_here {
            sim.dismiss(agent);
            st.blip(392.0);
        } else {
            sim.employ(agent, site);
            st.blip(740.0);
        }
    }
    true
}

/// Stamp the current roads draft as a persistent road, then clear the
/// project (shared by the Implement button and its shortcut).
fn implement_project(st: &mut UiState) {
    if st.planning_mode == Some(PlanningMode::Roads) && st.project_points.len() >= 2 {
        st.built_roads.push(std::mem::take(&mut st.project_points));
        // The sim routes development off the stamped network.
        st.sim.borrow_mut().set_roads(st.built_roads.clone());
    } else if st.planning_mode == Some(PlanningMode::Walls) && st.project_points.len() >= 2 {
        // Consecutive draft points become wall segments (chains draw
        // rooms; leave a 2u+ gap for doors).
        for w in st.project_points.windows(2) {
            st.built_walls.push([w[0], w[1]]);
        }
        st.project_points.clear();
        st.sim.borrow_mut().set_walls(
            st.built_walls
                .iter()
                .map(|[a, b]| SimWall { ax: a[0], az: a[1], bx: b[0], bz: b[1] })
                .collect(),
        );
        st.sim.borrow_mut().rebuild_rooms();
    } else if st.planning_mode == Some(PlanningMode::Furniture) && !st.project_points.is_empty() {
        // Every draft point places one item of the selected kind.
        // Furniture persists in the sim (goals snapshot covers it).
        if let Some(kind) = st.selected_furniture {
            let mut sim = st.sim.borrow_mut();
            for [x, z] in st.project_points.clone() {
                sim.spawn_furniture(x, z, kind);
            }
        }
        st.project_points.clear();
    } else if st.planning_mode == Some(PlanningMode::Zoning)
        && st.project_points.len() >= 3
    {
        if let Some(lu) = st.selected_land_use {
            stamp_zoning(st, land_use_function(lu));
        }
        st.project_points.clear();
    } else {
        st.project_points.clear();
    }
    st.current_project = None;
    st.planning_mode = None;
    st.selected_land_use = None;
    st.has_redo = false;
    st.inspected_building = None;
    st.building_pinned = false;
    st.building_anchor = None;
}

/// Stamp a zoning draft: every block whose center falls in the polygon
/// gets the function (re-stamping replaces the old function set), plus
/// the function-appropriate sim entities at its front door (west side,
/// off the footprint so agents and clicks can reach them).
fn door_point(min_x: f32, min_z: f32, max_z: f32) -> [f32; 2] {
    [min_x - 2.0, (min_z + max_z) / 2.0]
}

fn stamp_zoning(st: &mut UiState, function: ZoneFunction) {
    let poly = st.project_points.clone();
    let hits: Vec<(String, f32, f32, f32, f32)> = BLOCKS
        .iter()
        .filter(|b| point_in_polygon([b.cx, b.cz], &poly))
        .map(|b| {
            (
                b.id.to_string(),
                b.cx - b.w / 2.0,
                b.cz - b.d / 2.0,
                b.cx + b.w / 2.0,
                b.cz + b.d / 2.0,
            )
        })
        .collect();
    let mut sim = st.sim.borrow_mut();
    for (id, min_x, min_z, max_x, max_z) in hits {
        // Re-stamp: despawn the previous function set first.
        if let Some(old) = st.stamped.iter().find(|s| s.id == id) {
            for e in old.generated.clone() {
                sim.world.despawn(e);
            }
        }
        st.stamped.retain(|s| s.id != id);
        let building = sim.spawn_building(
            id.clone(),
            function,
            min_x,
            min_z,
            max_x,
            max_z,
        );
        let [dx, dz] = door_point(min_x, min_z, max_z);
        let capacity = sim.world.get::<CityBuilding>(building).map(|b| b.capacity).unwrap_or(1);
        let mut generated = vec![building];
        match function {
            ZoneFunction::Commercial => {
                generated.push(sim.spawn_workplace(dx, dz, 30.0, 9, 17));
                generated.push(sim.spawn_goal(dx, dz, NeedKind::Comfort));
            }
            ZoneFunction::Industrial => {
                generated.push(sim.spawn_workplace(dx, dz, 38.0, 6, 14));
            }
            ZoneFunction::Recreational => {
                generated.push(sim.spawn_goal_ads(
                    dx,
                    dz,
                    vec![(NeedKind::Fun, 0.7), (NeedKind::Energy, 0.3)],
                    DEFAULT_ATTENUATION,
                ));
            }
            ZoneFunction::Agricultural => {
                generated.push(sim.spawn_goal(dx, dz, NeedKind::Hunger));
            }
            ZoneFunction::Administrative => {
                generated.push(sim.spawn_goal(dx, dz, NeedKind::Sociability));
            }
            ZoneFunction::Residential => {
                generated.push(sim.spawn_dwelling(dx, dz, capacity));
            }
        }
        st.stamped.push(StampedBuilding { id, generated });
    }
}

/// Save manager: crash-safe RON store under the platform data dir
/// (OPFS on wasm). Same file the Game tab buttons use.
fn save_manager() -> SaveManager {
    SaveManager::new("com", "resims", "resims", "save.ron", SAVE_VERSION)
}

/// Write the full game: sim snapshot + city edits (roads, walls,
/// stamp functions).
fn do_save<S: Storage>(st: &mut UiState, mgr: &SaveManager<S>) {
    let roads = st.built_roads.clone();
    let walls: Vec<SimWall> = st
        .built_walls
        .iter()
        .map(|[a, b]| SimWall { ax: a[0], az: a[1], bx: b[0], bz: b[1] })
        .collect();
    let stamped: Vec<(String, ZoneFunction)> = {
        let mut sim = st.sim.borrow_mut();
        let mut q = sim.world.query::<&CityBuilding>();
        q.iter_mut(&mut sim.world)
            .map(|b| (b.label.clone(), b.function))
            .collect()
    };
    match st.sim.borrow_mut().save_game(mgr, roads, walls, stamped) {
        Ok(()) => {
            st.last_save_msg = "Saved.".to_string();
            st.blip(740.0);
        }
        Err(e) => st.last_save_msg = format!("Save failed: {e}"),
    }
}

/// Re-resolve a stamp record after load: the building entity by label
/// plus any goal/workplace/dwelling at its front door.
fn resolve_stamp(st: &UiState, id: &str) -> Vec<Entity> {
    let door = block_by_id(id).map(|b| {
        door_point(b.cx - b.w / 2.0, b.cz - b.d / 2.0, b.cz + b.d / 2.0)
    });
    let mut sim = st.sim.borrow_mut();
    let mut out = Vec::new();
    {
        let mut q = sim.world.query::<(Entity, &CityBuilding)>();
        out.extend(
            q.iter_mut(&mut sim.world)
                .filter(|(_, b)| b.label == id)
                .map(|(e, _)| e),
        );
    }
    if let Some([dx, dz]) = door {
        let near = |x: f32, z: f32| {
            (x - dx).powi(2) + (z - dz).powi(2) <= USE_RADIUS * USE_RADIUS
        };
        let mut q = sim.world.query::<(Entity, &Goal)>();
        out.extend(
            q.iter_mut(&mut sim.world)
                .filter(|(_, g)| near(g.x, g.z))
                .map(|(e, _)| e),
        );
        let mut q = sim.world.query::<(Entity, &Workplace)>();
        out.extend(
            q.iter_mut(&mut sim.world)
                .filter(|(_, w)| near(w.x, w.z))
                .map(|(e, _)| e),
        );
        let mut q = sim.world.query::<(Entity, &Dwelling)>();
        out.extend(
            q.iter_mut(&mut sim.world)
                .filter(|(_, d)| near(d.x, d.z))
                .map(|(e, _)| e),
        );
    }
    out
}

/// Load the full game: sim snapshot plus city edits. Jobs, homes,
/// queues and claims reset (entity links don't survive saves).
fn do_load<S: Storage>(st: &mut UiState, mgr: &SaveManager<S>) {
    let (file, status) = st.sim.borrow_mut().load_game(mgr);
    match status {
        LoadStatus::Ok => {
            st.built_roads = file.roads.clone();
            st.built_walls = file
                .walls
                .iter()
                .map(|w| [[w.ax, w.az], [w.bx, w.bz]])
                .collect();
            st.project_points.clear();
            st.current_project = None;
            st.planning_mode = None;
            st.selected_land_use = None;
            st.selected_agent = None;
            st.inspected_building = None;
            st.building_pinned = false;
            let ids: Vec<String> = file.stamped.iter().map(|(id, _)| id.clone()).collect();
            let stamped: Vec<StampedBuilding> = ids
                .iter()
                .map(|id| StampedBuilding { id: id.clone(), generated: resolve_stamp(st, id) })
                .collect();
            st.stamped = stamped;
            st.last_save_msg = format!(
                "Loaded day {}. Jobs, homes and queues reset.",
                file.snapshot.day
            );
            st.blip(880.0);
        }
        LoadStatus::Missing => st.last_save_msg = "No save yet.".to_string(),
        _ => st.last_save_msg = "Save unreadable; city untouched.".to_string(),
    }
}

fn seed_sim(st: &mut UiState) {
    let mut sim = st.sim.borrow_mut();
    for (x, z) in [
        (-6.0, 4.0),
        (4.0, -2.0),
        (0.0, 10.0),
        (-10.0, -14.0),
        (14.0, 6.0),
        (2.0, -12.0),
    ] {
        sim.spawn_agent(x, z);
    }
    sim.spawn_goal(-20.0, 0.0, NeedKind::Hunger);
    sim.spawn_goal(20.0, -6.0, NeedKind::Energy);
    sim.spawn_goal(0.0, 20.0, NeedKind::Sociability);
    // Old-town services: every need has a pre-build source.
    sim.spawn_goal(8.0, -20.0, NeedKind::Bladder);
    sim.spawn_goal(-8.0, 20.0, NeedKind::Hygiene);
    sim.spawn_goal(0.0, -20.0, NeedKind::Comfort);
    sim.spawn_goal(14.0, 20.0, NeedKind::Fun);
    // Job sites: office day shift, café evening shift.
    sim.spawn_workplace(20.0, -20.0, 40.0, 9, 17);
    sim.spawn_workplace(-24.0, 8.0, 25.0, 12, 20);
    // Agents route around buildings: feed footprints as obstacles.
    sim.set_obstacles(
        BLOCKS
            .iter()
            .map(|b| Obstacle {
                min_x: b.cx - b.w / 2.0,
                min_z: b.cz - b.d / 2.0,
                max_x: b.cx + b.w / 2.0,
                max_z: b.cz + b.d / 2.0,
            })
            .collect(),
    );
}

/// Advance the sim by wall-clock elapsed time. Requests the next frame
/// while running, which is what keeps the loop alive.
fn tick_sim(session: &SessionRef) {
    let speed = sim_speed_multiplier(session.borrow().sim_speed_log);
    let now = Instant::now();
    let dt_real = session
        .borrow()
        .last_frame
        .map(|t| (now - t).as_secs_f32())
        .unwrap_or(0.0)
        .clamp(0.0, 0.25);
    {
        let mut st = session.borrow_mut();
        st.last_frame = Some(now);
        if !st.sim_seeded {
            seed_sim(&mut st);
            st.sim_seeded = true;
        }
    }
    if speed > 0.0 {
        let dt = dt_real * speed;
        session.borrow_mut().sim.borrow_mut().step(dt);
        let mut st = session.borrow_mut();
        st.sim_secs += dt;
        let (h, m) = clock_hm(st.sim_secs);
        st.sim_hour = h;
        st.sim_min = m;
        st.sim.borrow_mut().set_hour(h as u8);
        request_frame();
    }
}


const PRIMARY: Color = Color(0, 72, 255, 255); // @primary-color #0048ff
const GRASS: Color = Color(201, 224, 171, 255); // body bg #c9e0ab
const TOOLBAR_BG: Color = Color(0, 0, 0, 221); // .ui2dTools #000000dd
const ICON_BG: Color = Color(68, 68, 68, 255); // #bbbbbb inverted (inactive)
const ICON_BG_ACTIVE: Color = Color(255, 255, 255, 255);
const HAIRLINE: Color = Color(217, 217, 217, 255); // antd border
const LOG_BG: Color = Color(51, 51, 51, 255); // .scrollableLog #333

/// Land-use swatch colors: colors.js mixes each base 90% over grass.
/// Precomputed to sRGB (see colors.js mix/toLinFloat/fromLinFloat).
fn land_use_color(lu: LandUse) -> Color {
    match lu {
        LandUse::Residential => Color(234, 207, 105, 255),
        LandUse::Commercial => Color(215, 120, 75, 255),
        LandUse::Industrial => Color(135, 103, 114, 255),
        LandUse::Agricultural => Color(149, 151, 124, 255),
        LandUse::Recreational => Color(139, 198, 136, 255),
        LandUse::Administrative => Color(87, 162, 220, 255),
    }
}

#[allow(dead_code)]
fn land_use_name(lu: LandUse) -> &'static str {
    match lu {
        LandUse::Residential => "Residential",
        LandUse::Commercial => "Commercial",
        LandUse::Industrial => "Industrial",
        LandUse::Agricultural => "Agricultural",
        LandUse::Recreational => "Recreational",
        LandUse::Administrative => "Administrative",
    }
}

/// Glyphs approximating the original icons8 PNGs (black line icons).
fn land_use_glyph(lu: LandUse) -> &'static str {
    match lu {
        LandUse::Residential => "⌂",
        LandUse::Commercial => "🏪",
        LandUse::Industrial => "🏭",
        LandUse::Agricultural => "🌾",
        LandUse::Recreational => "🌲",
        LandUse::Administrative => "🏛",
    }
}

const LAND_USES: [LandUse; 6] = [
    LandUse::Residential,
    LandUse::Commercial,
    LandUse::Industrial,
    LandUse::Agricultural,
    LandUse::Recreational,
    LandUse::Administrative,
];

fn short_project_name(id: &str) -> String {
    let head: String = id.chars().take(3).collect::<String>().to_uppercase();
    format!("Project '{head}'")
}

pub fn app(_s: &mut Scheduler, _rc: &RenderContext) -> View {
    let session: SessionRef = repose_core::remember_state_with_key("resims_state", UiState::default);
    with_content_color(Color::BLACK, || ResimsRoot(session))
}

fn ResimsRoot(session: SessionRef) -> View {
    // Step the live sim before composing (keeps requesting frames while
    // unpaused, which is what keeps this loop alive).
    tick_sim(&session);
    let focus = remember(FocusRequester::new);
    let fr_attach = (*focus).clone();
    let fr_init = (*focus).clone();
    let fr_click = (*focus).clone();
    let s_key = session.clone();
    ZStack(
        Modifier::new()
            .fill_max_size()
            .background(GRASS)
            .focusable(true)
            .focus_requester(fr_attach)
            .on_globally_positioned(move |_| {
                fr_init.request_focus();
            })
            .on_key_event(move |ke: KeyEvent| handle_shortcut(&s_key, ke))
            .on_pointer_down(move |_| {
                fr_click.request_focus();
            }),
    )
    .child((
        Canvas3d(session.clone()),
        UiOverlay(session),
    ))
}

/// Global shortcuts from the Settings tab (mirrors the original
/// Mousetrap bindings + PlanningMenu useInputBinding). Returns true
/// when the event was consumed. Pattern follows renamite's
/// `handle_viewport_key`: match on (key, command, shift, alt).
fn handle_shortcut(session: &SessionRef, event: KeyEvent) -> bool {
    if event.event_type != KeyEventType::Down || event.is_repeat {
        return false;
    }
    let m = &event.modifiers;
    match (&event.key, m.command, m.shift, m.alt) {
        (Key::Enter, true, false, false) => {
            let mut st = session.borrow_mut();
            if st.current_project.is_some() {
                implement_project(&mut st);
                st.blip(740.0);
                true
            } else {
                false
            }
        }
        (Key::Character('z'), true, false, false) => {
            if session.borrow().current_project.is_some() {
                session.borrow_mut().has_redo = true;
                true
            } else {
                false
            }
        }
        (Key::Character('z'), true, true, false)
        | (Key::Character('y'), true, false, false) => {
            if session.borrow().has_redo {
                session.borrow_mut().has_redo = false;
                true
            } else {
                false
            }
        }
        (Key::Character('.'), false, false, false) => {
            let v = !session.borrow().show_debug;
            session.borrow_mut().show_debug = v;
            true
        }
        _ => false,
    }
}

/// 3D canvas. Inspection picks flow back from the viewport into UI state;
/// ground clicks extend the current planning project. Agent + planning
/// markers are rebuilt from sim/UI state every frame.
fn Canvas3d(session: SessionRef) -> View {
    let input = {
        let s = session.borrow();
        // Agent avatars from the live sim: outfit/skin identity from
        // personality, floating badge coloured by state.
        let agents: Vec<AgentMarker> = {
            let mut sim = s.sim.borrow_mut();
            let mut q = sim.world.query::<(&Position, &SimAgentState, &Personality)>();
            q.iter_mut(&mut sim.world)
                .map(|(p, st, personality)| {
                    let (outfit, skin) = outfit_for(personality);
                    AgentMarker {
                        x: p.x,
                        z: p.z,
                        outfit,
                        skin,
                        accent: match st {
                            SimAgentState::Idle => [0.45, 0.65, 0.45],
                            SimAgentState::Walk => [0.90, 0.70, 0.30],
                            SimAgentState::SeekGoal => [0.95, 0.55, 0.25],
                            SimAgentState::Socialize => [0.95, 0.55, 0.70],
                            SimAgentState::Working => [0.45, 0.55, 0.95],
                            SimAgentState::Sleeping => [0.35, 0.40, 0.75],
                        },
                    }
                })
                .collect()
        };
        let markers: Vec<GroundMarker> = s
            .project_points
            .iter()
            .map(|[x, z]| GroundMarker {
                x: *x,
                z: *z,
                color: [0.25, 0.45, 1.0],
                size: 1.2,
            })
            // Sim goals as coloured diamonds so agent errands read on canvas.
            // Furniture goals render as boxes instead (see props below).
            .chain({
                let mut sim = s.sim.borrow_mut();
                let mut q = sim.world.query::<(&Goal, Option<&Furniture>)>();
                q.iter_mut(&mut sim.world)
                    .filter(|(_, f)| f.is_none())
                    .map(|(g, _)| GroundMarker {
                        x: g.x,
                        z: g.z,
                        color: match g.primary() {
                            NeedKind::Hunger => [0.95, 0.40, 0.20],
                            NeedKind::Energy => [0.30, 0.60, 0.95],
                            NeedKind::Sociability => [0.35, 0.80, 0.40],
                            NeedKind::Comfort => [0.95, 0.85, 0.45],
                            NeedKind::Hygiene => [0.45, 0.90, 0.90],
                            NeedKind::Bladder => [0.90, 0.60, 0.25],
                            NeedKind::Fun => [0.95, 0.45, 0.75],
                        },
                        size: 1.6,
                    })
                    .collect::<Vec<_>>()
            })
            // Workplaces as gold diamonds (employ via roster click).
            .chain({
                let mut sim = s.sim.borrow_mut();
                let mut q = sim.world.query::<&Workplace>();
                q.iter_mut(&mut sim.world)
                    .map(|w| GroundMarker {
                        x: w.x,
                        z: w.z,
                        color: [0.95, 0.75, 0.20],
                        size: 1.8,
                    })
                    .collect::<Vec<_>>()
            })
            // Dwellings as teal diamonds (move in via roster click).
            .chain({
                let mut sim = s.sim.borrow_mut();
                let mut q = sim.world.query::<&Dwelling>();
                q.iter_mut(&mut sim.world)
                    .map(|d| GroundMarker {
                        x: d.x,
                        z: d.z,
                        color: [0.35, 0.75, 0.85],
                        size: 1.6,
                    })
                    .collect::<Vec<_>>()
            })
            // Selected agent gets a white ring (roster click-to-direct).
            .chain({
                let sim = s.sim.borrow();
                s.selected_agent
                    .and_then(|e| sim.world.get::<Position>(e))
                    .map(|p| GroundMarker { x: p.x, z: p.z, color: [1.0, 1.0, 1.0], size: 2.2 })
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        // Built roads persist in asphalt grey; the live draft stays near-white.
        let mut paths: Vec<PathLine> = s
            .built_roads
            .iter()
            .filter(|pts| pts.len() >= 2)
            .map(|pts| PathLine {
                points: pts.clone(),
                width: 2.0,
                color: [0.55, 0.55, 0.58],
            })
            .collect();
        if s.planning_mode == Some(PlanningMode::Roads) && s.project_points.len() >= 2 {
            paths.push(PathLine {
                points: s.project_points.clone(),
                width: 2.0,
                color: [0.95, 0.95, 0.96],
            });
        }
        // Walls draft draws as a pale chain; built walls are boxes.
        if s.planning_mode == Some(PlanningMode::Walls) && s.project_points.len() >= 2 {
            paths.push(PathLine {
                points: s.project_points.clone(),
                width: 0.6,
                color: [0.98, 0.96, 0.90],
            });
        }
        let walls: Vec<WallSeg> = s
            .built_walls
            .iter()
            .map(|[a, b]| WallSeg {
                ax: a[0],
                az: a[1],
                bx: b[0],
                bz: b[1],
                height: 3.0,
                color: [0.88, 0.85, 0.80],
            })
            .collect();
        // Furniture renders as boxes (size/color by kind).
        let props: Vec<PropBox> = {
            let mut sim = s.sim.borrow_mut();
            let mut q = sim.world.query::<(&Goal, &Furniture)>();
            q.iter_mut(&mut sim.world)
                .map(|(g, f)| {
                    let (w, h, d, color) = furniture_prop(f.0);
                    PropBox { cx: g.x, cz: g.z, w, h, d, color }
                })
                .collect()
        };
        // Zoning draft fills with the selected land-use colour (opaque
        // grass mix, like the original zone layers).
        let polys = match (s.planning_mode, s.selected_land_use) {
            (Some(PlanningMode::Zoning), Some(lu)) if s.project_points.len() >= 3 => {
                vec![PathPoly {
                    points: s.project_points.clone(),
                    color: rgb_to_unit(land_use_color(lu)),
                }]
            }
            _ => Vec::new(),
        };
        ViewportInput {
            inspection: s.ui_mode == UiMode::Inspection,
            agents,
            markers,
            paths,
            polys,
            walls,
            props,
        }
    };
    let s_hover = session.clone();
    let s_select = session.clone();
    let s_click = session;
    Viewport3d(input, move |ev| match ev {
        PickEvent::Hover { id, screen } => {
            if !s_hover.borrow().building_pinned {
                let mut st = s_hover.borrow_mut();
                st.inspected_building = id;
                st.building_anchor = Some(screen);
            }
        }
        PickEvent::Select { id, screen } => {
            let mut st = s_select.borrow_mut();
            st.inspected_building = Some(id);
            st.building_pinned = true;
            st.building_anchor = Some(screen);
            st.blip(880.0);
        }
        PickEvent::GroundClick { x, z } => {
            let mut st = s_click.borrow_mut();
            if st.ui_mode == UiMode::Planning
                && st.current_project.is_some()
                && st.project_points.len() < 64
            {
                st.project_points.push([x, z]);
            } else if !maybe_move_in(&mut st, x, z) && !maybe_employ(&mut st, x, z) {
                // Click-to-direct: selected roster agent takes the order
                // (Use near a goal, Goto elsewhere).
                direct_order(&mut st, x, z);
            }
        }
    })
}


fn UiOverlay(session: SessionRef) -> View {
    let s = session.borrow().clone();
    // The overlay floats over the 3D canvas: the root and all decorative
    // boxes are hit-passthrough (like CSS `pointer-events: none` on .ui2d)
    // so canvas gestures work everywhere except on interactive children
    // (windows, toolbar), which own their hit regions via handlers.
    ZStack(Modifier::new().fill_max_size().hit_passthrough()).child((
        ReposeBox(
            Modifier::new()
                .absolute()
                .offset(None, Some(-2.0), Some(20.0), None)
                .hit_passthrough(),
        )
        .child(
            Column(Modifier::new().gap(0.0)).child(vec![
                Text("resims")
                    .size(24.0)
                    .color(Color(0, 0, 0, 77))
                    .letter_spacing(2.0),
                Text(format!("v{}", env!("CARGO_PKG_VERSION")))
                    .size(9.6)
                    .color(Color(0, 0, 0, 51)),
            ]),
        ),
        ReposeBox(
            Modifier::new()
                .absolute()
                .offset(Some(16.0), Some(16.0), None, None),
        )
        .child(SimTime(session.clone())),
        if s.ui_mode == UiMode::Inspection && s.inspected_building.is_some() {
            // Anchored above the pick cursor (the 3D projection of the
            // building); falls back to a fixed corner without a pick yet.
            let [ax, ay] = s.building_anchor.unwrap_or([136.0, 392.0]);
            ReposeBox(
                Modifier::new()
                    .absolute()
                    .offset(
                        Some((ax - 120.0).max(8.0)),
                        Some((ay - 296.0 - 16.0).max(72.0)),
                        None,
                        None,
                    ),
            )
            .child(BuildingWindow(session.clone()))
        } else {
            ReposeBox(Modifier::new())
        },
        if s.ui_mode == UiMode::Inspection && s.inspected_building.is_none() {
            ReposeBox(
                Modifier::new()
                    .absolute()
                    .offset(None, None, None, Some(80.0))
                    .fill_max_width()
                    .hit_passthrough(),
            )
            .child(
                Row(Modifier::new().fill_max_width().justify_content(
                    repose_core::JustifyContent::CENTER,
                ))
                .child(Text("Drag to orbit • wheel to zoom • hover a building, click to pin").size(11.0).color(
                    Color(0, 0, 0, 130),
                )),
            )
        } else {
            ReposeBox(Modifier::new())
        },
        if s.show_debug {
            ReposeBox(
                Modifier::new()
                    .absolute()
                    .offset(Some(320.0), Some(16.0), None, None),
            )
            .child(DebugWindow(session.clone()))
        } else {
            ReposeBox(Modifier::new())
        },
        if s.show_menu {
            ReposeBox(
                Modifier::new()
                    .absolute()
                    .offset(None, Some(16.0), Some(16.0), Some(72.0)),
            )
            .child(MenuWindow(session.clone()))
        } else {
            ReposeBox(Modifier::new())
        },
        ReposeBox(
            Modifier::new()
                .absolute()
                .offset(None, None, Some(0.0), Some(0.0))
                .fill_max_width()
                .height(64.0),
        )
        .child(BottomToolbar(session.clone())),
    ))
}

fn SimTime(session: SessionRef) -> View {
    let (h, m, funds, day, treasury, demand, ledger) = {
        let s = session.borrow();
        let sim = s.sim.borrow();
        (
            s.sim_hour,
            s.sim_min,
            sim.funds(),
            sim.day(),
            sim.treasury(),
            sim.demand(),
            sim.ledger(),
        )
    };
    let (pop, happy) = city_stats(&session.borrow());
    ReposeBox(Modifier::new().width(144.0).alpha(0.7)).child(
        Column(Modifier::new().gap(4.0)).child(vec![
            Text(format!("{h:02}:{m:02}")).size(17.6).color(Color::BLACK),
            Text(fmt_cents(funds)).size(12.8).color(Color(60, 60, 60, 255)),
            Text(format!("day {day} · city {}", fmt_cents(treasury))).size(12.8).color(Color(60, 60, 60, 255)),
            Text(demand_line(demand)).size(12.8).color(Color(60, 60, 60, 255)),
            Text(ledger_line(ledger)).size(12.8).color(Color(60, 60, 60, 255)),
            Text(format!("{pop} agents · {:.0}%", happy * 100.0)).size(12.8).color(Color(60, 60, 60, 255)),
            SpeedSlider(session),
            Row(Modifier::new().fill_max_width().gap(0.0)).child(vec![
                Text("||").size(11.0).color(Color(90, 90, 90, 255)),
                ReposeBox(Modifier::new().width(10.0)),
                Text("1x").size(11.0).color(Color(90, 90, 90, 255)),
                Spacer(),
                Text("4x").size(11.0).color(Color(90, 90, 90, 255)),
                Spacer(),
                Text("32x").size(11.0).color(Color(90, 90, 90, 255)),
            ]),
        ]),
    )
}

/// City stats for the time panel: population + average need level
/// (0..=1) across all agents.
fn city_stats(st: &UiState) -> (usize, f32) {    let mut sim = st.sim.borrow_mut();
    let mut q = sim.world.query::<&Needs>();
    let mut n = 0usize;
    let mut sum = 0.0;
    for needs in q.iter_mut(&mut sim.world) {
        n += 1;
        sum += needs.mean();
    }
    (n, if n == 0 { 1.0 } else { sum / n as f32 })
}

/// RCI demand as one compact panel line: "R+1.0 C-0.6 I+0.0".
fn demand_line(d: Demand) -> String {
    format!("R{:+.1} C{:+.1} I{:+.1}", d.residential, d.commercial, d.industrial)
}

/// Last settled day as one compact panel line: "$20 in · $8 out".
fn ledger_line(l: Ledger) -> String {
    let income = ((l.property_in + l.wage_in) * 100.0) as i64;
    let out = (l.services_out * 100.0) as i64;
    format!("{} in · {} out", fmt_cents(income), fmt_cents(out))
}

/// Thin antd-like slider: 4dp rail, round thumb, marks handled above.
/// Click/drag on the 112x20 hit area sets sim_speed_log in 0..=6.
fn SpeedSlider(session: SessionRef) -> View {
    let log = session.borrow().sim_speed_log;
    let t = (log / 6.0).clamp(0.0, 1.0);
    let track_rect = remember_state_with_key("speed_track", Rect::default);
    let dragging = remember_mutable_with_key("speed_drag", || false);

    // `track_rect` is dp (layout space); the cursor arrives in physical
    // px, so convert before comparing (same HiDPI rule as the viewport).
    let set_from_x = Rc::new(move |x_px: f32, rect: Rect, session: &SessionRef| {
        if rect.w <= 1.0 {
            return;
        }
        let frac = ((px_to_dp(x_px) - rect.x) / rect.w).clamp(0.0, 1.0);
        session.borrow_mut().sim_speed_log = (frac * 6.0).round().clamp(0.0, 6.0);
    });

    let tr_down = track_rect.clone();
    let s_down = session.clone();
    let set_down = set_from_x.clone();
    let drag_down = dragging.clone();
    let tr_move = track_rect.clone();
    let s_move = session.clone();
    let set_move = set_from_x.clone();
    let drag_move = dragging.clone();
    let drag_up = dragging.clone();
    let drag_cancel = dragging.clone();
    let tr_pos = track_rect.clone();

    ZStack(
        Modifier::new()
            .width(112.0)
            .height(20.0)
            .on_globally_positioned(move |rect| {
                *tr_pos.borrow_mut() = rect;
            })
            .on_pointer_down(move |pe| {
                drag_down.set(true);
                let r = *tr_down.borrow();
                set_down(pe.position_in_window().x, r, &s_down);
            })
            .on_pointer_move(move |pe| {
                if drag_move.with(|v| *v) {
                    let r = *tr_move.borrow();
                    set_move(pe.position_in_window().x, r, &s_move);
                }
            })
            .on_pointer_up(move |_| {
                drag_up.set(false);
            })
            .on_pointer_cancel(move |_| {
                drag_cancel.set(false);
            }),
    )
    .child((
        ReposeBox(
            Modifier::new()
                .absolute()
                .offset(Some(0.0), Some(8.0), None, None)
                .width(112.0)
                .height(4.0)
                .background(Color(217, 217, 217, 255))
                .clip_rounded(2.0),
        ),
        ReposeBox(
            Modifier::new()
                .absolute()
                .offset(Some(0.0), Some(8.0), None, None)
                .width(112.0 * t)
                .height(4.0)
                .background(PRIMARY)
                .clip_rounded(2.0),
        ),
        ReposeBox(
            Modifier::new()
                .absolute()
                .offset(Some(t * (112.0 - 14.0)), Some(3.0), None, None)
                .width(14.0)
                .height(14.0)
                .background(Color::WHITE)
                .border(2.0, PRIMARY, 7.0)
                .clip_rounded(7.0),
        ),
    ))
}


fn BottomToolbar(session: SessionRef) -> View {
    let s = session.borrow().clone();
    let in_planning = s.ui_mode == UiMode::Planning;
    let has_project = s.current_project.is_some();

    let mut items: Vec<View> = Vec::new();
    items.push(MainModeToolbar(session.clone()));
    items.push(AgentsRoster(session.clone()));

    if in_planning {
        if has_project {
            let current = s.current_project.clone().unwrap_or_default();
            items.push(ProjectSelect(session.clone(), current));
            items.push(ImplementButton(session.clone()));
            items.push(HistoryToolbar(session.clone()));
            items.push(PlanningModeToolbar(session.clone()));
            if s.planning_mode == Some(PlanningMode::Zoning) {
                items.push(ZoningToolbar(session.clone()));
            }
            if s.planning_mode == Some(PlanningMode::Furniture) {
                items.push(FurnitureToolbar(session.clone()));
            }
        } else if s.projects.is_empty() {
            items.push(StartProjectButton(session.clone()));
        } else {
            items.push(ProjectSelect(session.clone(), String::new()));
        }
    }

    items.push(Spacer());
    if s.selected_agent.is_some() && !in_planning {
        items.push(
            Text("click ground: move • gold: job • teal: home")
                .size(11.0)
                .color(Color(255, 255, 255, 200)),
        );
    }
    items.push(MenuToolbar(session));

    ReposeBox(
        Modifier::new()
            .fill_max_size()
            .background(TOOLBAR_BG)
            .padding_values(PaddingValues {
                left: 8.0,
                right: 8.0,
                top: 8.0,
                bottom: 8.0,
            }),
    )
    .child(Row(Modifier::new().fill_max_size().gap(8.0)).child(items))
}

/// 48x48 transparent button with an inner icon tile, like .toolbar button
/// + .button-icon (inactive dark tile w/ light glyph, active white w/ dark).
fn ToolbarButton(active: bool, enabled: bool, glyph: impl Into<String>) -> View {
    let (bg, fg) = if active {
        (ICON_BG_ACTIVE, Color::BLACK)
    } else {
        (ICON_BG, Color::WHITE)
    };
    let alpha = if enabled { 1.0 } else { 0.31 };
    Row(Modifier::new()
        .width(48.0)
        .height(48.0)
        .align_items(repose_core::AlignItems::CENTER)
        .justify_content(repose_core::JustifyContent::CENTER)
        .alpha(alpha))
    .child(
        ReposeBox(
            Modifier::new()
                .width(48.0)
                .height(32.0)
                .background(bg)
                .clip_rounded(2.0)
                .border(
                    1.0,
                    if active { Color::TRANSPARENT } else { Color::TRANSPARENT },
                    2.0,
                )
                .padding_values(PaddingValues {
                    left: 0.0,
                    right: 0.0,
                    top: 0.0,
                    bottom: 0.0,
                }),
        )
        .child(
            Row(Modifier::new()
                .fill_max_size()
                .align_items(repose_core::AlignItems::CENTER)
                .justify_content(repose_core::JustifyContent::CENTER))
            .child(Text(glyph.into()).size(18.0).color(fg)),
        ),
    )
}

fn Clickable(view: View, on_click: impl Fn() + 'static) -> View {
    let on_click = Rc::new(on_click);
    ReposeBox(Modifier::new().on_pointer_down(move |ev| {
        // Consume so clicks on UI never leak through to the 3D canvas
        // below (which would start an orbit drag).
        ev.consume();
        on_click();
    }))
    .child(view)
}

fn ClickableEnabled(view: View, enabled: bool, on_click: impl Fn() + 'static) -> View {
    if !enabled {
        return view;
    }
    Clickable(view, on_click)
}


fn MainModeToolbar(session: SessionRef) -> View {
    let mode = session.borrow().ui_mode;
    Row(Modifier::new().gap(2.0)).child(vec![
        Clickable(
            ToolbarButton(mode == UiMode::Inspection, true, "👁"),
            {
                let s = session.clone();
                move || {
                    let mut st = s.borrow_mut();
                    st.ui_mode = if st.ui_mode == UiMode::Inspection {
                        UiMode::None
                    } else {
                        UiMode::Inspection
                    };
                }
            },
        ),
        Clickable(
            ToolbarButton(mode == UiMode::Planning, true, "✎"),
            {
                let s = session.clone();
                move || {
                    let mut st = s.borrow_mut();
                    st.ui_mode = if st.ui_mode == UiMode::Planning {
                        UiMode::None
                    } else {
                        UiMode::Planning
                    };
                }
            },
        ),
    ])
}


fn StartProjectButton(session: SessionRef) -> View {
    Clickable(
        PrimaryButton("Start new project"),
        move || {
            let mut st = session.borrow_mut();
            st.project_seq += 1;
            let id = format!("prj{:03}", st.project_seq);
            st.projects.push(id.clone());
            st.current_project = Some(id);
            st.planning_mode = None;
            st.selected_land_use = None;
            st.has_redo = false;
            st.blip(520.0);
        },
    )
}

fn ProjectSelect(session: SessionRef, current: String) -> View {
    let label = if current.is_empty() {
        "Open a project".to_string()
    } else {
        short_project_name(&current)
    };
    Clickable(
        ReposeBox(
            Modifier::new()
                .width(180.0)
                .height(32.0)
                .background(Color::WHITE)
                .border(1.0, HAIRLINE, 2.0)
                .clip_rounded(2.0)
                .padding_values(PaddingValues {
                    left: 11.0,
                    right: 11.0,
                    top: 0.0,
                    bottom: 0.0,
                }),
        )
        .child(
            Row(Modifier::new()
                .fill_max_size()
                .align_items(repose_core::AlignItems::CENTER))
            .child(vec![
                Text(label)
                    .size(14.0)
                    .color(Color(0, 0, 0, 165))
                    .single_line()
                    .overflow_ellipsize(),
                Spacer(),
                Text("▾").size(12.0).color(Color(0, 0, 0, 140)),
            ]),
        ),
        move || {
            let mut st = session.borrow_mut();
            if st.projects.is_empty() {
                return;
            }
            let idx = st
                .current_project
                .as_ref()
                .and_then(|c| st.projects.iter().position(|p| p == c))
                .map(|i| (i + 1) % st.projects.len())
                .unwrap_or(0);
            st.current_project = Some(st.projects[idx].clone());
        },
    )
}

fn PrimaryButton(label: &str) -> View {
    ReposeBox(
        Modifier::new()
            .height(32.0)
            .background(PRIMARY)
            .clip_rounded(2.0)
            .padding_values(PaddingValues {
                left: 15.0,
                right: 15.0,
                top: 0.0,
                bottom: 0.0,
            }),
    )
    .child(
        Row(Modifier::new()
            .fill_max_size()
            .align_items(repose_core::AlignItems::CENTER)
            .justify_content(repose_core::JustifyContent::CENTER))
        .child(Text(label).size(14.0).color(Color::WHITE).single_line()),
    )
}

fn ImplementButton(session: SessionRef) -> View {
    Clickable(PrimaryButton("Implement"), move || {
        let mut st = session.borrow_mut();
        implement_project(&mut st);
        st.blip(740.0);
    })
}


/// Agent chips in the bottom bar (A1..An + state word). Click selects
/// for click-to-direct (click again to deselect); ground clicks then
/// order the selected agent (Use near a goal, Goto elsewhere).
fn AgentsRoster(session: SessionRef) -> View {
    let agents: Vec<(Entity, String, bool)> = {
        let s = session.borrow();
        let selected = s.selected_agent;
        let mut sim = s.sim.borrow_mut();
        let mut q = sim.world.query::<(Entity, &SimAgentState)>();
        q.iter_mut(&mut sim.world)
            .enumerate()
            .map(|(i, (e, st))| {
                let word = match st {
                    SimAgentState::Idle => "idle",
                    SimAgentState::Walk => "walk",
                    SimAgentState::SeekGoal => "seek",
                    SimAgentState::Socialize => "chat",
                    SimAgentState::Working => "work",
                    SimAgentState::Sleeping => "sleep",
                };
                (e, format!("A{}·{word}", i + 1), selected == Some(e))
            })
            .collect()
    };
    Row(Modifier::new().gap(8.0)).child(
        agents
            .into_iter()
            .map(|(e, label, sel)| {
                let s = session.clone();
                Clickable(ToolbarButton(sel, true, label), move || {
                    let mut st = s.borrow_mut();
                    st.selected_agent =
                        if st.selected_agent == Some(e) { None } else { Some(e) };
                    st.blip(520.0);
                })
            })
            .collect::<Vec<View>>(),
    )
}

fn HistoryToolbar(session: SessionRef) -> View {
    let (has_undo, has_redo) = {
        let s = session.borrow();
        (s.current_project.is_some(), s.has_redo)
    };
    Row(Modifier::new().gap(2.0)).child(vec![
        ClickableEnabled(
            ToolbarButton(false, has_undo, "↩"),
            has_undo,
            {
                let s = session.clone();
                move || {
                    s.borrow_mut().has_redo = true;
                }
            },
        ),
        ClickableEnabled(
            ToolbarButton(false, has_redo, "↪"),
            has_redo,
            {
                let s = session.clone();
                move || {
                    s.borrow_mut().has_redo = false;
                }
            },
        ),
    ])
}


fn PlanningModeToolbar(session: SessionRef) -> View {
    let mode = session.borrow().planning_mode;
    Row(Modifier::new().gap(2.0)).child(vec![
        Clickable(
            ToolbarButton(mode == Some(PlanningMode::Roads), true, "🛣"),
            {
                let s = session.clone();
                move || {
                    s.borrow_mut().planning_mode = Some(PlanningMode::Roads);
                }
            },
        ),
        Clickable(
            ToolbarButton(mode == Some(PlanningMode::Zoning), true, "▦"),
            {
                let s = session.clone();
                move || {
                    let mut st = s.borrow_mut();
                    st.planning_mode = Some(PlanningMode::Zoning);
                    st.selected_land_use = None;
                }
            },
        ),
        Clickable(
            ToolbarButton(mode == Some(PlanningMode::Walls), true, "🧱"),
            {
                let s = session.clone();
                move || {
                    s.borrow_mut().planning_mode = Some(PlanningMode::Walls);
                }
            },
        ),
        Clickable(
            ToolbarButton(mode == Some(PlanningMode::Furniture), true, "🛋"),
            {
                let s = session.clone();
                move || {
                    let mut st = s.borrow_mut();
                    st.planning_mode = Some(PlanningMode::Furniture);
                    if st.selected_furniture.is_none() {
                        st.selected_furniture = Some(FurnitureKind::Fridge);
                    }
                }
            },
        ),
    ])
}


fn ZoningToolbar(session: SessionRef) -> View {
    let selected = session.borrow().selected_land_use;
    Row(Modifier::new().gap(8.0)).child(
        LAND_USES
            .iter()
            .map(|lu| {
                let is_active = Some(*lu) == selected;
                let color = land_use_color(*lu);
                let glyph = land_use_glyph(*lu);
                let s = session.clone();
                let lu_copy = *lu;
                Clickable(
                    ReposeBox(Modifier::new().width(44.0).height(44.0)).child(
                        ReposeBox(
                            Modifier::new()
                                .width(44.0)
                                .height(44.0)
                                .background(color)
                                .clip_rounded(22.0)
                                .border(
                                    if is_active { 3.0 } else { 1.0 },
                                    if is_active {
                                        Color::WHITE
                                    } else {
                                        Color(85, 85, 85, 255)
                                    },
                                    22.0,
                                ),
                        )
                        .child(
                            Row(Modifier::new()
                                .fill_max_size()
                                .align_items(repose_core::AlignItems::CENTER)
                                .justify_content(repose_core::JustifyContent::CENTER))
                            .child(Text(glyph).size(20.0).color(Color::BLACK)),
                        ),
                    ),
                    move || s.borrow_mut().selected_land_use = Some(lu_copy),
                )
            })
            .collect::<Vec<_>>(),
    )
}

/// Furniture dimensions + color by kind (matches view3d PropBox).
fn furniture_prop(kind: FurnitureKind) -> (f32, f32, f32, [f32; 3]) {
    match kind {
        FurnitureKind::Fridge => (1.2, 2.2, 1.2, [0.92, 0.92, 0.94]),
        FurnitureKind::Bed => (2.2, 0.8, 3.0, [0.50, 0.60, 0.90]),
        FurnitureKind::Sofa => (2.4, 1.0, 1.2, [0.50, 0.75, 0.55]),
        FurnitureKind::Toilet => (1.0, 1.2, 1.4, [0.90, 0.90, 0.92]),
        FurnitureKind::Tub => (2.0, 1.0, 3.0, [0.65, 0.85, 0.90]),
        FurnitureKind::TV => (2.0, 1.4, 0.6, [0.20, 0.20, 0.25]),
    }
}

/// Furniture kind picker: fridge / bed / sofa (each restores one need).
fn FurnitureToolbar(session: SessionRef) -> View {
    const KINDS: [(FurnitureKind, &str); 6] = [
        (FurnitureKind::Fridge, "🧊"),
        (FurnitureKind::Bed, "🛏"),
        (FurnitureKind::Sofa, "🛋"),
        (FurnitureKind::Toilet, "🚽"),
        (FurnitureKind::Tub, "🛁"),
        (FurnitureKind::TV, "📺"),
    ];
    let selected = session.borrow().selected_furniture;
    Row(Modifier::new().gap(2.0)).child(
        KINDS
            .iter()
            .map(|(kind, glyph)| {
                let s = session.clone();
                let kind = *kind;
                Clickable(
                    ToolbarButton(Some(kind) == selected, true, *glyph),
                    move || s.borrow_mut().selected_furniture = Some(kind),
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn MenuToolbar(session: SessionRef) -> View {
    let is_open = session.borrow().show_menu;
    Clickable(
        ToolbarButton(is_open, true, "☰"),
        move || session.borrow_mut().show_menu = !is_open,
    )
}

fn WindowChrome(
    session: SessionRef,
    target: HoverWindow,
    width: f32,
    unhovered_alpha: f32,
    content: View,
) -> View {
    let hovered = session.borrow().hovered_window == Some(target);
    let s_enter = session.clone();
    let s_leave = session;
    ReposeBox(
        Modifier::new()
            .width(width)
            .background(Color::WHITE)
            .clip_rounded(2.0)
            .padding(16.0)
            .alpha(if hovered { 1.0 } else { unhovered_alpha })
            .on_pointer_enter(move |_| {
                s_enter.borrow_mut().hovered_window = Some(target);
            })
            .on_pointer_leave(move |_| {
                if s_leave.borrow().hovered_window == Some(target) {
                    s_leave.borrow_mut().hovered_window = None;
                }
            }),
    )
    .child(content)
}

fn CloseButton(session: SessionRef, target: HoverWindow) -> View {
    Clickable(
        Text("×").size(24.0).color(Color(80, 80, 80, 255)),
        move || match target {
            HoverWindow::Menu => session.borrow_mut().show_menu = false,
            HoverWindow::Debug => session.borrow_mut().show_debug = false,
            HoverWindow::Building => {
                session.borrow_mut().inspected_building = None;
                session.borrow_mut().building_pinned = false;
            }
        },
    )
}

/// .window.building 15em x 18em, opacity 0.8 (1.0 pinned), triangle pointer.
fn BuildingWindow(session: SessionRef) -> View {
    let (id, pinned, hovered) = {
        let s = session.borrow();
        (
            s.inspected_building.clone().unwrap_or_default(),
            s.building_pinned,
            s.hovered_window == Some(HoverWindow::Building),
        )
    };
    let s_enter = session.clone();
    let s_leave = session.clone();
    ZStack(Modifier::new().width(240.0).height(296.0)).child((
        ReposeBox(
            Modifier::new()
                .absolute()
                .offset(Some(0.0), Some(0.0), None, None)
                .width(240.0)
                .height(288.0)
                .background(Color::WHITE)
                .clip_rounded(2.0)
                .padding(16.0)
                .alpha(if hovered || pinned { 1.0 } else { 0.8 })
                .on_pointer_enter(move |_| {
                    s_enter.borrow_mut().hovered_window = Some(HoverWindow::Building);
                })
                .on_pointer_leave(move |_| {
                    if s_leave.borrow().hovered_window == Some(HoverWindow::Building) {
                        s_leave.borrow_mut().hovered_window = None;
                    }
                }),
        )
        .child(
            Column(Modifier::new().gap(6.0)).child(vec![
                Row(Modifier::new().gap(6.0)).child({
                    let mut r = vec![Text(format_id(&id))
                        .size(12.8)
                        .color(Color(60, 60, 60, 255))];
                    if pinned {
                        r.push(Spacer());
                        r.push(CloseButton(session.clone(), HoverWindow::Building));
                    }
                    r
                }),
                BuildingBody(session.clone(), &id),
            ]),
        ),
        ReposeBox(
            Modifier::new()
                .absolute()
                .offset(Some(112.0), Some(280.0), None, None)
                .width(16.0)
                .height(16.0)
                .background(Color::WHITE)
                .rotate(0.785)
                .alpha(if hovered || pinned { 1.0 } else { 0.8 }),
        ),
    ))
}

/// Live building body: block dimensions plus the sim agents currently
/// inside its footprint (replaces the old mock households).
fn BuildingBody(session: SessionRef, id: &str) -> View {
    let (dims, agents, zoning) = {
        let s = session.borrow();
        let dims = block_by_id(id).map(|b| (b.w, b.h, b.d));
        let zoning: Option<(ZoneFunction, f32, u32)> = {
            let mut sim = s.sim.borrow_mut();
            let mut q = sim.world.query::<&CityBuilding>();
            q.iter_mut(&mut sim.world)
                .find(|b| b.label == id)
                .map(|b| (b.function, b.development, b.capacity))
        };
        let agents: Vec<(usize, String, f32)> = match block_by_id(id) {
            Some(b) => {
                let mut sim = s.sim.borrow_mut();
                let mut q = sim.world.query::<(&Position, &SimAgentState, &Needs)>();
                q.iter_mut(&mut sim.world)
                    .filter(|(p, _, _)| {
                        (p.x - b.cx).abs() <= b.w / 2.0 && (p.z - b.cz).abs() <= b.d / 2.0
                    })
                    .enumerate()
                    .map(|(i, (_, st, needs))| {
                        let label = match st {
                            SimAgentState::Idle => "idling",
                            SimAgentState::Walk => "strolling",
                            SimAgentState::SeekGoal => "seeking",
                            SimAgentState::Socialize => "chatting",
                            SimAgentState::Working => "working",
                            SimAgentState::Sleeping => "sleeping",
                        }
                        .to_string();
                        (i, label, needs.lowest())
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        (dims, agents, zoning)
    };
    let mut items: Vec<View> = Vec::new();
    match dims {
        Some((w, h, d)) => items.push(
            Text(format!("Block {w:.0}×{d:.0}, {h:.0} high"))
                .size(20.8)
                .color(Color::BLACK),
        ),
        None => items.push(Text("Unknown block").size(20.8).color(Color::BLACK)),
    }
    match zoning {
        Some((function, dev, cap)) => {
            let kind = match function {
                ZoneFunction::Residential => "homes",
                ZoneFunction::Commercial | ZoneFunction::Industrial => "jobs",
                _ => "sites",
            };
            items.push(
                Text(format!("{function:?} · {:.0}% grown · {cap} {kind}", dev * 100.0))
                    .size(12.8)
                    .color(Color(80, 80, 80, 255)),
            );
        }
        None => items.push(
            Text("Unzoned — stamp a zone, then Implement".to_string())
                .size(12.8)
                .color(Color(80, 80, 80, 255)),
        ),
    }
    // City-wide beds (homes live in residential doors, not this block).
    let (taken, beds) = {
        let s = session.borrow();
        let mut sim = s.sim.borrow_mut();
        let mut q = sim.world.query::<&Dwelling>();
        q.iter_mut(&mut sim.world)
            .fold((0u32, 0u32), |(t, c), d| (t + d.taken, c + d.capacity))
    };
    if beds > 0 {
        items.push(
            Text(format!("Beds {taken}/{beds} taken city-wide"))
                .size(12.8)
                .color(Color(80, 80, 80, 255)),
        );
    }
    items.push(
        Text(if agents.is_empty() {
            "No agents inside.".to_string()
        } else {
            format!("{} agent{} inside:", agents.len(), if agents.len() == 1 { "" } else { "s" })
        })
        .size(12.8)
        .color(Color(80, 80, 80, 255)),
    );
    for (idx, label, low) in agents.into_iter().take(6) {
        items.push(
            Column(Modifier::new().gap(2.0)).child(vec![
                Text(format!("Agent {idx}")).size(16.0).color(Color::BLACK),
                Text(format!("{label} • needs {:.0}%", low * 100.0))
                    .size(12.8)
                    .color(Color(80, 80, 80, 255)),
            ]),
        );
    }
    ReposeBox(Modifier::new().fill_max_width().height(150.0))
        .child(Column(Modifier::new().gap(8.0)).child(items))
}

fn format_id(id: &str) -> String {
    if id.len() > 8 {
        id.chars().take(8).collect()
    } else {
        id.to_string()
    }
}

/// .window.debug with <details> sections + .scrollableLog.
fn DebugWindow(session: SessionRef) -> View {
    WindowChrome(
        session.clone(),
        HoverWindow::Debug,
        360.0,
        0.5,
        Column(Modifier::new().gap(10.0)).child(vec![
            Row(Modifier::new().gap(8.0)).child(vec![
                Text("Debugging")
                    .size(25.6)
                    .color(Color::BLACK)
                    .font_weight(FontWeight::BOLD),
                Spacer(),
                CloseButton(session.clone(), HoverWindow::Debug),
            ]),
            DetailsSection(
                session.clone(),
                "Debug Actions",
                session.borrow().debug_sec_actions,
                {
                    let s = session.clone();
                    move || {
                        let v = !s.borrow().debug_sec_actions;
                        s.borrow_mut().debug_sec_actions = v;
                    }
                },
                DebugActions(session.clone()),
            ),
            DetailsSection(
                session.clone(),
                "Networking",
                session.borrow().debug_sec_net,
                {
                    let s = session.clone();
                    move || {
                        let v = !s.borrow().debug_sec_net;
                        s.borrow_mut().debug_sec_net = v;
                    }
                },
                Column(Modifier::new().gap(6.0)).child(vec![
                    Text("browser: 0").size(12.0).color(Color(80, 80, 80, 255)),
                    ScrollableLog(vec!["queues: —".to_string()]),
                    ScrollableLog(vec!["messages: —".to_string()]),
                ]),
            ),
            DetailsSection(
                session.clone(),
                "Simulation Log",
                session.borrow().debug_sec_log,
                {
                    let s = session.clone();
                    move || {
                        let v = !s.borrow().debug_sec_log;
                        s.borrow_mut().debug_sec_log = v;
                    }
                },
                ScrollableLog(vec!["0 [setup] resims: sim ready".to_string()]),
            ),
            DetailsSection(
                session.clone(),
                "Simulation",
                session.borrow().debug_sec_sim,
                {
                    let s = session.clone();
                    move || {
                        let v = !s.borrow().debug_sec_sim;
                        s.borrow_mut().debug_sec_sim = v;
                    }
                },
                SimStats(session.clone()),
            ),
        ]),
    )
}

fn DetailsSection(
    _session: SessionRef,
    title: &str,
    open: bool,
    toggle: impl Fn() + 'static,
    content: View,
) -> View {
    let header = Clickable(
        Row(Modifier::new().gap(6.0)).child(vec![
            Text(if open { "▾" } else { "▸" })
                .size(12.0)
                .color(Color(80, 80, 80, 255)),
            Text(title).size(14.0).color(Color::BLACK),
        ]),
        toggle,
    );
    if open {
        Column(Modifier::new().gap(6.0))
            .child(vec![header, ReposeBox(Modifier::new().padding_values(PaddingValues { left: 18.0, right: 0.0, top: 0.0, bottom: 0.0 })).child(content)])
    } else {
        Column(Modifier::new()).child(header)
    }
}

fn DebugActions(session: SessionRef) -> View {
    let s = session.borrow().clone();
    let mut rows: Vec<View> = Vec::new();
    if s.current_project.is_some() {
        rows.push(
            Row(Modifier::new().gap(8.0)).child(vec![
                Text(format!("Grid size {}", s.debug_grid_n)).size(12.0),
                Text(format!("Lanes {}", s.debug_grid_lanes)).size(12.0),
                Text(format!("Spacing {}", s.debug_grid_spacing)).size(12.0),
            ]),
        );
        rows.push(SmallButton("Plan grid", {
            let _s = session.clone();
            move || {}
        }));
    } else {
        rows.push(Text("(open a project to plan a grid)").size(12.8));
    }
    rows.push(
        Row(Modifier::new().gap(8.0)).child(vec![
            Text(format!("Cars per lane (tries) {}", s.debug_spawn_tries)).size(12.0),
            SmallButton("Spawn cars", {
                let _s = session.clone();
                move || {}
            }),
        ]),
    );
    let label = if s.rendering_enabled {
        "Disable rendering"
    } else {
        "Enable rendering"
    };
    rows.push(SmallButton(label, move || {
        let v = !session.borrow().rendering_enabled;
        session.borrow_mut().rendering_enabled = v;
    }));
    Column(Modifier::new().gap(8.0)).child(rows)
}

fn SmallButton(label: &str, on_click: impl Fn() + 'static) -> View {
    Clickable(
        ReposeBox(
            Modifier::new()
                .background(Color::WHITE)
                .border(1.0, HAIRLINE, 2.0)
                .clip_rounded(2.0)
                .padding_values(PaddingValues {
                    left: 8.0,
                    right: 8.0,
                    top: 4.0,
                    bottom: 4.0,
                }),
        )
        .child(Text(label).size(12.0).color(Color(60, 60, 60, 255))),
        on_click,
    )
}

/// Live sim readout for the debug window + a spawner.
fn SimStats(session: SessionRef) -> View {
    let (agents, secs, points) = {
        let s = session.borrow();
        let mut sim = s.sim.borrow_mut();
        let mut q = sim.world.query::<(&Position, &SimAgentState)>();
        let n = q.iter_mut(&mut sim.world).count();
        (n, s.sim_secs, s.project_points.len())
    };
    Column(Modifier::new().gap(8.0)).child(vec![
        Text(format!("agents: {agents}   sim time: {secs:.1}s")).size(12.0),
        Text(format!("planning points: {points}")).size(12.0),
        SmallButton("Spawn agent", move || {
            let st = session.borrow();
            let mut sim = st.sim.borrow_mut();
            let mut q = sim.world.query::<(&Position, &SimAgentState)>();
            let n = q.iter_mut(&mut sim.world).count();
            let x = ((n * 37) % 80) as f32 - 40.0;
            let z = ((n * 53) % 80) as f32 - 40.0;
            sim.spawn_agent(x, z);
        }),
    ])
}

fn ScrollableLog(lines: Vec<String>) -> View {
    ReposeBox(
        Modifier::new()
            .fill_max_width()
            .height(120.0)
            .background(LOG_BG)
            .padding(16.0),
    )
    .child(
        Column(Modifier::new().gap(2.0)).child(
            lines
                .into_iter()
                .map(|l| {
                    Text(l)
                        .size(12.0)
                        .color(Color::WHITE)
                        .font_family("monospace")
                })
                .collect::<Vec<_>>(),
        ),
    )
}

/// .window.menu 40em wide, full height minus toolbar, tabs on top.
fn MenuWindow(session: SessionRef) -> View {
    let tab = session.borrow().menu_tab;
    let tabs = ["About", "Credits", "Tutorial", "Settings & Controls", "Game"];
    let s = session.clone();
    ZStack(Modifier::new().width(640.0).fill_max_height()).child((
        WindowChromeFullHeight(
            session.clone(),
            Column(Modifier::new().gap(12.0)).child(vec![
                Row(Modifier::new().gap(6.0)).child(vec![
                    Spacer(),
                    CloseButton(s, HoverWindow::Menu),
                ]),
                Row(Modifier::new().gap(6.0)).child(
                    tabs.iter()
                        .enumerate()
                        .map(|(i, name)| {
                            let sc = session.clone();
                            Clickable(
                                TabButton(i == tab, name),
                                move || sc.borrow_mut().menu_tab = i,
                            )
                        })
                        .collect::<Vec<_>>(),
                ),
                MenuScrollContent(session.clone(), tab),
            ]),
        ),
        ReposeBox(
            Modifier::new()
                .absolute()
                .offset(None, None, Some(8.0), Some(-8.0))
                .width(16.0)
                .height(16.0)
                .background(Color::WHITE)
                .rotate(0.785),
        ),
    ))
}

/// Menu variant of WindowChrome that stretches to full height.
fn WindowChromeFullHeight(session: SessionRef, content: View) -> View {
    let hovered = session.borrow().hovered_window == Some(HoverWindow::Menu);
    let s_enter = session.clone();
    let s_leave = session;
    ReposeBox(
        Modifier::new()
            .width(640.0)
            .fill_max_height()
            .background(Color::WHITE)
            .clip_rounded(2.0)
            .padding(16.0)
            .alpha(if hovered { 1.0 } else { 0.9 })
            .on_pointer_enter(move |_| {
                s_enter.borrow_mut().hovered_window = Some(HoverWindow::Menu);
            })
            .on_pointer_leave(move |_| {
                if s_leave.borrow().hovered_window == Some(HoverWindow::Menu) {
                    s_leave.borrow_mut().hovered_window = None;
                }
            }),
    )
    .child(content)
}

/// antd card tabs, size large: active tab white w/ primary text.
fn TabButton(active: bool, label: &str) -> View {
    ReposeBox(
        Modifier::new()
            .background(if active {
                Color::WHITE
            } else {
                Color(250, 250, 250, 255)
            })
            .border(1.0, Color(232, 232, 232, 255), 2.0)
            .clip_rounded(2.0)
            .padding_values(PaddingValues {
                left: 16.0,
                right: 16.0,
                top: 10.0,
                bottom: 10.0,
            }),
    )
    .child(
        Text(label)
            .size(16.0)
            .color(if active {
                PRIMARY
            } else {
                Color(60, 60, 60, 255)
            }),
    )
}

fn MenuTabContent(session: SessionRef, tab: usize) -> View {
    match tab {
        0 => AboutTab(),
        1 => CreditsTab(),
        2 => TutorialTab(),
        3 => SettingsTab(),
        _ => GameTab(session),
    }
}

fn MenuScrollContent(session: SessionRef, tab: usize) -> View {
    let state = remember_scroll_state("menu_tab_scroll");
    ScrollArea(
        Modifier::new().fill_max_width().weight(1.0),
        state,
        MenuTabContent(session, tab),
    )
}

fn AboutTab() -> View {
    Column(Modifier::new().gap(8.0)).child(vec![
        Text("resims").size(30.0).color(Color::BLACK).letter_spacing(2.0),
        Text(format!("v{}", env!("CARGO_PKG_VERSION")))
            .size(22.8)
            .color(Color::BLACK)
            .font_weight(FontWeight::BOLD),
        Text("Tiny sims on Repose - a fresh citybuilder that dogfoods the Repose UI stack.")
            .size(16.0)
            .color(Color::BLACK),
        Text("Expect construction zones: the sim, planning gestures and canvas picking are still being built.")
            .size(16.0)
            .color(Color::BLACK),
        Text("Upcoming:")
            .size(20.3)
            .color(Color::BLACK)
            .font_weight(FontWeight::BOLD),
        MilestoneProgress(25),
        Text("TODO:").size(18.0).color(Color::BLACK).font_weight(FontWeight::BOLD),
        Text("☐ Sim <-> UI wiring (agents on canvas)").size(14.0),
        Text("☐ Planning gestures (roads, zones)").size(14.0),
        Text("DONE:").size(18.0).color(Color::BLACK).font_weight(FontWeight::BOLD),
        Text("☑ 3D viewport (orbit camera, picking)").size(14.0),
    ])
}

fn MilestoneProgress(percent: u32) -> View {
    Column(Modifier::new().gap(4.0)).child(vec![
        ReposeBox(
            Modifier::new()
                .fill_max_width()
                .height(8.0)
                .background(Color(245, 245, 245, 255))
                .clip_rounded(4.0),
        )
        .child(
            ReposeBox(
                Modifier::new()
                    .width(608.0 * (percent as f32 / 100.0))
                    .height(8.0)
                    .background(PRIMARY)
                    .clip_rounded(4.0),
            ),
        ),
        Text(format!("{percent}%")).size(12.0).color(Color(100, 100, 100, 255)),
    ])
}

fn CreditsTab() -> View {
    Column(Modifier::new().gap(8.0)).child(vec![
        Text("resims").size(30.0).color(Color::BLACK).letter_spacing(2.0),
        Text("A new citybuilder that dogfoods the Repose UI stack.").size(16.0),
        Text("Built with:")
            .size(18.0)
            .color(Color::BLACK)
            .font_weight(FontWeight::BOLD),
        Text("• Repose (UI, platform, renderer)").size(16.0),
        Text("• bevy_ecs (headless simulation)").size(16.0),
    ])
}

fn TutorialTab() -> View {
    Column(Modifier::new().gap(8.0)).child(vec![
        Text("Please note that this tutorial is super bare-bones, but it should get you going.")
            .size(16.0),
        Text("(You can open and close this whole window while following the tutorial by clicking the menu icon)")
            .size(16.0),
        Text("1) Click the pencil icon to go into planning mode.")
            .size(16.0)
            .font_weight(FontWeight::BOLD),
        Text("2) Click the \"Start a new project\" button.")
            .size(16.0)
            .font_weight(FontWeight::BOLD),
        Text("Planning Roads").size(22.8).font_weight(FontWeight::BOLD),
        Text("1) Go to road planning mode by clicking the road icon.")
            .size(16.0)
            .font_weight(FontWeight::BOLD),
        Text("2) Start a new road by clicking on the map and continue to click to add road nodes.")
            .size(16.0),
        Text("3) To finish a road, double-click when placing the last node.").size(16.0),
        Text("Planning Zones").size(22.8).font_weight(FontWeight::BOLD),
        Text("1) Go to zone planning mode by clicking the zone icon next to the road icon.")
            .size(16.0)
            .font_weight(FontWeight::BOLD),
        Text("2) Draw zone shapes by selecting a zone type, then clicking on the map to define its corners.")
            .size(16.0),
        Text("Implementing Projects").size(22.8).font_weight(FontWeight::BOLD),
        Text("Press the \"Implement\" button to implement your project plan.").size(16.0),
        Text("Further Steps").size(22.8).font_weight(FontWeight::BOLD),
        Text("Speed up time using the slider next to the clock in the top left corner and see what happens.")
            .size(16.0)
            .font_weight(FontWeight::BOLD),
        Text("Click on the eye icon and hover/click on buildings to inspect them")
            .size(16.0)
            .font_weight(FontWeight::BOLD),
    ])
}

fn SettingsTab() -> View {
    Column(Modifier::new().gap(8.0)).child(vec![
        Text("Settings & Controls").size(20.3).font_weight(FontWeight::BOLD),
        SettingsRow("Pan", "arrow keys / drag"),
        SettingsRow("Rotate", "alt + drag"),
        SettingsRow("Zoom", "wheel / pinch"),
        SettingsRow("Implement Plan", "ctrl+enter"),
        SettingsRow("Undo Plan Step", "ctrl+z"),
        SettingsRow("Redo Plan Step", "ctrl+shift+z"),
        SettingsRow("Toggle Debug", "."),
        SettingsRow("Oversampling/Retina", "2.0"),
    ])
}

/// Save / load tab: crash-safe RON store under the platform data dir.
/// Loading restores sim + city edits; jobs, homes, queues reset.
fn GameTab(session: SessionRef) -> View {
    let msg = session.borrow().last_save_msg.clone();
    let rate = session.borrow().sim.borrow().tax_rate();
    let s_save = session.clone();
    let s_load = session.clone();
    let s_down = session.clone();
    let s_up = session.clone();
    let mut items = vec![
        Text("Saved Game").size(20.3).font_weight(FontWeight::BOLD),
        Row(Modifier::new().gap(8.0)).child(vec![
            Clickable(PrimaryButton("Save game"), move || {
                do_save(&mut s_save.borrow_mut(), &save_manager())
            }),
            Clickable(PrimaryButton("Load game"), move || {
                do_load(&mut s_load.borrow_mut(), &save_manager())
            }),
        ]),
        Text("Jobs, homes, queues and claims reset on load.".to_string())
            .size(14.0)
            .color(Color(80, 80, 80, 255)),
        Text("City Budget".to_string()).size(20.3).font_weight(FontWeight::BOLD),
        Row(Modifier::new().gap(8.0)).child(vec![
            Clickable(PrimaryButton("-"), move || {
                let ui = s_down.borrow_mut();
                let r = ui.sim.borrow().tax_rate();
                ui.sim.borrow_mut().set_tax_rate(r - 0.01);
            }),
            Text(format!("Tax {:.0}%", rate * 100.0)).size(14.0),
            Clickable(PrimaryButton("+"), move || {
                let ui = s_up.borrow_mut();
                let r = ui.sim.borrow().tax_rate();
                ui.sim.borrow_mut().set_tax_rate(r + 0.01);
            }),
        ]),
        Text("Funds services; high taxes slow growth.".to_string())
            .size(14.0)
            .color(Color(80, 80, 80, 255)),
    ];
    if !msg.is_empty() {
        items.push(Text(msg).size(14.0).color(Color::BLACK));
    }
    Column(Modifier::new().gap(8.0)).child(items)
}

fn SettingsRow(label: &str, value: &str) -> View {
    Row(Modifier::new().gap(12.0)).child(vec![
        ReposeBox(Modifier::new().width(160.0)).child(Text(label).size(14.0)),
        ReposeBox(
            Modifier::new()
                .background(Color(240, 240, 240, 255))
                .clip_rounded(2.0)
                .padding_values(PaddingValues {
                    left: 8.0,
                    right: 8.0,
                    top: 4.0,
                    bottom: 4.0,
                }),
        )
        .child(Text(value).size(14.0)),
    ])
}

#[cfg(target_arch = "wasm32")]
pub fn init_wasm() {
    console_error_panic_hook::set_once();
    if web_workers::web::has_spawn_support() {
        let _ = web_sys::console::log_1(&"wasm worker threads: available".into());
    } else {
        let _ = web_sys::console::warn_1(
            &"wasm worker threads: unavailable (need COOP/COEP)".into(),
        );
    }
}

#[cfg(test)]
mod shortcut_tests {
    use super::*;
    use repose_core::input::Modifiers;

    fn key_down(key: Key, command: bool, shift: bool) -> KeyEvent {
        KeyEvent {
            key,
            modifiers: Modifiers {
                command,
                shift,
                ctrl: command,
                alt: false,
                meta: false,
            },
            is_repeat: false,
            event_type: KeyEventType::Down,
            utf16_code_point: 0,
        }
    }

    fn session_with_project() -> SessionRef {
        Rc::new(RefCell::new(UiState {
            current_project: Some("prj001".to_string()),
            planning_mode: Some(PlanningMode::Roads),
            selected_land_use: Some(LandUse::Residential),
            ..UiState::default()
        }))
    }

    #[test]
    fn implement_shortcut_clears_project() {
        let s = session_with_project();
        s.borrow_mut().project_points.push([1.0, 2.0]);
        s.borrow_mut().inspected_building = Some("bld:1".to_string());
        s.borrow_mut().building_pinned = true;
        assert!(handle_shortcut(&s, key_down(Key::Enter, true, false)));
        let st = s.borrow();
        assert!(st.current_project.is_none());
        assert!(st.planning_mode.is_none());
        assert!(st.selected_land_use.is_none());
        assert!(st.project_points.is_empty());
        assert!(st.inspected_building.is_none());
        assert!(!st.building_pinned);
    }

    #[test]
    fn implement_shortcut_stamps_roads() {
        let s = session_with_project();
        s.borrow_mut().project_points = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]];
        assert!(handle_shortcut(&s, key_down(Key::Enter, true, false)));
        let st = s.borrow();
        assert!(st.current_project.is_none());
        assert!(st.project_points.is_empty());
        assert_eq!(st.built_roads.len(), 1);
        assert_eq!(st.built_roads[0].len(), 3);
    }

    #[test]
    fn sim_speed_marks_match_slider_labels() {
        assert_eq!(sim_speed_multiplier(0.0), 0.0); // ||
        assert_eq!(sim_speed_multiplier(1.0), 1.0); // 1x
        assert_eq!(sim_speed_multiplier(3.0), 4.0); // 4x
        assert_eq!(sim_speed_multiplier(6.0), 32.0); // 32x
    }

    #[test]
    fn clock_formats_sim_seconds() {
        assert_eq!(clock_hm(0.0), (0, 0));
        assert_eq!(clock_hm(59.0), (0, 0));
        assert_eq!(clock_hm(60.0), (0, 1));
        assert_eq!(clock_hm(34200.0), (9, 30));
        assert_eq!(clock_hm(86399.0), (23, 59));
        assert_eq!(clock_hm(86400.0), (0, 0));
    }

    #[test]
    fn undo_redo_shortcuts_flip_redo_flag() {
        let s = session_with_project();
        assert!(handle_shortcut(&s, key_down(Key::Character('z'), true, false)));
        assert!(s.borrow().has_redo);
        assert!(handle_shortcut(&s, key_down(Key::Character('z'), true, true)));
        assert!(!s.borrow().has_redo);
    }

    #[test]
    fn debug_toggle_shortcut_flips_window() {
        let s: SessionRef = Rc::new(RefCell::new(UiState::default()));
        let dot = KeyEvent {
            key: Key::Character('.'),
            modifiers: Modifiers::default(),
            is_repeat: false,
            event_type: KeyEventType::Down,
            utf16_code_point: 0,
        };
        assert!(!s.borrow().show_debug);
        assert!(handle_shortcut(&s, dot.clone()));
        assert!(s.borrow().show_debug);
        assert!(handle_shortcut(&s, dot));
        assert!(!s.borrow().show_debug);
    }

    #[test]
    fn shortcuts_ignore_repeats_releases_and_plain_keys() {
        let s = session_with_project();
        let mut ev = key_down(Key::Enter, true, false);
        ev.is_repeat = true;
        assert!(!handle_shortcut(&s, ev));
        let mut ev = key_down(Key::Enter, true, false);
        ev.event_type = KeyEventType::Up;
        assert!(!handle_shortcut(&s, ev));
        assert!(!handle_shortcut(&s, key_down(Key::Enter, false, false)));
        assert!(s.borrow().current_project.is_some());
    }

    fn session_seeded() -> SessionRef {
        let s: SessionRef = Rc::new(RefCell::new(UiState::default()));
        {
            let mut st = s.borrow_mut();
            seed_sim(&mut st);
        }
        s
    }

    fn first_agent(s: &SessionRef) -> Entity {
        let st = s.borrow();
        let mut sim = st.sim.borrow_mut();
        let mut q = sim.world.query::<(Entity, &Position)>();
        q.iter_mut(&mut sim.world).next().unwrap().0
    }

    #[test]
    fn direct_order_without_selection_is_noop() {
        let s = session_seeded();
        let agent = first_agent(&s);
        assert!(!direct_order(&mut s.borrow_mut(), 50.0, 50.0));
        s.borrow().sim.borrow_mut().step(0.1);
        assert_eq!(
            s.borrow().sim.borrow_mut().world.get::<SimAgentState>(agent),
            Some(&SimAgentState::Idle)
        );
    }

    #[test]
    fn direct_order_sends_goto() {
        let s = session_seeded();
        let agent = first_agent(&s);
        s.borrow_mut().selected_agent = Some(agent);
        // Far from every seeded goal: plain move order.
        assert!(direct_order(&mut s.borrow_mut(), 50.0, 50.0));
        s.borrow().sim.borrow_mut().step(0.1);
        let st = s.borrow();
        let sim = st.sim.borrow_mut();
        let target = sim.world.get::<resims_sim::WalkTarget>(agent).copied();
        assert!(target.is_some());
        assert_eq!((target.unwrap().x, target.unwrap().z), (50.0, 50.0));
    }

    #[test]
    fn direct_order_near_goal_sends_use() {
        let s = session_seeded();
        let agent = first_agent(&s);
        s.borrow_mut().selected_agent = Some(agent);
        // Seeded hunger goal sits at (-20, 0).
        assert!(direct_order(&mut s.borrow_mut(), -20.0, 0.0));
        s.borrow().sim.borrow_mut().step(0.1);
        let st = s.borrow();
        let sim = st.sim.borrow_mut();
        assert_eq!(
            sim.world.get::<SimAgentState>(agent),
            Some(&SimAgentState::SeekGoal)
        );
        let target = sim.world.get::<resims_sim::WalkTarget>(agent).copied();
        assert_eq!((target.unwrap().x, target.unwrap().z), (-20.0, 0.0));
    }

    #[test]
    fn employ_click_toggles_job() {
        let s = session_seeded();
        let agent = first_agent(&s);
        // No selection, and far clicks: untouched.
        assert!(!maybe_employ(&mut s.borrow_mut(), -24.0, 8.0));
        s.borrow_mut().selected_agent = Some(agent);
        assert!(!maybe_employ(&mut s.borrow_mut(), 50.0, 50.0));
        // Seeded café workplace sits at (-24, 8).
        assert!(maybe_employ(&mut s.borrow_mut(), -24.0, 8.0));
        assert!(s.borrow().sim.borrow_mut().world.get::<Employment>(agent).is_some());
        // Click again: dismissed.
        assert!(maybe_employ(&mut s.borrow_mut(), -24.0, 8.0));
        assert!(s.borrow().sim.borrow_mut().world.get::<Employment>(agent).is_none());
    }

    fn session_zoning(land_use: LandUse, poly: Vec<[f32; 2]>) -> SessionRef {
        let s: SessionRef = Rc::new(RefCell::new(UiState {
            current_project: Some("prj009".to_string()),
            planning_mode: Some(PlanningMode::Zoning),
            selected_land_use: Some(land_use),
            project_points: poly,
            ..UiState::default()
        }));
        s
    }

    /// Polygon around bld:3's center (12, 10).
    fn bld3_poly() -> Vec<[f32; 2]> {
        vec![[8.0, 6.0], [16.0, 6.0], [16.0, 14.0], [8.0, 14.0]]
    }

    #[test]
    fn point_in_polygon_sides() {
        assert!(point_in_polygon([12.0, 10.0], &bld3_poly()));
        assert!(!point_in_polygon([0.0, 0.0], &bld3_poly()));
        assert!(!point_in_polygon([12.0, 20.0], &bld3_poly()));
    }

    #[test]
    fn zoning_implement_stamps_building_and_spawns_jobs() {
        let s = session_zoning(LandUse::Commercial, bld3_poly());
        implement_project(&mut s.borrow_mut());
        let st = s.borrow();
        assert_eq!(st.stamped.len(), 1);
        assert_eq!(st.stamped[0].id, "bld:3");
        let mut sim = st.sim.borrow_mut();
        let mut bq = sim.world.query::<&CityBuilding>();
        let buildings: Vec<_> = bq.iter_mut(&mut sim.world).collect();
        assert_eq!(buildings.len(), 1);
        assert_eq!(buildings[0].function, ZoneFunction::Commercial);
        assert_eq!(buildings[0].capacity, 2); // 8x8=64u² / 25
        let mut wq = sim.world.query::<&Workplace>();
        let sites: Vec<_> = wq.iter_mut(&mut sim.world).collect();
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].pay_per_hour, 30.0);
    }

    #[test]
    fn restamp_replaces_function_set() {
        let s = session_zoning(LandUse::Commercial, bld3_poly());
        implement_project(&mut s.borrow_mut());
        // Re-stamp the same block industrial.
        {
            let mut st = s.borrow_mut();
            st.current_project = Some("prj010".to_string());
            st.planning_mode = Some(PlanningMode::Zoning);
            st.selected_land_use = Some(LandUse::Industrial);
            st.project_points = bld3_poly();
            implement_project(&mut st);
        }
        let st = s.borrow();
        assert_eq!(st.stamped.len(), 1);
        let mut sim = st.sim.borrow_mut();
        let mut bq = sim.world.query::<&CityBuilding>();
        let buildings: Vec<_> = bq.iter_mut(&mut sim.world).collect();
        assert_eq!(buildings.len(), 1);
        assert_eq!(buildings[0].function, ZoneFunction::Industrial);
        let mut wq = sim.world.query::<&Workplace>();
        let sites: Vec<_> = wq.iter_mut(&mut sim.world).collect();
        // Old commercial site despawned, one industrial site stands.
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].pay_per_hour, 38.0);
    }

    #[test]
    fn roads_implement_syncs_sim_network() {
        use resims_sim::Roads;
        let s = session_with_project(); // roads mode, 2-point project below
        s.borrow_mut().project_points = vec![[0.0, 0.0], [10.0, 0.0]];
        implement_project(&mut s.borrow_mut());
        let st = s.borrow();
        let sim = st.sim.borrow();
        let roads = sim.world.get_resource::<Roads>().unwrap();
        assert_eq!(roads.0.len(), 1);
    }

    #[test]
    fn residential_stamp_spawns_dwelling_at_door() {
        let s = session_zoning(LandUse::Residential, bld3_poly());
        implement_project(&mut s.borrow_mut());
        let st = s.borrow();
        assert_eq!(st.stamped.len(), 1);
        let mut sim = st.sim.borrow_mut();
        let mut dq = sim.world.query::<&Dwelling>();
        let homes: Vec<_> = dq.iter_mut(&mut sim.world).collect();
        assert_eq!(homes.len(), 1);
        assert_eq!(homes[0].capacity, 2);
        // Door: west of bld:3 (min_x 8) at mid-depth 10.
        assert_eq!((homes[0].x, homes[0].z), (6.0, 10.0));
    }

    #[test]
    fn commercial_stamp_sits_workplace_at_door() {
        let s = session_zoning(LandUse::Commercial, bld3_poly());
        implement_project(&mut s.borrow_mut());
        let st = s.borrow();
        let mut sim = st.sim.borrow_mut();
        let mut wq = sim.world.query::<&Workplace>();
        let sites: Vec<_> = wq.iter_mut(&mut sim.world).collect();
        assert_eq!(sites.len(), 1);
        assert_eq!((sites[0].x, sites[0].z), (6.0, 10.0));
    }

    #[test]
    fn move_in_click_toggles_home() {
        let s = session_seeded();
        let agent = first_agent(&s);
        let home = s.borrow().sim.borrow_mut().spawn_dwelling(-24.0, 8.0, 2);
        assert!(!maybe_move_in(&mut s.borrow_mut(), -24.0, 8.0)); // no selection
        s.borrow_mut().selected_agent = Some(agent);
        assert!(!maybe_move_in(&mut s.borrow_mut(), 50.0, 50.0)); // open ground
        assert!(maybe_move_in(&mut s.borrow_mut(), -24.0, 8.0));
        assert_eq!(
            s.borrow().sim.borrow_mut().world.get::<Home>(agent),
            Some(&Home(home))
        );
        // Click again: moved out.
        assert!(maybe_move_in(&mut s.borrow_mut(), -24.0, 8.0));
        assert!(s.borrow().sim.borrow_mut().world.get::<Home>(agent).is_none());
    }

    #[test]
    fn city_stats_counts_agents_and_mood() {
        let s = session_seeded(); // 6 agents at default 0.8 needs
        let (pop, happy) = city_stats(&s.borrow());
        assert_eq!(pop, 6);
        assert!((happy - 0.8).abs() < 1e-6, "happy: {happy}");
    }

    #[test]
    fn panel_lines_format_demand_and_ledger() {
        use resims_sim::Demand;
        let d = Demand { residential: 1.0, commercial: -0.6, industrial: 0.05 };
        assert_eq!(demand_line(d), "R+1.0 C-0.6 I+0.1");
        let l = Ledger { property_in: 20.0, wage_in: 0.1, services_out: 8.0 };
        assert_eq!(ledger_line(l), "$20.10 in · $8.00 out");
    }

    #[test]
    fn seeded_agents_get_valid_stable_outfits() {
        let s = session_seeded();
        let st = s.borrow();
        let mut sim = st.sim.borrow_mut();
        let mut q = sim.world.query::<&Personality>();
        let mut count = 0;
        for p in q.iter(&sim.world) {
            let (o, skin) = outfit_for(p);
            for c in o.into_iter().chain(skin) {
                assert!((0.0..=1.0).contains(&c), "channel {c}");
            }
            // Stable identity: same traits, same look.
            assert_eq!(outfit_for(p), (o, skin));
            count += 1;
        }
        assert!(count > 0, "seeded session must have agents");
    }

    #[test]
    fn walls_implement_stamps_chain_and_syncs() {
        use resims_sim::Walls;
        let s: SessionRef = Rc::new(RefCell::new(UiState {
            current_project: Some("prj011".to_string()),
            planning_mode: Some(PlanningMode::Walls),
            project_points: vec![[0.0, 0.0], [0.0, 8.0], [8.0, 8.0]],
            ..UiState::default()
        }));
        implement_project(&mut s.borrow_mut());
        let st = s.borrow();
        // 3-point chain -> 2 segments, draft cleared, project closed.
        assert_eq!(st.built_walls.len(), 2);
        assert_eq!(st.built_walls[0], [[0.0, 0.0], [0.0, 8.0]]);
        assert!(st.project_points.is_empty());
        assert!(st.current_project.is_none());
        let sim = st.sim.borrow();
        let walls = sim.world.get_resource::<Walls>().unwrap();
        assert_eq!(walls.0.len(), 2);
        assert_eq!(walls.0[1].bx, 8.0);
    }

    fn mem_manager() -> SaveManager<game_utils::MemoryStorage> {
        SaveManager::new_with_storage(
            "com",
            "resims-test",
            "ui-save",
            "save.ron",
            SAVE_VERSION,
            game_utils::MemoryStorage::new(),
        )
    }

    #[test]
    fn save_load_restores_stamps() {
        let mgr = mem_manager();
        let s = session_zoning(LandUse::Commercial, bld3_poly());
        implement_project(&mut s.borrow_mut());
        do_save(&mut s.borrow_mut(), &mgr);
        assert_eq!(s.borrow().last_save_msg, "Saved.");
        // Fresh session loads the stamp back with resolved entities.
        let s2: SessionRef = Rc::new(RefCell::new(UiState::default()));
        do_load(&mut s2.borrow_mut(), &mgr);
        let st = s2.borrow();
        assert_eq!(st.stamped.len(), 1);
        assert_eq!(st.stamped[0].id, "bld:3");
        assert!(!st.stamped[0].generated.is_empty());
        let mut sim = st.sim.borrow_mut();
        let mut bq = sim.world.query::<&CityBuilding>();
        assert_eq!(bq.iter_mut(&mut sim.world).count(), 1);
        assert!(st.last_save_msg.starts_with("Loaded day"));
    }

    #[test]
    fn load_missing_reports_no_save() {
        let mgr = mem_manager();
        let s: SessionRef = Rc::new(RefCell::new(UiState::default()));
        do_load(&mut s.borrow_mut(), &mgr);
        assert_eq!(s.borrow().last_save_msg, "No save yet.");
        assert!(s.borrow().stamped.is_empty());
    }

    #[test]
    fn furniture_implement_places_pieces() {
        let s: SessionRef = Rc::new(RefCell::new(UiState {
            current_project: Some("prj012".to_string()),
            planning_mode: Some(PlanningMode::Furniture),
            selected_furniture: Some(FurnitureKind::Bed),
            project_points: vec![[2.0, 2.0], [6.0, 6.0]],
            ..UiState::default()
        }));
        implement_project(&mut s.borrow_mut());
        let st = s.borrow();
        assert!(st.project_points.is_empty());
        assert!(st.current_project.is_none());
        let mut sim = st.sim.borrow_mut();
        let mut q = sim.world.query::<(&Goal, &Furniture)>();
        let pieces: Vec<_> = q.iter_mut(&mut sim.world).collect();
        assert_eq!(pieces.len(), 2);
        assert!(pieces.iter().all(|(_, f)| **f == Furniture(FurnitureKind::Bed)));
        assert_eq!(furniture_prop(FurnitureKind::Fridge).1, 2.2);
    }
}
