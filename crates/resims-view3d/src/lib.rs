//! Full-screen 3D viewport: [`Viewport3d`] renders [`crate::scene`] through
//! [`crate::render::CityRenderer`] and owns the orbit camera + gestures.
//!
//! The camera never leaves this crate: the UI passes `inspection` mode in
//! and receives [`PickEvent`]s out. Painter-sorted CPU picking keeps this
//! working with zero GPU picking infrastructure.
#![allow(non_snake_case)]

use std::rc::Rc;
use repose_core::input::{PointerButton, PointerEventKind};
use repose_core::locals::px_to_dp;
use repose_core::{Modifier, Rect, Vec2, View, remember_mutable_with_key, remember_state_with_key};
use repose_render_wgpu::Callback;
use repose_ui::{ViewExt, ZStack};

pub mod camera;
pub mod render;
pub mod scene;

pub use camera::OrbitCamera;
pub use scene::{AgentMarker, GroundMarker, PathLine, PathPoly, PropBox, WallSeg, BLOCKS, block_by_id};

/// UI-facing canvas events. `Hover` fires on pointer-move in inspection
/// mode (cheap footprint test); `Select`/`GroundClick` fire on clean
/// left-clicks (press+release without dragging). `screen` is the cursor
/// viewport position in px so the UI can anchor windows to picks.
#[derive(Clone, Debug)]
pub enum PickEvent {
    Hover { id: Option<String>, screen: [f32; 2] },
    Select { id: String, screen: [f32; 2] },
    GroundClick { x: f32, z: f32 },
}

/// Everything the UI feeds the viewport per frame. Plain data, cheap to
/// rebuild during composition.
#[derive(Clone, Default)]
pub struct ViewportInput {
    pub inspection: bool,
    pub agents: Vec<AgentMarker>,
    pub markers: Vec<GroundMarker>,
    pub paths: Vec<PathLine>,
    pub polys: Vec<PathPoly>,
    pub walls: Vec<WallSeg>,
    pub props: Vec<PropBox>,
}

#[derive(Clone, Copy)]
struct DragState {
    button: PointerButton,
    last_x: f32,
    last_y: f32,
    /// Accumulated px travel; release below CLICK_TRAVEL_PX counts as click.
    travel: f32,
    pan: bool,
}

const CLICK_TRAVEL_PX: f32 = 5.0;

/// Full-screen 3D canvas. Gesture map: left-drag orbits, shift/middle/
/// right-drag pans, wheel zooms, clean clicks pick (inspection) or
/// report the ground point (planning).
pub fn Viewport3d(input: ViewportInput, on_event: impl Fn(PickEvent) + 'static) -> View {
    let inspection = input.inspection;
    let cam = remember_mutable_with_key("view3d_camera", OrbitCamera::default);
    let drag = remember_mutable_with_key("view3d_drag", || None::<DragState>);
    let rect = remember_state_with_key("view3d_rect", Rect::default);
    let on_event = Rc::new(on_event);

    // Snapshot for the Send+Sync render callback (never signals).
    let aspect = {
        let r = rect.borrow();
        if r.h > 1.0 { r.w / r.h } else { 16.0 / 9.0 }
    };
    let snap = crate::render::CitySnapshot {
        cam: *cam.get(),
        aspect,
        agents: input.agents,
        markers: input.markers,
        paths: input.paths,
        polys: input.polys,
        walls: input.walls,
        props: input.props,
    };

    // --- gesture handlers (all share remembered cells via clones) ---
    let r_down = rect.clone();
    let d_down = drag.clone();
    let r_move = rect.clone();
    let c_move = cam.clone();
    let d_move = drag.clone();
    let e_move = on_event.clone();
    let r_up = rect.clone();
    let c_up = cam.clone();
    let d_up = drag.clone();
    let e_up = on_event.clone();
    let d_cancel = drag.clone();
    let e_leave = on_event.clone();
    let c_zoom = cam;

    // Pointer positions arrive in physical px; layout rects are dp.
    // The cursor must be converted to dp first — mixing the two offsets
    // every pick / orbit delta by the display scale factor.
    let local = |rect: &Rect, pe: &repose_core::input::PointerEvent| -> [f32; 2] {
        let p = pe.position_in_window();
        [px_to_dp(p.x) - rect.x, px_to_dp(p.y) - rect.y]
    };
    let local_rc = Rc::new(local);

    let l_down = local_rc.clone();
    let l_move = local_rc.clone();
    let l_up = local_rc;

    ZStack(Modifier::new().fill_max_size()).child(Callback::embedded_view(
        Modifier::new()
            .fill_max_size()
            .on_globally_positioned(move |r| {
                *rect.borrow_mut() = r;
            })
            .on_pointer_down(move |pe| {
                if pe.is_consumed() {
                    return;
                }
                let button = match pe.event {
                    PointerEventKind::Down(b) => b,
                    _ => PointerButton::Primary,
                };
                let r = r_down.borrow();
                let [x, y] = l_down(&r, &pe);
                let pan = !matches!(button, PointerButton::Primary) || pe.modifiers.shift;
                d_down.set(Some(DragState {
                    button,
                    last_x: x,
                    last_y: y,
                    travel: 0.0,
                    pan,
                }));
            })
            .on_pointer_move(move |pe| {
                let r = r_move.borrow();
                let vp = if r.h > 1.0 {
                    glam::Vec2::new(r.w, r.h)
                } else {
                    glam::Vec2::new(1600.0, 900.0)
                };
                let aspect = vp.x / vp.y;
                let [x, y] = l_move(&r, &pe);
                drop(r);
                // Copy the drag state out first: matching on `*cell.get()`
                // would hold the borrow across the arms, and the arms
                // write the cell back -> RefCell panic on every drag.
                let cur: Option<DragState> = *d_move.get();
                match cur {
                    Some(mut d) => {
                        let dx = x - d.last_x;
                        let dy = y - d.last_y;
                        d.travel += dx.abs() + dy.abs();
                        d.last_x = x;
                        d.last_y = y;
                        d_move.set(Some(d));
                        if dx.abs() + dy.abs() > 0.0 {
                            c_move.update(|c| {
                                if d.pan {
                                    c.pan(dx, dy);
                                } else {
                                    c.orbit(dx, dy);
                                }
                            });
                        }
                    }
                    None => {
                        if inspection {
                            let id = scene::pick(
                                &c_move.get(),
                                aspect,
                                vp,
                                glam::Vec2::new(x, y),
                            )
                            .map(str::to_string);
                            e_move(PickEvent::Hover { id, screen: [x, y] });
                        }
                    }
                }
            })
            .on_pointer_up(move |pe| {
                let r = r_up.borrow();
                let vp = if r.h > 1.0 {
                    glam::Vec2::new(r.w, r.h)
                } else {
                    glam::Vec2::new(1600.0, 900.0)
                };
                let aspect = vp.x / vp.y;
                let [x, y] = l_up(&r, &pe);
                drop(r);
                // Same borrow discipline as the move handler: copy out,
                // then write back.
                let cur: Option<DragState> = *d_up.get();
                if let Some(d) = cur {
                    d_up.set(None);
                    let primary = matches!(d.button, PointerButton::Primary);
                    if primary && d.travel < CLICK_TRAVEL_PX && !pe.is_consumed() {
                        let cam_now = *c_up.get();
                        let p = glam::Vec2::new(x, y);
                        if inspection {
                            if let Some(id) = scene::pick(&cam_now, aspect, vp, p) {
                                e_up(PickEvent::Select {
                                    id: id.to_string(),
                                    screen: [x, y],
                                });
                            } else if let Some(g) =
                                cam_now.ground_point(aspect, vp, p)
                            {
                                e_up(PickEvent::GroundClick { x: g.x, z: g.y });
                            }
                        } else if let Some(g) = cam_now.ground_point(aspect, vp, p) {
                            e_up(PickEvent::GroundClick { x: g.x, z: g.y });
                        }
                    }
                }
            })
            .on_pointer_cancel(move |_| {
                d_cancel.set(None);
            })
            .on_pointer_leave(move |_| {
                if inspection {
                    e_leave(PickEvent::Hover { id: None, screen: [0.0, 0.0] });
                }
            })
            .on_scroll(move |d: Vec2| {
                if d.y.abs() > 0.5 {
                    let factor = (1.0 + (-d.y) * 0.002).clamp(0.5, 2.0);
                    c_zoom.update(|c| c.zoom(factor));
                    Vec2::ZERO
                } else {
                    d
                }
            }),
        crate::render::CityRenderer(snap),
    ))
}
