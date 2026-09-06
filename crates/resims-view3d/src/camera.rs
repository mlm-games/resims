//! Orbit camera: yaw/pitch/distance around a ground target, Y-up right-handed.
//!
//! Pure math (glam only), so it is unit-tested here and shared by the
//! renderer (CPU vertex transform) and picking (screen ray vs footprints).

use glam::camera::rh::proj::opengl::perspective as perspective_gl;
use glam::camera::rh::view::look_at_mat4;
use glam::{Mat4, Vec2, Vec3};

/// Maps OpenGL-style NDC depth [-1, 1] to wgpu [0, 1] (learn-wgpu pattern).
pub const OPENGL_TO_WGPU: Mat4 = Mat4::from_cols(
    glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.5, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.5, 1.0),
);

/// Orbit camera around a ground target. Angles in radians.
#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub fov_y_deg: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 0.0, -2.0),
            yaw: -0.7,
            pitch: 0.85,
            dist: 110.0,
            fov_y_deg: 45.0,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cp * cy, sp, cp * sy) * self.dist
    }

    pub fn view_matrix(&self) -> Mat4 {
        look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        OPENGL_TO_WGPU
            * perspective_gl(
                self.fov_y_deg.to_radians(),
                aspect.max(0.01),
                0.5,
                2000.0,
            )
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Drag deltas in px: horizontal orbits, vertical tilts.
    pub fn orbit(&mut self, dx_px: f32, dy_px: f32) {
        self.yaw -= dx_px * 0.005;
        self.pitch = (self.pitch + dy_px * 0.005).clamp(0.12, 1.45);
    }

    /// Grab-style pan: content follows the cursor. Deltas in px.
    pub fn pan(&mut self, dx_px: f32, dy_px: f32) {
        let fwd = (self.target - self.eye()).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd).normalize();
        let s = self.dist / 800.0;
        self.target += -right * dx_px * s + up * dy_px * s;
        self.target.y = 0.0;
    }

    /// Multiplicative zoom, e.g. `1.0 + -wheel_y * 0.002`.
    pub fn zoom(&mut self, factor: f32) {
        self.dist = (self.dist * factor).clamp(15.0, 600.0);
    }

    /// World-space ray for a viewport pixel. `viewport_px` is (w, h) in px.
    pub fn screen_ray(&self, aspect: f32, viewport_px: Vec2, px: Vec2) -> (Vec3, Vec3) {
        let ndc = Vec2::new(
            (px.x / viewport_px.x) * 2.0 - 1.0,
            1.0 - (px.y / viewport_px.y) * 2.0,
        );
        let inv = self.view_proj(aspect).inverse();
        let p0 = inv.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
        let p1 = inv.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
        (p0, (p1 - p0).normalize())
    }

    /// Ground-plane (y = 0) point under a viewport pixel, if the ray hits.
    pub fn ground_point(&self, aspect: f32, viewport_px: Vec2, px: Vec2) -> Option<Vec2> {
        let (origin, dir) = self.screen_ray(aspect, viewport_px, px);
        if dir.y > -1e-6 {
            return None;
        }
        let t = -origin.y / dir.y;
        if t < 0.0 {
            return None;
        }
        Some(Vec2::new(origin.x + dir.x * t, origin.z + dir.z * t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aspect() -> f32 {
        16.0 / 9.0
    }

    #[test]
    fn default_camera_sees_origin() {
        let cam = OrbitCamera::default();
        let vp = cam.view_proj(aspect());
        let clip = vp * glam::Vec4::new(0.0, 2.0, 0.0, 1.0);
        let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        assert!(clip.w > 0.0, "origin must be in front of camera");
        assert!(ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0, "origin on screen: {ndc:?}");
        assert!((0.0..=1.0).contains(&ndc.z), "depth remapped: {}", ndc.z);
    }

    #[test]
    fn orbit_zoom_pan_keep_camera_valid() {
        let mut cam = OrbitCamera::default();
        cam.orbit(400.0, -300.0);
        cam.zoom(0.05);
        cam.zoom(50.0);
        cam.pan(1000.0, -1000.0);
        assert!((15.0..=600.0).contains(&cam.dist));
        assert!((0.12..=1.45).contains(&cam.pitch));
        assert_eq!(cam.target.y, 0.0);
        assert!(cam.eye().y > 0.0);
    }

    #[test]
    fn ground_point_at_viewport_center_is_near_target() {
        let cam = OrbitCamera::default();
        let vp = Vec2::new(1600.0, 900.0);
        let g = cam
            .ground_point(aspect(), vp, Vec2::new(800.0, 450.0))
            .expect("center ray hits ground");
        let t = Vec2::new(cam.target.x, cam.target.z);
        assert!((g - t).length() < cam.dist * 0.75, "center ~= target: {g:?}");
    }
}
