//! Raw-wgpu 3D pass behind the Repose UI.
//!
//! [`CityRenderer`] implements [`WgpuCallback`][repose_render_wgpu::WgpuCallback]:
//! the CPU transforms [`crate::scene`] triangles to NDC each frame
//! (`prepare`), uploads them, and draws them inside Repose's UI render
//! pass (`paint`). Deliberately simple on purpose:
//!
//! * one pipeline, position+color vertices, **no uniforms** (camera is
//!   applied on the CPU, so there is nothing to bind);
//! * **no depth buffer** — triangles are painter-sorted back-to-front.
//!   Correct for the starter scene; revisit (GPU depth + instancing)
//!   once tri counts grow;
//! * **no face culling**, so winding mistakes can only cost overdraw,
//!   never holes.

use repose_render_wgpu::{CallbackResources, ScreenDescriptor, WgpuCallback};
use repose_core::PaintCallbackInfo;

use crate::camera::OrbitCamera;
use crate::scene::{AgentMarker, GroundMarker, PathLine, PathPoly, PropBox, WallSeg, build_sorted_tris};

const SHADER: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec3f,
};

@vertex
fn vs_main(@location(0) pos: vec3f, @location(1) color: vec3f) -> VsOut {
    var out: VsOut;
    out.pos = vec4f(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
"#;

/// Frozen frame inputs. Snapshotted at composition time because
/// [`WgpuCallback`] is `Send + Sync` (see the `Embedded` docs: snapshot
/// `Copy` values, never signals).
#[derive(Clone, Default)]
pub struct CitySnapshot {
    pub cam: OrbitCamera,
    pub aspect: f32,
    pub agents: Vec<AgentMarker>,
    pub markers: Vec<GroundMarker>,
    pub paths: Vec<PathLine>,
    pub polys: Vec<PathPoly>,
    pub walls: Vec<WallSeg>,
    pub props: Vec<PropBox>,
}

pub struct CityRenderer(pub CitySnapshot);

#[derive(Default)]
struct CityPipes {
    pipeline: Option<wgpu::RenderPipeline>,
    key: Option<(wgpu::TextureFormat, u32)>,
    vbuf: Option<wgpu::Buffer>,
    cap_floats: usize,
    count: u32,
}

fn ensure_pipeline(
    device: &wgpu::Device,
    pipes: &mut CityPipes,
    format: wgpu::TextureFormat,
    samples: u32,
) {
    if pipes.pipeline.is_some() && pipes.key == Some((format, samples)) {
        return;
    }
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resims-city"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("resims-city-layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    pipes.pipeline = Some(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("resims-city"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                // Must match the UI pass attachment (Depth24PlusStencil8,
                // depth ops disabled): painter sorting handles occlusion,
                // this just satisfies pipeline/pass compatibility.
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        }),
    );
    pipes.key = Some((format, samples));
}

impl WgpuCallback for CityRenderer {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        screen: &ScreenDescriptor,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let pipes = resources.get_or_insert_with::<CityPipes>();
        ensure_pipeline(device, pipes, screen.target_format, screen.sample_count);

        let (floats, _) = build_sorted_tris(
            &self.0.cam,
            self.0.aspect,
            &self.0.agents,
            &self.0.markers,
            &self.0.paths,
            &self.0.polys,
            &self.0.walls,
            &self.0.props,
        );
        let verts = (floats.len() / 6) as u32;
        if verts == 0 {
            pipes.count = 0;
            return Vec::new();
        }
        if pipes.vbuf.is_none() || pipes.cap_floats < floats.len() {
            let cap = floats.len().next_power_of_two().max(4096);
            pipes.vbuf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("resims-city-vbuf"),
                size: (cap * 4) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            pipes.cap_floats = cap;
        }
        queue.write_buffer(
            pipes.vbuf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&floats),
        );
        pipes.count = verts;
        Vec::new()
    }

    fn paint(
        &self,
        _info: PaintCallbackInfo,
        rpass: &mut wgpu::RenderPass,
        resources: &CallbackResources,
    ) {
        let Some(pipes) = resources.get::<CityPipes>() else {
            return;
        };
        if pipes.count == 0 {
            return;
        }
        let (Some(pipeline), Some(vbuf)) = (pipes.pipeline.as_ref(), pipes.vbuf.as_ref()) else {
            return;
        };
        rpass.set_pipeline(pipeline);
        rpass.set_vertex_buffer(0, vbuf.slice(..));
        rpass.draw(0..pipes.count, 0..1);
    }
}
