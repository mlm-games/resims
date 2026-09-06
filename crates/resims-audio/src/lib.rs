//! Tiny UI blips for resims: procedurally generated sine bursts, no assets.
//!
//! Real playback on desktop via rodio; silent no-op stubs on wasm/android
//! (rodio output there is a separate adventure). [`Audio::try_init`] fails
//! soft — callers keep `Option<Audio>` and skip sound when unavailable.

#[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
mod desktop {
    use kira::{
        AudioManager, AudioManagerSettings, DefaultBackend, Frame,
        sound::static_sound::{StaticSoundData, StaticSoundSettings},
    };
    use std::cell::RefCell;
    use std::sync::Arc;

    pub struct Audio {
        manager: RefCell<AudioManager>,
    }

    impl Audio {
        pub fn try_init() -> anyhow::Result<Self> {
            let manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())
                .map_err(|e| anyhow::anyhow!("audio backend unavailable: {e:?}"))?;
            Ok(Self {
                manager: RefCell::new(manager),
            })
        }

        /// Short sine blip at `freq_hz` (0.09s, exponential decay).
        /// Fire-and-forget; overlapping blips mix on the main track.
        /// `play` needs `&mut`, hence the interior mutability.
        pub fn blip(&self, freq_hz: f32) {
            let frames: Arc<[Frame]> = super::blip_frames(freq_hz).into();
            let data = StaticSoundData {
                sample_rate: 44100,
                frames,
                settings: StaticSoundSettings::default(),
                slice: None,
            };
            let _ = self.manager.borrow_mut().play(data);
        }
    }
}

#[cfg(any(target_arch = "wasm32", target_os = "android"))]
mod stub {
    pub struct Audio(());

    impl Audio {
        pub fn try_init() -> anyhow::Result<Self> {
            anyhow::bail!("audio output not wired on this platform yet")
        }

        pub fn blip(&self, _freq_hz: f32) {}
    }
}

#[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
pub use desktop::Audio;
#[cfg(any(target_arch = "wasm32", target_os = "android"))]
pub use stub::Audio;

/// 0.09s mono sine burst at 44.1kHz with exponential decay. Pure + tested.
/// Desktop-only: `kira::Frame` needs the kira dep, which is desktop-only.
#[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
pub fn blip_frames(freq_hz: f32) -> Vec<kira::Frame> {
    const RATE: f32 = 44100.0;
    const DUR: f32 = 0.09;
    let n = (RATE * DUR) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / RATE;
            let v = (2.0 * std::f32::consts::PI * freq_hz * t).sin() * (-t * 40.0).exp() * 0.5;
            kira::Frame::from_mono(v)
        })
        .collect()
}

#[cfg(all(test, not(any(target_arch = "wasm32", target_os = "android"))))]
mod tests {
    use super::*;

    #[test]
    fn blip_has_expected_shape() {
        let s = blip_frames(660.0);
        assert_eq!(s.len(), (44100.0 * 0.09) as usize);
        assert!(s.iter().all(|f| f.left.is_finite() && f.left.abs() <= 0.5));
        assert!(s.iter().all(|f| (f.left - f.right).abs() < 1e-6));
        // Decays: first quarter louder than last quarter.
        let q = s.len() / 4;
        let head: f32 = s[..q].iter().map(|f| f.left.abs()).sum();
        let tail: f32 = s[3 * q..].iter().map(|f| f.left.abs()).sum();
        assert!(head > tail * 4.0, "head {head} tail {tail}");
        // Starts near zero (no click).
        assert!(s[0].left.abs() < 0.05);
    }
}
