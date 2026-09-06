//! Disk persistence on top of [`game_utils`]: crash-safe RON store,
//! version stamping/migration, platform data dirs (native) and OPFS
//! (wasm). The [`SaveFile`] holds the sim snapshot plus the
//! authoritative city edits (roads, walls, stamp functions); entity
//! links (jobs, homes, claims, queues) don't survive, per [`Sim::restore`].

use game_utils::Storage;
use game_utils::save::{SaveManager, Versioned};
use game_utils::save_store::LoadStatus;
use serde::{Deserialize, Serialize};

use super::{Sim, SimSnapshot, Wall, ZoneFunction};

/// Save format version. Bump on [`SaveFile`] shape changes; old files
/// migrate through [`Versioned::migrate`]. (Inner [`SimSnapshot`] has
/// its own [`SNAPSHOT_VERSION`] counter.)
pub const SAVE_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SaveFile {
    pub version: u32,
    pub snapshot: SimSnapshot,
    pub roads: Vec<Vec<[f32; 2]>>,
    pub walls: Vec<Wall>,
    pub stamped: Vec<(String, ZoneFunction)>,
}

impl Versioned for SaveFile {
    fn version(&self) -> u32 {
        self.version
    }
    fn set_version(&mut self, version: u32) {
        self.version = version;
    }
}

impl Sim {
    /// Write the full game: sim snapshot + city edits.
    pub fn save_game<S: Storage>(
        &mut self,
        mgr: &SaveManager<S>,
        roads: Vec<Vec<[f32; 2]>>,
        walls: Vec<Wall>,
        stamped: Vec<(String, ZoneFunction)>,
    ) -> Result<(), String> {
        let mut file = SaveFile {
            version: SAVE_VERSION,
            snapshot: self.snapshot(),
            roads,
            walls,
            stamped,
        };
        mgr.save_versioned(&mut file)
    }

    /// Load the full game: restores the sim snapshot and hands the
    /// city edits back for the UI to apply (roads, walls, stamps).
    /// The sim is untouched unless the load succeeds.
    pub fn load_game<S: Storage>(&mut self, mgr: &SaveManager<S>) -> (SaveFile, LoadStatus) {
        let (file, status) = mgr.load_with_status::<SaveFile>();
        if matches!(status, LoadStatus::Ok) {
            self.restore(&file.snapshot);
        }
        (file, status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NeedKind;
    use game_utils::MemoryStorage;

    fn mem_manager() -> SaveManager<MemoryStorage> {
        SaveManager::new_with_storage(
            "com",
            "resims-test",
            "save-roundtrip",
            "save.ron",
            SAVE_VERSION,
            MemoryStorage::new(),
        )
    }

    fn sample_sim() -> Sim {
        let mut sim = Sim::new();
        sim.set_hour(10);
        let agent = sim.spawn_agent(0.0, 0.0);
        sim.spawn_goal(10.0, 0.0, NeedKind::Hunger);
        let office = sim.spawn_workplace(20.0, 0.0, 40.0, 9, 17);
        sim.employ(agent, office);
        sim.spawn_building(
            "bld:9".to_string(),
            ZoneFunction::Commercial,
            8.0,
            6.0,
            16.0,
            14.0,
        );
        sim.set_roads(vec![vec![[0.0, 0.0], [10.0, 0.0]]]);
        sim.set_walls(vec![Wall { ax: 0.0, az: 5.0, bx: 10.0, bz: 5.0 }]);
        sim.step(0.1);
        sim
    }

    #[test]
    fn save_load_roundtrip_through_manager() {
        let mgr = mem_manager();
        let mut sim = sample_sim();
        let roads = vec![vec![[0.0, 0.0], [10.0, 0.0]]];
        let walls = vec![Wall { ax: 0.0, az: 5.0, bx: 10.0, bz: 5.0 }];
        let stamped = vec![("bld:9".to_string(), ZoneFunction::Commercial)];
        sim.save_game(&mgr, roads.clone(), walls.clone(), stamped.clone())
            .expect("saves");
        let mut sim2 = Sim::new();
        let (file, status) = sim2.load_game(&mgr);
        assert_eq!(status, LoadStatus::Ok);
        assert_eq!(file.roads, roads);
        assert_eq!(file.walls, walls);
        assert_eq!(file.stamped, stamped);
        assert_eq!(file.snapshot, sim.snapshot());
    }

    #[test]
    fn missing_save_leaves_sim_untouched() {
        let mgr = mem_manager();
        let mut sim = sample_sim();
        let before = sim.snapshot();
        let (_, status) = sim.load_game(&mgr);
        assert_eq!(status, LoadStatus::Missing);
        assert_eq!(sim.snapshot(), before);
    }
}
