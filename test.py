import hydra

from third_person_man.testing import *

def trajectory_replay(cfg):
    traj_replay = TrajectoryReplay(
        data_path = cfg.data_path, 
        demo_num = cfg.demo_num, 
        representations = cfg.representations, 
        env_cfg = cfg.env_cfg
    )
    traj_replay.save_trajectory()

def fingertip_replay(cfg):
    fingertip_replay = FingertipReplay(
        data_path = cfg.data_path, 
        demo_num = cfg.demo_num, 
        representations = cfg.representations, 
        env_cfg = cfg.env_cfg
    )

    fingertip_replay.save_trajectory()


@hydra.main(version_base=None, config_path='configs', config_name='testing')
def main(cfg) -> None: 
    cfg = cfg.fingertip_replay
    fingertip_replay(cfg)

if __name__ == '__main__': 
    main()