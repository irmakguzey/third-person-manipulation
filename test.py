import hydra

from third_person_man.testing import *

def trajectory_replay(cfg):
    traj_replay = TrajectoryReplay(
        data_path = cfg.data_path, 
        demo_num = cfg.demo_num, 
        representations = cfg.representations, 
        env_cfg = cfg.env_cfg
    )
    traj_replay.save_trajectory(title='cube_flipping_replay_diff_view.mp4')

def fingertip_replay(cfg):
    fingertip_replay = FingertipReplay(
        data_path = cfg.data_path, 
        demo_num = cfg.demo_num, 
        representations = cfg.representations, 
        env_cfg = cfg.env_cfg
    )

    fingertip_replay.save_trajectory(title='fingertip_replay_eef_motion.mp4')

def fingertip_ik_replay_ikpy(cfg):
    fingertip_replay = FingertipIKReplay(
        data_path = cfg.data_path, 
        demo_num = cfg.demo_num, 
        representations = cfg.representations, 
        env_cfg = cfg.env_cfg
    )

    fingertip_replay.save_trajectory(title='fingertip_ik_replay.mp4')


@hydra.main(version_base=None, config_path='configs', config_name='testing')
def main(cfg) -> None: 
    # cfg = cfg.fingertip_replay
    # fingertip_replay(cfg)

    # cfg = cfg.trajectory_replay
    # trajectory_replay(cfg)

    cfg = cfg.fingertip_ik_replay
    fingertip_ik_replay_ikpy(cfg)


if __name__ == '__main__': 
    main()