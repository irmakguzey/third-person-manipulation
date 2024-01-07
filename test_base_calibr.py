import hydra

from third_person_man.testing import *
from third_person_man.calibration import *



def baseframe_replay(cfg):
    baseframe_replay = BaseframeReplay(
        cfg = cfg
    )
    baseframe_replay.save_baseframe_trajectory()
    # baseframe_replay.save_end_effector_trajectory()
    baseframe_replay.save_fingertip_trajectory()


@hydra.main(version_base=None, config_path='configs', config_name='testing')
def main(cfg) -> None: 
    cfg = cfg.base_calibr
    baseframe_replay(cfg)

if __name__ == '__main__': 
    main()