if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hydra
import torch as t

from policy_generator.model.common.normalizer import LinearNormalizer
from policy_generator.policy.policy_generator import PolicyGenerator

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model: PolicyGenerator = hydra.utils.instantiate(cfg.policy)
    def run(self):
        normalizer = LinearNormalizer()
        data = {}
        data['trajectory'] = t.randn((10, self.cfg.shape_meta['trajectory']['shape'][0]))
        data['params'] = t.randn((10, self.cfg.shape_meta['params']['shape'][0]))
        normalizer.fit(data, last_n_dims=1, mode='limits')
        self.model.set_normalizer(normalizer)
    def predict(self, trajectory):
        traj_dict = {}
        traj_dict["trajectory"] = trajectory
        parameters = self.model.predict_paremeters(traj_dict)
        return parameters


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'policy_generator', 'config'))
)
def main(cfg):
    workspace = Workspace(cfg)
    input_dim = cfg.shape_meta['trajectory']['shape'][0]
    workspace.run()
    trajectory = t.randn((1,1,input_dim))
    workspace.predict(trajectory)


if __name__ == "__main__":
    main()