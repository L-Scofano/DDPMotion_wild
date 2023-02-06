from model_base import ModelBase


class Model_H36M(ModelBase):
    def __init__(self, config, device, target_dim=96):
        super(Model_H36M, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        pose = batch["pose"].to(self.device).float() #! (Bs, L, DIMs) (L=pose timesteps)
        tp = batch["timepoints"].to(self.device).float() #! (Bs, L) (L=pose timesteps)
        mask = batch["mask"].to(self.device).float() #! (Bs, L, DIMs) (L=pose timesteps)

        pose = pose.permute(0, 2, 1) #! (Bs, DIMs, L) (L=pose timesteps)
        mask = mask.permute(0, 2, 1) #! (Bs, DIMs, L) (L=pose timesteps)

        return (
            pose,
            tp,
            mask
        )
