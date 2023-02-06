from typing import Callable, Dict, Tuple
import math

import torch
import torch.nn as nn
import numpy as np
from src.models.diffusion.csdi import CSDI


class ModelH36M(nn.Module):
    def __init__(
        self,
        config: Dict,
        device: torch.device,
        target_dim: int = 96,
    ) -> None:
        """
        Args:
            config: The configuration dictionary.
            device: The device to use.
            target_dim: The dimension of the target. Defaults to 96.
        """
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.conditional = config["model"]["conditional"]

        # ! We just concatenate the time and feature embeddings.
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim

        if self.conditional:
            # For conditional mask.
            # ! I'd say it is for actions.
            self.emb_total_dim += 1
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        # ? No idea why is called `side_dim`.
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 2 if self.conditional else 1
        self.diffmodel = CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        # Variance scheduling.
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        elif config_diff["schedule"] == "cosine":
            self.beta = self.betas_for_alpha_bar(
                num_diffusion_timesteps=self.num_steps,
                alpha_bar=lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
                max_beta=0.5,
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    def betas_for_alpha_bar(
        self, num_diffusion_timesteps: int, alpha_bar: Callable, max_beta: float = 0.008
    ) -> np.ndarray:
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].
        Args:
            num_diffusion_timesteps: The number of betas to produce.
            alpha_bar: A lambda that takes an argument t from 0 to 1 and
                       produces the cumulative product of (1-beta) up to that
                       part of the diffusion process.
            max_beta: The maximum beta to use, values should be lower than 1 to
                      prevent singularities; the default in the paper is 0.999.
                      Defaults to 0.008.
        Returns:
            The beta schedule.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def time_embedding(self, pos: torch.Tensor, d_model: int = 128) -> torch.Tensor:
        """
        Build sinusoidal positional time embeddings.
        Args:
            pos: The positions to embed.
            d_model: The dimension of the embedding.
        Returns:
            The embeddings.
        """
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(
        self, observed_tp: torch.Tensor, cond_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine the time and feature embeddings.
        Args:
            observed_tp: The observed time points.
            cond_mask: The conditional mask.
        Returns:
            The side information.
            ? Why is it called like that?
        """
        b, k, l = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, k, -1)

        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (k, emb)
        feature_embed = feature_embed[None, None, ...].expand(b, l, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (b,l,k,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (b,*,k,l)

        if self.conditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        side_info: torch.Tensor,
        train: bool,
    ):
        """TODO: is it really needed?"""
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(observed_data, cond_mask, side_info, train, set_t=t)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        side_info: torch.Tensor,
        train: bool,
        set_t: int = -1,
    ):
        """
        Calculate the loss.
        Args:
            observed_data: The observed data.
            cond_mask: The conditional mask.
            side_info: The side information.
            train: Whether we are training or validating.
            set_t: The timestep to use for validation.
        Returns:
            The loss.
        """
        b, k, l = observed_data.shape

        if train:
            # Sample random timesteps.
            t = torch.randint(0, self.num_steps, [b]).to(self.device)
        else:
            # Validation. Timesteps are fixed to -1.
            # ? `set_t` should be a set time, don't know why is -1.
            t = (torch.ones(b) * set_t).long().to(self.device)

        # Noise input.
        current_alpha = self.alpha_torch[t].to(self.device)  # (b,1,1)
        noise = torch.randn_like(observed_data).to(self.device)
        noisy_data = (current_alpha**0.5) * observed_data + (
            1.0 - current_alpha
        ) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        # ! Slow.
        predicted = self.diffmodel(total_input, side_info, t)  # (b,k,l)

        target_mask = 1 - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()  # ? No idea what this is.
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(
        self,
        noisy_data: torch.Tensor,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Set the input to the diffusion model.
        Args:
            noisy_data: The noisy data.
            observed_data: The observed data.
            cond_mask: The conditional mask.
            Returns:
                The input to the diffusion model.
        """
        if self.conditional:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            # ? Not clear why.
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (b,2,k,l)
        else:
            total_input = noisy_data.unsqueeze(1)  # (b,1,k,l)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if not self.conditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                        t
                    ] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if not self.conditional:
                    diff_input = (
                        cond_mask * noisy_cond_history[t]
                        + (1.0 - cond_mask) * current_sample
                    )
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(
                    diff_input, side_info, torch.tensor([t]).to(self.device)
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = (
                current_sample * (1 - cond_mask) + observed_data * cond_mask
            ).detach()
        return imputed_samples

    def process_data(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the data from the batch into the correct format for the model.
        Args:
            batch: The batch of data to process.
        Returns:
            pose: The pose data.
            tp: The timepoints.
            mask: The mask.
        """
        pose = batch["pose"].to(self.device).float().permute(0, 2, 1)
        tp = batch["timepoints"].to(self.device).float()
        # TODO Not clear what the mask is used for. If I don't use it it has shape (32,66,55)
        mask = batch["mask"].to(self.device).float().permute(0, 2, 1)
        return (pose, tp, mask)

    def forward(self, batch: Dict, train: bool = True) -> torch.Tensor:
        """
        Args:
            batch: The batch of data to process.
            train: Whether to train or not.
        Returns:
            The loss.
        """
        observed_data, observed_tp, cond_mask = self.process_data(batch)
        # ! Computing it was slow, took 0.71s.
        side_info = self.get_side_info(observed_tp, cond_mask)

        # Get the respective loss for train or validation.
        loss_fn = self.calc_loss if train else self.calc_loss_valid
        # ! Slow.
        return loss_fn(observed_data, cond_mask, side_info, train)

    def evaluate(self, batch, n_samples):
        (observed_data, observed_tp, gt_mask) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = 1 - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        return samples, observed_data, target_mask, observed_tp
