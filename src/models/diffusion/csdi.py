from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# * Functions.
def get_torch_trans(heads: int = 8, layers: int = 1, channels: int = 64) -> nn.Module:
    """
    Returns a Transformer encoder layer.
    Args:
        heads: Number of attention heads.
        layers: Number of layers.
        channels: Number of channels.
    Returns:
        Transformer encoder layer.
    """
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(
    in_channels: int, out_channels: int, kernel_size: int
) -> nn.modules.conv.Conv1d:
    """
    Returns a 1D convolutional layer with Kaiming normal initialization.
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
    Returns:
        1D convolutional layer.
    """
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


# * Classes
class DiffusionEmbedding(nn.Module):
    def __init__(
        self,
        num_steps: int,
        embedding_dim: int = 128,
        projection_dim: Optional[int] = None,
    ) -> None:
        """
        Diffusion embedding.
        Args:
            num_steps: Number of diffusion steps.
            embedding_dim: Embedding dimension.
            projection_dim: Projection dimension.
        """
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim

        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: int) -> torch.Tensor:
        """
        Args:
            diffusion_step: Diffusion step.
        Returns:
            Diffusion embedding.
        """
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps: int, dim: int = 64) -> torch.Tensor:
        """
        Builds the embedding table.
        Args:
            num_steps: Number of diffusion steps.
            dim: Embedding dimension.
        Returns:
            Embedding table.
        """
        steps = torch.arange(num_steps).unsqueeze(1)  # (t,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (t,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (t,dim*2)
        return table


class ResidualBlock(nn.Module):
    def __init__(
        self, side_dim: int, channels: int, diffusion_embedding_dim: int, nheads: int
    ):
        """
        Args:
            side_dim: Side dimension.
            channels: Number of channels.
            diffusion_embedding_dim: Diffusion embedding dimension.
            nheads: Number of attention heads.
        """
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y: torch.Tensor, base_shape: torch.Size) -> torch.Tensor:
        """
        Temporal transformer layer.
        Args:
            y: Input tensor.
            base_shape: Base shape.
        Returns:
            Output tensor.
        """
        b, channel, k, l = base_shape
        if l == 1:
            return y
        y = y.reshape(b, channel, k, l).permute(0, 2, 1, 3).reshape(b * k, channel, l)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(b, k, channel, l).permute(0, 2, 1, 3).reshape(b, channel, k * l)
        return y

    def forward_feature(self, y: torch.Tensor, base_shape: torch.Size) -> torch.Tensor:
        """
        Feature transformer layer.
        Args:
            y: Input tensor.
            base_shape: Base shape.
        Returns:
            Output tensor.
        """
        b, channel, k, l = base_shape
        if k == 1:
            return y
        y = y.reshape(b, channel, k, l).permute(0, 3, 1, 2).reshape(b * l, channel, k)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(b, l, channel, k).permute(0, 2, 3, 1).reshape(b, channel, k * l)
        return y

    def forward(
        self, x: torch.Tensor, cond_info: torch.Tensor, diffusion_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, channel, k, l = x.shape
        base_shape = x.shape

        x = x.reshape(b, channel, k * l)  # TODO

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (b,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (b,channel,k*l)
        y = self.mid_projection(y)  # (b,2*channel,k*l)

        cond_info = cond_info.reshape(b, cond_info.shape[1], k * l)
        cond_info = self.cond_projection(cond_info)  # (b,2*channel,k*l)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (b,channel,k*l)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        # ? Why this correction?
        return (x + residual) / math.sqrt(2.0), skip


class CSDI(nn.Module):
    """Conditional Score based Diffusion for Imputation."""

    def __init__(self, config: Dict, inputdim: int = 2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = nn.Conv1d(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(
        self, x: torch.Tensor, cond_info: torch.Tensor, diffusion_step: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (b, inputdim, k, l).
            cond_info: Conditional information of shape (b, side_dim, k, l).
            TODO diffusion_step: Diffusion step of shape (b,).
        Returns:
            Output tensor of shape (b, k, l).
        """
        b, inputdim, k, l = x.shape

        x = x.reshape(b, inputdim, k * l)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(b, self.channels, k, l)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        # ? I guess it corrects over the number of layers
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(b, self.channels, k * l)
        x = self.output_projection1(x)  # (b,channel,k*l)
        x = F.relu(x)
        x = self.output_projection2(x)  # (b,1,k*l)
        x = x.reshape(b, k, l)
        return x
