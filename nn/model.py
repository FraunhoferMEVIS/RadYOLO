from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import torch
from torch import nn

from nn.layers import Detect
from nn.util import initialize_weights, intersect_dicts, parse_model


class BaseModel(nn.Module, ABC):
    @property
    @abstractmethod
    def model(self) -> nn.Sequential: ...

    @property
    @abstractmethod
    def save(self) -> list[int]: ...

    def forward(self, x, *args, **kwargs):
        return self.predict(x, *args, **kwargs)

    def predict(
        self,
        x: torch.Tensor,
        # profile: bool = False,
        # visualize: bool = False,
        augment: bool = False,
        # embed: list[int] | None = None,
    ) -> torch.Tensor:
        if augment:
            return self._predict_augment(x)
        return self._predict_once(
            x,
            # profile,
            # visualize,
            # embed,
        )

    def _predict_once(
        self,
        x_in: torch.Tensor,
        # profile: bool = False,
        # visualize: bool = False,
        # embed: list[int] | None = None,
    ) -> torch.Tensor:
        x: torch.Tensor | list[torch.Tensor] = x_in
        y: torch.Tensor = x
        ys: list[torch.Tensor | None] = []
        # dt = []
        # embeddings: list[torch.Tensor] = []
        for m in self.model:
            if m.f != -1:
                if isinstance(m.f, int):
                    x_ = ys[m.f]
                    if x_ is None:
                        raise ValueError  # TODO add error message
                    x = x_
                else:
                    x_prev = x
                    x = []

                    for j in m.f:
                        j: int
                        if j == -1:
                            if not isinstance(x_prev, torch.Tensor):
                                # the idea is to break a potential recursion of adding lists to
                                # lists, this should not happen
                                raise ValueError  # TODO add error message
                            x.append(x_prev)
                        else:
                            x_ = ys[j]
                            if x_ is None:
                                raise ValueError  # TODO add error message
                            x.append(x_)

            # if profile:
            #     self._profile_one_layer(m, x_, dt)
            x = y = m(x)
            ys.append(y if m.i in self.save else None)
            # if visualize:
            #     feature_visualization(x_, m.type, m.i, save_dir=visualize)
            # if embed and m.i in embed:
            #     embeddings.append(
            #         nn.functional.adaptive_avg_pool2d(
            #             y,
            #             (
            #                 1,
            #                 1,
            #             ),
            #         )
            #         .squeeze(-1)
            #         .squeeze(-1)
            #     )
            #     if m.i == max(embed):
            #         return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return y

    def _predict_augment(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_once(x)

    def load(self, weights: dict | nn.Module, verbose: bool = True) -> None:
        model: nn.Module = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)


class DetectionModel(BaseModel):
    def __init__(
        self,
        cfg: str | dict = "yolov11n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc
        self._model, self._save = parse_model(
            deepcopy(self.yaml), input_channels=ch, verbose=verbose
        )
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )
            self.stride = m.stride
            # m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)

    @property
    def model(self) -> nn.Sequential:
        return self._model

    @property
    def save(self) -> list[int]:
        return self._save

    def _predict_augment(self, x: torch.Tensor) -> torch.Tensor:
        # img_size = x.shape[-2:]
        # s = [1, 0.83, 0.67]
        # f = [None, 3, None]
        f = [None, 3]  # TODO check axis for 3D
        y = []
        # for si, fi in zip(s, f):
        for fi in f:
            xi = x.flip(fi) if fi else x
            yi = super().predict(xi)  # TODO check indexing
            y.append(yi)
        return torch.cat(y, -1)  # TODO check indexing
