import numpy as np
import math
import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, linear, rank, init, init_range=None):
        super(LowRankLinear, self).__init__()
        self.in_dim = linear.in_features
        self.out_dim = linear.out_features
        self.init = init
        self.init_range = init_range
        self.full_rank = min(self.in_dim, self.out_dim)

        self.weight_in = nn.Parameter(
            torch.randn(self.in_dim, rank).to(linear.weight),
            requires_grad=True,
        )
        self.weight_out = nn.Parameter(
            torch.randn(self.out_dim, rank).to(linear.weight),
            requires_grad=True,
        )

        self.bias = (
            nn.Parameter(
                torch.zeros(self.out_dim).to(linear.weight),
                requires_grad=True,
            )
            if hasattr(linear, "bias") and linear.bias is not None
            else None
        )

        self._init_weight()

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, rank={self.rank}, init={self.init}"

    def _init_weight(self):
        if self.init == "xavier":
            nn.init.xavier_normal_(self.weight_in)
            nn.init.xavier_normal_(self.weight_out)

        elif self.init == "auto":
            assert self.init_range is not None
            std = math.sqrt(math.sqrt(self.init_range**2 / self.rank))
            nn.init.normal_(self.weight_in, mean=0, std=std)
            nn.init.normal_(self.weight_out, mean=0, std=std)

        elif self.init == "xavorth":
            dtype = self.weight_in.dtype
            nn.init.orthogonal_(self.weight_in.float()).to(dtype)
            nn.init.orthogonal_(self.weight_out.float()).to(dtype)
            self.weight_in.data *= np.sqrt(2 / (self.in_dim + self.rank))
            self.weight_out.data *= np.sqrt(2 / (self.out_dim + self.rank))

        elif self.init == "kaiming":
            nn.init.kaiming_normal_(self.weight_in)
            nn.init.kaiming_normal_(self.weight_out)

        elif self.init == "orth":
            dtype = self.weight_in.dtype
            nn.init.orthogonal_(self.weight_in.float()).to(dtype)
            nn.init.orthogonal_(self.weight_out.float()).to(dtype)

        elif self.init.startswith("randn"):
            std = float(self.init.split("_")[-1])
            nn.init.normal_(self.weight_in, mean=0, std=std)
            nn.init.normal_(self.weight_out, mean=0, std=std)

        elif "const" in self.init:
            const = float(self.init.split("_")[-1])
            assert const != 0
            self.weight_in.data.fill_(const)
            self.weight_out.data.fill_(const)

        else:
            raise ValueError(f"Invalid init method: {self.init}")

    @property
    def rank(self):
        return min(min(self.weight_in.shape), min(self.weight_out.shape))

    def forward(self, x):
        out = x @ self.weight_in @ self.weight_out.T
        if self.bias is not None:
            out = out + self.bias

        return out


def apply_lowrank_param(
    model,
    model_config,
    model_type,
    scope,
    attn_rank,
    mlp_rank,
    init,
    verbose=True,
):
    import gc

    full_param = sum(p.numel() for p in model.parameters())

    if model_type == "llama":

        init_range = model_config.initializer_range

        module_names_dict = {
            "all": ["self_attn", "mlp"],
            "attn": ["self_attn"],
            "mlp": ["mlp"],
        }
        rank_dict = {
            "self_attn": attn_rank,
            "mlp": mlp_rank,
        }
        module_names = module_names_dict[scope]

        for i, layer in enumerate(model.model.layers):
            for m_name in module_names:
                module = getattr(layer, m_name)
                rank = rank_dict[m_name]
                for name, sub_module in module.named_modules():
                    if isinstance(sub_module, nn.Linear):
                        setattr(
                            module,
                            name,
                            LowRankLinear(sub_module, rank, init, init_range),
                        )
                        if verbose:
                            print(
                                f"layer.{i}.{m_name}.{name}: {sub_module} --> {getattr(module, name)}"
                            )
                        del sub_module
                        torch.cuda.empty_cache()
                        gc.collect()

    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented")

    lowrank_param = sum(p.numel() for p in model.parameters())
    print(
        f"\nLowRankLinear modules are set successfully!\n"
        f"Self-attention rank: {attn_rank}, MLP rank: {mlp_rank}.\n"
        f"Full param: {full_param / 1e6} M = {full_param / 1e9} G\n"
        f"Low-rank param: {lowrank_param / 1e6} M = {lowrank_param / 1e9} G\n"
        f"Cprs rate: {lowrank_param / full_param:.5%}\n"
    )


def get_lowrank_param(model, model_config, lr_scaler=-1):
    # lr_scaler = -1: use adaptive lr_scaler = r / d, see ./loro_torch/loro_optim.py
    lowrank_params_in = []
    lowrank_params_out = []
    lowrank_params_in_type = []
    lowrank_params_out_type = []

    for name, module in model.named_modules():
        if isinstance(module, LowRankLinear):
            if "weight_in" in module.state_dict():
                lowrank_params_in.append(module.weight_in)
                assert any([t in name for t in ["self_attn", "mlp"]])
                if "mlp" in name:
                    lowrank_params_in_type.append("mlp")
                elif "self_attn" in name:
                    lowrank_params_in_type.append("attn")

            if "weight_out" in module.state_dict():
                lowrank_params_out.append(module.weight_out)
                assert any([t in name for t in ["self_attn", "mlp"]])
                if "mlp" in name:
                    lowrank_params_out_type.append("mlp")
                elif "self_attn" in name:
                    lowrank_params_out_type.append("attn")

    id_lowrank_params = [id(p) for p in lowrank_params_in + lowrank_params_out]
    regular_params = [p for p in model.parameters() if id(p) not in id_lowrank_params]

    init_std = model_config.initializer_range
    hidden_size = model_config.hidden_size
    intermediate_size = model_config.intermediate_size

    param_groups = [
        {
            "type": "regular",
            "params": regular_params,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "init_std": init_std,
        },
        {
            "type": "lowrank_in",
            "params": lowrank_params_in,
            "params_type": lowrank_params_in_type,
            "lr_scaler": lr_scaler,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "init_std": init_std,
        },
        {
            "type": "lowrank_out",
            "params": lowrank_params_out,
            "params_type": lowrank_params_out_type,
            "lr_scaler": lr_scaler,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "init_std": init_std,
        },
    ]

    return param_groups
