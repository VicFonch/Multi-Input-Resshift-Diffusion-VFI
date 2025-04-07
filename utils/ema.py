import torch
import torch.nn as nn

class EMA:
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, ema_model in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, up_weight = ema_model.data, current_params.data
            ema_model.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 2000) -> None:
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def copy_to(self, ema_model: nn.Module, model: nn.Module) -> None:
        model.load_state_dict(ema_model.state_dict())

    def reset_parameters(self, ema_model: nn.Module, model: nn.Module) -> None:
        ema_model.load_state_dict(model.state_dict())