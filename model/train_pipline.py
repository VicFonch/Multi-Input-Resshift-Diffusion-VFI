import os
import copy
import matplotlib.pyplot as plt
from typing import Any

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from lightning import LightningModule

from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS

from model.model import MultiInputResShift

from utils.utils import denorm, make_grid_images#, save_triplet 
from utils.ema import EMA
from utils.inter_frame_idx import get_inter_frame_temp_index
from utils.raft import raft_flow


class TrainPipline(LightningModule):
    def __init__(self, 
                 confg: dict, 
                 test_dataloader: DataLoader):
        super(TrainPipline, self).__init__()

        self.test_dataloader = test_dataloader

        self.confg = confg

        self.mean, self.sd = confg["data_confg"]["mean"], confg["data_confg"]["sd"]

        self.model = MultiInputResShift(**confg["model_confg"])

        self.ema = EMA(beta=0.995)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        self.charbonnier_loss = lambda x, y: torch.mean(torch.sqrt((x - y)**2 + 1e-6))
        self.lpips_loss = LPIPS(net_type='vgg')

        self.train_metrics = MetricCollection({
            "train_lpips": LPIPS(net_type='alex'),
            "train_psnr": PSNR(),
            "train_ssim": SSIM()
        })
        self.val_metrics = MetricCollection({
            "val_lpips": LPIPS(net_type='alex'),
            "val_psnr": PSNR(),
            "val_ssim": SSIM()
        })

    def loss_fn(self, 
                x: torch.Tensor, 
                predicted_x: torch.Tensor) -> torch.Tensor:
        percep_loss = 0.2 * self.lpips_loss(x, predicted_x.clamp(-1, 1))
        pix2pix_loss = self.charbonnier_loss(x, predicted_x)
        return percep_loss + pix2pix_loss

    def sample_t(self, 
                 shape: tuple[int, ...], 
                 max_t: int, 
                 device: torch.device) -> torch.Tensor:
        p = torch.linspace(1, max_t, steps=max_t, device=device) ** 2 
        p = p / p.sum()  
        t = torch.multinomial(p, num_samples=shape[0], replacement=True)
        return t

    def forward(self, 
                I0: torch.Tensor, 
                It: torch.Tensor, 
                I1: torch.Tensor) -> torch.Tensor:
        flow0tot = raft_flow(I0, It, 'animation')
        flow1tot = raft_flow(I1, It, 'animation')
        mid_idx = get_inter_frame_temp_index(I0, It, I1, flow0tot, flow1tot).to(It.dtype)

        tau = torch.stack([mid_idx, 1 - mid_idx], dim=1)
        
        if self.current_epoch > 5:
            t = torch.randint(low=1, high=self.model.timesteps, size=(It.shape[0],), device=It.device, dtype=torch.long)
        else:
            t = self.sample_t(shape=(It.shape[0],), max_t=self.model.timesteps, device=It.device)

        predicted_It = self.model(I0, It, I1, tau=tau, t=t)
        return predicted_It

    def get_step_plt_images(self, 
                            It: torch.Tensor, 
                            predicted_It: torch.Tensor) -> plt.Figure:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(denorm(predicted_It.clamp(-1, 1), self.mean, self.sd)[0].permute(1, 2, 0).cpu().numpy())
        ax[0].axis("off")
        ax[0].set_title("Predicted")
        ax[1].imshow(denorm(It, self.mean, self.sd)[0].permute(1, 2, 0).cpu().numpy())
        ax[1].axis("off")
        ax[1].set_title("Ground Truth")
        plt.tight_layout() 
        #img_path = "step_image.png"
        #fig.savefig(img_path, dpi=300, bbox_inches='tight') 
        plt.close(fig)
        return fig

    def training_step(self, batch: tuple[torch.Tensor, ...], _) -> torch.Tensor:
        I0, It, I1 = batch
        predicted_It = self(I0, It, I1)
        loss = self.loss_fn(It, predicted_It)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        self.ema.step_ema(self.ema_model, self.model)
        with torch.inference_mode():
            fig = self.get_step_plt_images(It, predicted_It)
            self.logger.experiment.add_figure("Train Predictions", fig, self.global_step)
            mets = self.train_metrics(It, predicted_It.clamp(-1, 1))
            self.log_dict(mets, prog_bar=True, on_step=True,on_epoch=False)
        return loss
    
    @torch.no_grad()
    def validation_step(self,  batch: tuple[torch.Tensor, ...], _) -> None:
        I0, It, I1 = batch
        predicted_It = self(I0, It, I1)
        loss = self.loss_fn(It, predicted_It)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        mets = self.val_metrics(It, predicted_It.clamp(-1, 1))
        self.log_dict(mets, prog_bar=True, on_step=False, on_epoch=True)

    @torch.inference_mode()
    def on_train_epoch_end(self) -> None:
        torch.save(self.ema_model.state_dict(), 
                   os.path.join("_checkpoint", f"resshift_diff_{self.current_epoch}.pth"))

        batch = next(iter(self.test_dataloader))
        I0, It, I1 = batch
        I0, It, I1 = I0.to(self.device), It.to(self.device), I1.to(self.device)

        flow0tot = raft_flow(I0, It, 'animation')
        flow1tot = raft_flow(I1, It, 'animation')
        mid_idx = get_inter_frame_temp_index(I0, It, I1, flow0tot, flow1tot).to(It.dtype)
        tau = torch.stack([mid_idx, 1 - mid_idx], dim=1)

        predicted_It = self.ema_model.reverse_process([I0, I1], tau)

        I0 = denorm(I0, self.mean, self.sd)
        I1 = denorm(I1, self.mean, self.sd)
        It = denorm(It, self.mean, self.sd)
        predicted_It = denorm(predicted_It.clamp(-1, 1), self.mean, self.sd)

        #save_triplet([I0, It, predicted_It, I1], f"./_output/target_{self.current_epoch}.png", nrow=1)
        grid = make_grid_images([I0, It, predicted_It, I1], nrow=1)
        self.logger.experiment.add_image("Predicted Images", grid, self.global_step)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = [AdamW(
                        self.model.parameters(),
                        **self.confg["optim_confg"]['optimizer_confg']
                    )]
         
        scheduler = [{
            'scheduler': ReduceLROnPlateau(
                optimizer[0],
                **self.confg["optim_confg"]['scheduler_confg']
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
            'strict': True,
        }] 
                    
        return optimizer, scheduler

