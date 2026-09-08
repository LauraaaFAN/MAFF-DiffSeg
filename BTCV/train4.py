import numpy as np
from dataset.btcv_transunet_datasetings import get_loader_btcv
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss
from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
import os
import torch.distributed as dist

# Set deterministic behavior for reproducibility
set_determinism(123)

# Global configurations
data_dir = "./RawData/Training/"
logdir = "./logs_btcv/diffunet_transunet_datasettings/"
model_save_path = os.path.join(logdir, "model")
max_epoch = 3000
batch_size = 1
val_every = 100
env = "DDP"
num_gpus = 4

def setup_distributed():
    """Set up distributed environment and return device and local_rank."""
    if not dist.is_initialized():  # Avoid multiple initializations
        local_rank = int(os.environ["LOCAL_RANK"])  # Automatically set by torchrun
        torch.cuda.set_device(local_rank)  # Assign GPU to current process
        dist.init_process_group(backend="nccl")
        print(f"Process {dist.get_rank()} initialized on GPU {local_rank}")
    else:
        local_rank = torch.cuda.current_device()
    return torch.device(f"cuda:{local_rank}"), local_rank

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, 1, 2, [64, 64, 128, 256, 512, 64])
        self.model = BasicUNetDe(3, 14, 13, [64, 64, 128, 256, 512, 64], 
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [1000]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [10]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise
        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)
        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.model, (1, 13, 96, 96, 96),
                model_kwargs={"image": image, "embeddings": embeddings}
            )
            return sample_out["pred_xstart"]

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device, local_rank, val_every, num_gpus, logdir):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir)
        
        # Initialize model and distributed settings
        self.model = DiffUNet().to(device)
        if env_type == "DDP" and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if env_type == "DDP":
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])

        # Initialize loss, optimizer, and scheduler
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-3)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=100, max_epochs=max_epochs)
        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.5)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        x_start = (label * 2) - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)
        loss_mse = self.mse(torch.sigmoid(pred_xstart), label)
        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)
        return loss

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
        label = self.convert_labels(label).float()
        return image, label

    def convert_labels(self, labels):
        return torch.cat([(labels == i).float() for i in range(1, 14)], dim=1)

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = torch.sigmoid(output).cpu().numpy()
        target = label.cpu().numpy()
        
        dices = [dice(output[:, i], target[:, i]) for i in range(13)]
        return dices

    def validation_end(self, mean_val_outputs):
        mean_dice = sum(mean_val_outputs) / len(mean_val_outputs)
        self.log("mean_dice", mean_dice, step=self.epoch)
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, f"{model_save_path}/best_model_{mean_dice:.4f}.pt", "best_model")
        save_new_model_and_delete_last(self.model, f"{model_save_path}/final_model_{mean_dice:.4f}.pt", "final_model")
        print(f"Mean dice score: {mean_dice}")

if __name__ == "__main__":
    device, local_rank = setup_distributed()
    trainer = BraTSTrainer(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        local_rank=local_rank,
        val_every=val_every,
        num_gpus=num_gpus,
        logdir=logdir
    )
    train_ds, val_ds, test_ds = get_loader_btcv(data_dir)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
