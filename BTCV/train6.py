import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from monai.losses import DiceLoss
import os
import torch.distributed as dist
from dataset.btcv_transunet_datasetings import get_loader_btcv  # 确保路径正确
from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.trainer import Trainer
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

# 设置随机种子
set_determinism(123)

# 路径和训练参数
data_dir = "./RawData/Training/"
logdir = "./logs_btcv/diffunet_transunet_datasettings/"
model_save_path = os.path.join(logdir, "model")

max_epoch = 3000
batch_size = 1
val_every = 100
env = "DDP"
num_gpus = 4


def setup_distributed():
    """
    初始化分布式训练的设置
    """
    assert "LOCAL_RANK" in os.environ, "LOCAL_RANK 未设置，确保使用 torchrun 启动脚本。"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    print(f"Process rank {local_rank} initialized on GPU {local_rank}")
    return torch.device(f"cuda:{local_rank}")


class DiffUNet(nn.Module):
    def __init__(self):
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
            loss_type=LossType.MSE
        )

        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [10]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE
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
                self.model, (1, 13, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings})
            return sample_out["pred_xstart"]


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_port):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_port)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.5)
        self.model = DiffUNet().to(device)

        if env_type == "DDP":
            local_rank = int(os.environ["LOCAL_RANK"])
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-3)
        self.dice_loss = DiceLoss(sigmoid=True)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=100, max_epochs=max_epochs)


if __name__ == "__main__":
    device = setup_distributed()
    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           logdir=logdir,
                           master_port=17751)

    train_ds, val_ds, test_ds = get_loader_btcv(data_dir=data_dir)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
