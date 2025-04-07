import os
import yaml

from argparse import ArgumentParser

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from model.train_pipline import TrainPipline

from torch.utils.data import DataLoader
from datamodule.datamodule import TripletImagesDataset

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/confg.yaml")
    parser.add_argument("--data_path", type=str, default="_data/ATD-12K")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    trainer = Trainer(
        logger=TensorBoardLogger("tb_logs", name="Diffusion_Animation_Training"),
        **config['trainer_confg']
    )   

    data_confg = config['data_confg']

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    test_dir = os.path.join(args.data_path, "test")

    train_dataset = TripletImagesDataset(train_dir, **data_confg)
    train_dataloader = DataLoader(train_dataset, batch_size=data_confg['train_batch_size'], shuffle=True)
    val_dataset = TripletImagesDataset(val_dir, **data_confg)
    val_dataloader = DataLoader(val_dataset, batch_size=data_confg['val_batch_size'], shuffle=False)
    test_dataset = TripletImagesDataset(test_dir, **data_confg)
    test_dataloader = DataLoader(test_dataset, batch_size=data_confg['test_batch_size'], shuffle=True)

    train_pipline = TrainPipline(config, test_dataloader)

    trainer.fit(train_pipline, train_dataloader, val_dataloader)