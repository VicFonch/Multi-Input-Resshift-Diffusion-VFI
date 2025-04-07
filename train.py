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
    parser.add_argument("--train_dir", type=str, default="_data/ATD-12K/train")
    parser.add_argument("--val_dir", type=str, default="_data/ATD-12K/val")
    parser.add_argument("--test_dir", type=str, default="_data/ATD-12K/test")
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

    train_dataset = TripletImagesDataset(args.train_dir, **data_confg)
    train_dataloader = DataLoader(train_dataset, batch_size=data_confg['train_batch_size'], shuffle=True)
    val_dataset = TripletImagesDataset(args.val_dir, **data_confg)
    val_dataloader = DataLoader(val_dataset, batch_size=data_confg['val_batch_size'], shuffle=False)
    test_dataset = TripletImagesDataset(args.test_dir, **data_confg)
    test_dataloader = DataLoader(test_dataset, batch_size=data_confg['test_batch_size'], shuffle=True)

    model = TrainPipline(config, test_dataloader)

    trainer.fit(model, train_dataloader, val_dataloader)