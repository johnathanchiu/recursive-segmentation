import pytorch_lightning as pl
from torch.utils.data import DataLoader

from model.seg.data import BoundingBoxDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, json_file: str, batch_size: int = 32):
        super().__init__()
        self.json_file_path = json_file
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train_data = BoundingBoxDataset(self.json_file_path)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)
