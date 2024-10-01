import pytorch_lightning as pl

from model.seg.dloader import SegmentationDataModule
from model.seg.rseg import SegmentationModel

# Training
model = SegmentationModel()
trainer = pl.Trainer()

imagenet = SegmentationDataModule()
trainer.fit(model, datamodule=imagenet)
