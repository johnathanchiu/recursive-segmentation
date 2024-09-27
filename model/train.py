import torch


# Sample data
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Training
model = SimpleModel()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader)
