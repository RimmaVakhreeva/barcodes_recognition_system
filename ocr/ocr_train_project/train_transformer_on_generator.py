import random
from itertools import zip_longest

import numpy as np
import torch
import wandb
from torchsummary import summary

from crnn import TransformerOcr
from generator.transformer_dataset_generator import BarcodeDataset, TestBarcodeDataset

# Initialize Weights and Biases for experiment tracking
run = wandb.init(project="barcode_detection_recognition")
config = wandb.config
# Setting configuration variables for the project
wandb.config.vocab = '0123456789'  # Characters to recognize
wandb.config.seed = 44  # Seed for reproducibility
wandb.config.bs = 128  # Batch size
wandb.config.epoch_size = 1024 * 32  # Number of samples per epoch
wandb.config.lr = 1e-3  # Learning rate
wandb.config.epochs = 48  # Number of training epochs

# Seed the random number generators for consistency
random.seed(wandb.config.seed)
np.random.seed(wandb.config.seed)
torch.manual_seed(wandb.config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(wandb.config.seed)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Initialize the OCR model with specified architecture parameters
model = TransformerOcr(
    cnn_backbone_name='resnet18d',
    cnn_backbone_pretrained=True,
    cnn_output_size=4608,
    transformer_features_num=128,
    transformer_dropout=0.1,
    transformer_nhead=32,
    transformer_num_layers=2,
    num_classes=len(wandb.config.vocab) + 2  # Include classes for characters and special symbols
)
model.train()  # Set the model to training mode
summary(model=model, depth=4)  # Print a summary of the model
wandb.watch(model, log_freq=10, log="parameters", log_graph=True)  # Log model parameters and graph to wandb

# Data loaders for training and testing datasets
barcode_loader = torch.utils.data.DataLoader(
    BarcodeDataset(epoch_size=wandb.config.epoch_size, vocab=wandb.config.vocab),
    batch_size=wandb.config.bs,
    num_workers=30
)
test_barcode_loader = torch.utils.data.DataLoader(
    TestBarcodeDataset(vocab=wandb.config.vocab, directory='../ocr/ocr_train_project/test_images'),
    batch_size=32
)



# Optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
lambda1 = lambda epoch: 1 - epoch / (800 * 32)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# Device configuration for GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Function to calculate the accuracy of predictions
def calculate_accuracy(gt_texts, pred_texts):
    accuracy = []
    for gt_text, pred_text in zip(gt_texts, pred_texts):
        correct, text_length = 0, 0
        for gt_ch, pred_ch in zip_longest(gt_text, pred_text):
            text_length += 1
            if gt_ch == pred_ch:
                correct += 1
        accuracy.append(correct / text_length)
    return np.mean(accuracy)


# Main training loop
if __name__ == '__main__':
    model.to(device)
    model.train()

    for epoch in range(wandb.config.epochs):
        for iter_n, (input_images, targets, gt_texts) in enumerate(barcode_loader):
            input_images = input_images.to(device)
            targets = targets.to(device)

            output = model(input_images)
            pred_texts = model.decode_output(output, wandb.config.vocab)
            train_acc = calculate_accuracy(gt_texts, pred_texts)

            loss = criterion(output.permute(1, 0, 2).reshape(-1, model.num_classes), targets.reshape(-1, ))
            if loss.isinf() or loss.isnan():
                print(f"loss is inf or nan, skipping the batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                test_losses = []
                for input_images, targets, gt_texts in test_barcode_loader:
                    input_images = input_images.to(device)
                    targets = targets.to(device)

                    output = model(input_images)
                    pred_texts = model.decode_output(output, wandb.config.vocab)
                    test_acc = calculate_accuracy(gt_texts, pred_texts)

                    loss = criterion(output.permute(1, 0, 2).reshape(-1, model.num_classes), targets.reshape(-1, ))
                    if not (loss.isinf() or loss.isnan()):
                        test_losses.append(loss)
                test_loss = sum(test_losses) / len(test_losses)
            model.train()

            run.log({
                "epoch": epoch,
                "iter_n": iter_n,
                "train_loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            })
            print(f"epoch {epoch}\t"
                  f"loss {loss.item():.5f}\t"
                  f"lr: {optimizer.param_groups[0]['lr']:.5f}\t"
                  f"train_acc: {train_acc:.5f}\t"
                  f"test_loss: {test_loss:.5f}\t"
                  f"test_acc: {test_acc:.5f}")

            cp = model.state_dict()
            torch.save(cp, "../../crnn_last.pt")
