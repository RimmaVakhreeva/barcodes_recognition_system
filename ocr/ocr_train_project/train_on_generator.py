import random
from itertools import zip_longest

import numpy as np
import torch
import wandb  # Weights & Biases for tracking experiments
from torchsummary import summary

from crnn import CRNN  # Import the CRNN model
from generator.dataset_generator import BarcodeDataset, TestBarcodeDataset  # Custom dataset loaders

# Initialize a Weights & Biases run
run = wandb.init(project="barcode_detection_recognition")
config = wandb.config  # Config object for hyperparameters
# Set hyperparameters in wandb config
wandb.config.vocab = '0123456789'  # Vocabulary for decoding outputs
wandb.config.seed = 44  # Seed for reproducibility
wandb.config.bs = 128  # Batch size
wandb.config.epoch_size = 1024 * 32  # Size of an epoch
wandb.config.lr = 1e-3  # Learning rate
wandb.config.epochs = 48  # Number of epochs

# Seed setting for reproducibility across multiple libraries
random.seed(wandb.config.seed)
np.random.seed(wandb.config.seed)
torch.manual_seed(wandb.config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(wandb.config.seed)

# Loss function configuration
criterion = torch.nn.CTCLoss(zero_infinity=True)  # CTC loss for sequence modeling

# Initialize the CRNN model with specified configurations
crnn = CRNN(
    cnn_backbone_name='resnet18d',  # CNN backbone model
    cnn_backbone_pretrained=True,  # Use a pretrained model
    cnn_output_size=4608,  # Expected CNN output size
    rnn_features_num=128,  # Number of RNN features
    rnn_dropout=0.1,  # Dropout rate in RNN
    rnn_bidirectional=True,  # Use bidirectional RNN
    rnn_num_layers=2,  # Number of RNN layers
    num_classes=11  # Number of classes (digits + blank for CTC)
)
crnn.train()  # Set model to training mode
summary(model=crnn, depth=4)  # Print a summary of the model with a depth of 4
wandb.watch(crnn, log_freq=10, log="parameters", log_graph=True)  # Log model parameters to wandb

# Setup data loaders for training and testing
barcode_loader = torch.utils.data.DataLoader(
    BarcodeDataset(epoch_size=wandb.config.epoch_size, vocab=wandb.config.vocab),
    batch_size=wandb.config.bs,
    num_workers=1  # Number of workers for loading data
)

test_barcode_loader = torch.utils.data.DataLoader(
    TestBarcodeDataset(vocab=wandb.config.vocab, directory='./test_images'),
    batch_size=32  # Batch size for testing
)

#"/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/ocr/ocr_train_project"

# Optimizer and learning rate scheduler setup
optimizer = torch.optim.Adam(crnn.parameters(), lr=wandb.config.lr)
lambda1 = lambda epoch: 1 - epoch / (700 * 32)  # Decay function for learning rate
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# Determine device (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_accuracy(gt_texts, pred_texts):
    """ Calculate accuracy by comparing ground truth texts with predicted texts """
    accuracy = []
    for gt_text, pred_text in zip(gt_texts, pred_texts):
        correct, text_length = 0, 0
        for gt_ch, pred_ch in zip_longest(gt_text, pred_text):
            text_length += 1
            if gt_ch == pred_ch:
                correct += 1
        accuracy.append(correct / text_length)
    return np.mean(accuracy)


if __name__ == '__main__':
    crnn.to(device)
    crnn.train()  # Ensure the model is in training mode

    # Main training loop
    for epoch in range(wandb.config.epochs):
        for iter_n, (input_images, targets, target_lengths, gt_texts) in enumerate(barcode_loader):
            input_images = input_images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            output = crnn(input_images)  # Forward pass
            pred_texts = crnn.decode_output(output, wandb.config.vocab)  # Decode outputs
            train_acc = calculate_accuracy(gt_texts, pred_texts)  # Calculate training accuracy

            input_lengths = [output.size(0) for _ in input_images]  # Prepare input lengths for CTC Loss
            input_lengths = torch.LongTensor(input_lengths)
            target_lengths = torch.flatten(target_lengths)

            loss = criterion(output, targets, input_lengths, target_lengths)  # Compute loss
            if loss.isinf() or loss.isnan():
                print(f"loss is inf or nan, skipping the batch")
                continue

            loss.backward()  # Backpropagation
            torch.nn.utils.clip_grad_norm_(crnn.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()  # Optimizer step
            optimizer.zero_grad()  # Zero out gradients
            scheduler.step()  # Update learning rate

            crnn.eval()  # Set model to evaluation mode for testing
            with torch.no_grad():
                test_losses = []  # Store test losses
                for input_images, targets, target_lengths, gt_texts in test_barcode_loader:
                    input_images = input_images.to(device)
                    targets = targets.to(device)
                    target_lengths = target_lengths.to(device)

                    output = crnn(input_images)
                    pred_texts = crnn.decode_output(output, wandb.config.vocab)
                    test_acc = calculate_accuracy(gt_texts, pred_texts)

                    input_lengths = [output.size(0) for _ in input_images]
                    input_lengths = torch.LongTensor(input_lengths)
                    target_lengths = torch.flatten(target_lengths)

                    loss = criterion(output, targets, input_lengths, target_lengths)
                    if not (loss.isinf() or loss.isnan()):
                        test_losses.append(loss)
                test_loss = sum(test_losses) / len(test_losses)  # Calculate average test loss
            crnn.train()  # Revert to training mode

            # Log metrics to wandb
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

            # Save model checkpoint
            cp = crnn.state_dict()
            torch.save(cp, "../../crnn_best.pt")
