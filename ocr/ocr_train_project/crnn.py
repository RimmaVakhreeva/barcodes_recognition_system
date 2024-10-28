from typing import List

import timm
import torch
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class CRNN(torch.nn.Module):
    """Convolutional Recurrent Neural Network for Optical Character Recognition.

    This network architecture combines a convolutional neural network (CNN) backbone
    with a gated recurrent unit (GRU) to process images and extract text.
    """

    def __init__(
            self,
            cnn_backbone_name: str,
            cnn_backbone_pretrained: bool,
            cnn_output_size: int,
            rnn_features_num: int,
            rnn_dropout: float,
            rnn_bidirectional: bool,
            rnn_num_layers: int,
            num_classes: int,
    ) -> None:
        super().__init__()

        # Initialize the CNN backbone with optional pretrained weights
        self.backbone = timm.create_model(
            cnn_backbone_name, pretrained=cnn_backbone_pretrained
        )
        # Remove global pooling and fully connected layer from backbone
        self.backbone.global_pool = torch.nn.Identity()
        self.backbone.fc = torch.nn.Identity()

        # Gate layer to adjust feature size from CNN to RNN input size
        self.gate = torch.nn.Linear(cnn_output_size, rnn_features_num)

        # Setup the GRU with specified features and layers
        self.rnn = torch.nn.GRU(
            rnn_features_num,
            rnn_features_num,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
            num_layers=rnn_num_layers,
        )

        # Classifier to map RNN outputs to class scores
        classifier_in_features = rnn_features_num * (2 if rnn_bidirectional else 1)
        self.fc = torch.nn.Linear(classifier_in_features, num_classes)

        # LogSoftmax for generating probabilities
        self.softmax = torch.nn.LogSoftmax(dim=2)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes weights with specific schemes for better convergence."""
        torch.nn.init.kaiming_normal_(self.gate.weight, mode='fan_out', nonlinearity='relu')
        if self.gate.bias is not None:
            self.gate.bias.data.fill_(0.01)

        # Initialize GRU weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Initialize classifier weights
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.01)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through CRNN. Processes input tensor through CNN, RNN and classifier."""
        cnn_features = self.backbone(tensor)
        batch_size, channels, height, width = cnn_features.shape
        cnn_features = cnn_features.view(
            batch_size, height * channels, width
        ).permute(2, 0, 1)
        cnn_features = torch.nn.functional.relu(self.gate(cnn_features))
        rnn_output, _ = self.rnn(cnn_features)
        logits = self.fc(rnn_output)
        output = self.softmax(logits)
        return output

    def decode_output(self, pred: torch.Tensor, vocab: str) -> List[str]:
        """Decode the output tensor into human-readable text using the provided vocabulary."""
        texts = []
        index2char = {idx + 1: char for idx, char in enumerate(vocab)}
        index2char[0] = ""
        for idx in range(pred.shape[1]):
            classes_b = pred[:, idx, :].argmax(dim=1).cpu().numpy().tolist()
            chars = list(map(lambda x: index2char[x], classes_b))[:13]
            texts.append("".join(chars))
        return texts


class TransformerOcr(torch.nn.Module):
    """Transformer-based OCR model for optical character recognition.

    This model uses a CNN backbone for feature extraction, followed by a transformer decoder
    to process the features and predict text classes.
    """

    def __init__(
            self,
            cnn_backbone_name: str,
            cnn_backbone_pretrained: bool,
            cnn_output_size: int,
            transformer_features_num: int,
            transformer_dropout: float,
            transformer_nhead: int,
            transformer_num_layers: int,
            num_classes: int,
    ) -> None:
        super().__init__()

        # Number of classes (characters)
        self.num_classes = num_classes

        # Initialize the CNN backbone with optional pretrained weights
        self.backbone = timm.create_model(
            cnn_backbone_name, pretrained=cnn_backbone_pretrained
        )
        self.backbone.global_pool = torch.nn.Identity()
        self.backbone.fc = torch.nn.Identity()

        # Gate layer to adjust feature size from CNN to Transformer input size
        self.gate = torch.nn.Linear(cnn_output_size, transformer_features_num)

        # Transformer decoder setup
        decoder_layer = TransformerDecoderLayer(
            d_model=transformer_features_num,
            nhead=transformer_nhead,
            dropout=transformer_dropout,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=transformer_num_layers
        )

        # Classifier to map transformer outputs to class scores
        self.fc = torch.nn.Linear(transformer_features_num, num_classes)

        # LogSoftmax for generating probabilities
        self.softmax = torch.nn.LogSoftmax(dim=2)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes weights with specific schemes for better convergence."""
        torch.nn.init.kaiming_normal_(self.gate.weight, mode='fan_out', nonlinearity='relu')
        if self.gate.bias is not None:
            self.gate.bias.data.fill_(0.01)

        # Initialize classifier weights
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.01)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer OCR model. Processes input tensor through CNN, Transformer, and classifier."""
        cnn_features = self.backbone(tensor)
        batch_size, channels, height, width = cnn_features.shape
        cnn_features = cnn_features.view(
            batch_size, height * channels, width
        ).permute(2, 0, 1)
        cnn_features = torch.nn.functional.relu(self.gate(cnn_features))

        # Placeholder for memory in transformer
        memory = torch.zeros_like(cnn_features)

        transformer_output = self.transformer_decoder(cnn_features, memory)

        logits = self.fc(transformer_output)
        output = self.softmax(logits)
        return output

    def decode_output(self, pred: torch.Tensor, vocab: str) -> List[str]:
        """Decode the output tensor into human-readable text using the provided vocabulary, including handling EOS token."""
        texts = []
        index2char = {idx + 1: char for idx, char in enumerate(vocab)}
        index2char[0] = ""
        index2char[len(vocab) + 1] = "<eos>"
        for idx in range(pred.shape[1]):
            classes_b = pred[:, idx, :].argmax(dim=1).cpu().numpy().tolist()
            chars = list(map(lambda x: index2char[x], classes_b))
            text = "".join(chars).split("<eos>")[0]
            texts.append(text)
        return texts
