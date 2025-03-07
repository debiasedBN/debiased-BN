import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import CHANNEL_NUM, RAW_SEQUENCE_LENGTH, EMBED_DIM, WINDOW, STRIDE, HEAD_SIZE, NUM_HEADS, FF_DIM, SEQUENCE_LENGTH, DROPOUT_RATE, NUM_CLASSES, DEBIASED_BN, SUB_BATCH_NUM, SUB_BATCH_SIZE, NUM_LAYERS, LEARNING_RATE, WEIGHT_DECAY
# Import DebiasedBatchNorm from its new module
from debiased_bn import DebiasedBatchNorm

class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, output_dim):
        super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, output_dim))

    def forward(self, inputs):
        return inputs + self.pos_embed

class AddClsToken(nn.Module):
    def __init__(self, embed_dim):
        super(AddClsToken, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat((inputs, cls_tokens), dim=1)

class EEGEmbedNetModel(nn.Module):
    def __init__(self, channel_num, raw_sequence_length, embed_dim, window, stride, head_size, num_heads, ff_dim, sequence_length, dropout_rate, num_classes):
        super(EEGEmbedNetModel, self).__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.window = window
        self.stride = stride

        self.timeseries_feature = nn.Conv2d(1, embed_dim, (1, window), (1, stride))
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, (channel_num, 1), stride=(1, 1))
        self.embedding = nn.Linear(embed_dim, embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.add_cls = AddClsToken(embed_dim)
        self.positional_encoding = PositionalEncoding(sequence_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                                                   dropout=dropout_rate, activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        if DEBIASED_BN:
            self.batch_norm1 = DebiasedBatchNorm(embed_dim, SUB_BATCH_NUM, SUB_BATCH_SIZE)
            self.batch_norm2 = DebiasedBatchNorm(embed_dim, SUB_BATCH_NUM, SUB_BATCH_SIZE)
        else:
            self.batch_norm1 = nn.BatchNorm1d(embed_dim)
            self.batch_norm2 = nn.BatchNorm1d(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.timeseries_feature(x)
        x = self.conv2(x)
        x = x.squeeze(2)
        x = self.batch_norm1(x)
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.layer_norm1(x)
        x = self.add_cls(x)
        x = self.positional_encoding(x)
        x = self.layer_norm2(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.batch_norm2(x)
        x = self.classifier(x)
        return x

def define_model():
    model = EEGEmbedNetModel(CHANNEL_NUM, RAW_SEQUENCE_LENGTH, EMBED_DIM, WINDOW, STRIDE, HEAD_SIZE, NUM_HEADS,
                             FF_DIM, SEQUENCE_LENGTH, DROPOUT_RATE, NUM_CLASSES)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return model, optimizer
