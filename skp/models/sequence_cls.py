import numpy as np
import torch
import torch.nn as nn

from transformers.models.distilbert.modeling_distilbert import Transformer as T


class Config:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Net(nn.Module):
    """
    If predict_sequence is True, then the model will predict an output
    for each element in the sequence. If False, then the model will
    predict a single output for the sequence. 
    
    e.g., classifying each image in a CT scan vs the entire CT scan
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        config = Config(**{
                'dim': cfg.embedding_dim,
                'hidden_dim': cfg.hidden_dim,
                'n_layers': cfg.n_layers,
                'n_heads': cfg.n_heads,
                'dropout': cfg.dropout,
                'attention_dropout': cfg.attention_dropout,
                'output_attentions': cfg.output_attentions,
                'activation': cfg.activation,
                'output_hidden_states': cfg.output_hidden_states,
                'chunk_size_feed_forward': cfg.chunk_size_feed_forward
            })

        self.transformer = T(config)
        self.classifier = nn.Linear(cfg.embedding_dim, cfg.num_classes)

    def extract_features(self, x, mask):
        x = self.transformer(x, attn_mask=mask, head_mask=[None]*x.size(1))
        x = x[0]
        return x

    def forward(self, batch, return_loss=False, return_features=False):
        x = batch["x"]
        y = batch["y"] if "y" in batch else None
        mask = batch["mask"]

        if return_loss:
            assert isinstance(y, torch.Tensor)

        features = self.extract_features(x, mask)
        # Take last vector in sequence
        features = features[:, 0]

        logits = self.classifier(features)
        
        out = {"logits": logits}
        if return_features:
            out["features"] = features 
        if return_loss: 
            loss = self.criterion(logits, y)
            out["loss"] = loss

        return out

    def set_criterion(self, loss):
        self.criterion = loss
