"""ST1 classifier: XLM-R (text) + ViT (image), concatenation fusion.

Both encoders are fine-tuned end to end; the [CLS] vectors are concatenated and
fed to one linear layer. Three modes share the class:
  - text_only:   XLM-R only
  - image_only:  ViT only
  - multimodal:  XLM-R + ViT concatenated
ST1 uses num_labels=8 (the eight emotions).
"""
import torch
import torch.nn as nn
from transformers import AutoModel

TEXT_MODEL_NAME = "xlm-roberta-base"
IMAGE_MODEL_NAME = "google/vit-base-patch16-224"


class MultimodalSentimentModel(nn.Module):
    def __init__(self, mode="multimodal", num_labels=2, dropout=0.1):
        super().__init__()
        assert mode in ("text_only", "image_only", "multimodal")
        self.mode = mode

        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME) if mode in ("text_only", "multimodal") else None
        self.image_encoder = AutoModel.from_pretrained(IMAGE_MODEL_NAME) if mode in ("image_only", "multimodal") else None

        text_dim = self.text_encoder.config.hidden_size if self.text_encoder is not None else 0
        image_dim = self.image_encoder.config.hidden_size if self.image_encoder is not None else 0
        fusion_dim = text_dim + image_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None):
        feats = []
        if self.text_encoder is not None:
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            feats.append(text_out.last_hidden_state[:, 0])  # [CLS]
        if self.image_encoder is not None:
            image_out = self.image_encoder(pixel_values=pixel_values)
            feats.append(image_out.last_hidden_state[:, 0])  # [CLS]

        fused = torch.cat(feats, dim=-1)
        logits = self.classifier(self.dropout(fused))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
