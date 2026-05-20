"""T5 wrapper that accepts a continuous MERT embedding via a linear projection
into the encoder hidden dim, in place of token embeddings.
"""
from __future__ import annotations

# TODO: subclass T5ForConditionalGeneration, override encoder forward to use
# inputs_embeds from a Linear(768, d_model) projection.


class T5WithMERTProjection:
    pass
