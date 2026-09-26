"""Model definitions: ADGKT (attention + GRU hybrid) and the DKT baseline.

Both predict the probability that the student answers the *target* question
correctly, given the interaction history.

ADGKT v2 fixes relative to the original notebook implementation:

1. **Causal attention mask** — the original ``nn.MultiheadAttention`` call did
   not pass ``attn_mask``, so position *i* could attend to *future*
   interactions. Because responses are embedded into every position, the
   model could read the label it was asked to predict (r_{t+1}) one step
   ahead, which inflated validation AUC to ~0.96. The attention now uses a
   lower-triangular mask, matching a Transformer-decoder-style autoregressive
   view: the prediction for step t only sees interactions up to t.

2. **Padding key mask** — padded positions (id 0) are masked out of attention
   keys, so they cannot contribute to any prediction.

3. **Length-based loss mask** (in :mod:`edqr.data`) replaces the original
   ``mask = (target_q > 0)`` which silently dropped the question mapped to
   id 0. Ids now start at 1, so 0 is unambiguously padding.
"""

import torch
import torch.nn as nn


class ADGKT(nn.Module):
    """Attention-augmented Deep Knowledge Tracing.

    Question + module embeddings are fused per interaction; the response
    embedding is added residually (the strongest signal gets a direct path).
    One multi-head self-attention block (causal) captures long-range
    dependencies across interactions; a GRU condenses the attended sequence
    into a temporal knowledge state; the state is concatenated with the target
    question's embedding to produce a correctness probability.
    """

    def __init__(self, num_q: int, num_c: int, embed_size: int = 128,
                 hidden_size: int = 128, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        # +1 on both: id 0 is reserved for padding
        self.q_embed = nn.Embedding(num_q + 1, embed_size, padding_idx=0)
        self.c_embed = nn.Embedding(num_c + 1, embed_size, padding_idx=0)
        self.r_embed = nn.Embedding(2, embed_size)

        self.interaction_fusion = nn.Linear(embed_size * 2, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads,
                                               dropout=dropout, batch_first=True)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.predict_fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, input_q, input_c, input_r, target_q, target_c):
        q_emb = self.q_embed(input_q)
        c_emb = self.c_embed(input_c)
        r_emb = self.r_embed(input_r)

        interaction_emb = self.interaction_fusion(torch.cat([q_emb, c_emb], dim=-1)) + r_emb

        seq_len = input_q.size(1)
        # lower-triangular causal mask: the prediction for step t may only
        # attend to interactions up to and including t (fixes the original
        # label leak where attention could read r_{t+1} from future rows).
        # Note: a key-padding mask is deliberately NOT combined here — for a
        # left-padded row, query 0 would lose its only (diagonal) key and
        # softmax would produce NaN. Padding ids embed to the zero vector
        # (padding_idx=0), and padded positions are excluded from the loss.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_q.device), diagonal=1
        )

        attn_input = self.layer_norm1(interaction_emb)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, attn_mask=causal_mask)
        attn_output = self.dropout(attn_output) + interaction_emb

        gru_input = self.layer_norm2(attn_output)
        gru_output, _ = self.gru(gru_input)

        target_q_emb = self.q_embed(target_q)
        target_c_emb = self.c_embed(target_c)

        predict_input = torch.cat([gru_output, target_q_emb + target_c_emb], dim=-1)
        predict_hidden = torch.relu(self.predict_fusion(predict_input))
        output = self.out(predict_hidden)
        return torch.sigmoid(output).squeeze(-1)


class DKT(nn.Module):
    """Canonical DKT baseline (Piech et al., 2015): question-response
    interaction embedding -> LSTM -> per-question correctness logits."""

    def __init__(self, num_q: int, embed_size: int = 128, hidden_size: int = 128):
        super().__init__()
        # interactions: (question_id * 2 + response); ids start at 1 -> +2 slots for safety
        self.qr_embed = nn.Embedding((num_q + 1) * 2, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, num_q + 1)

    def forward(self, x):
        embed = self.qr_embed(x)
        lstm_out, _ = self.lstm(embed)
        logits = self.out(lstm_out)
        return logits
