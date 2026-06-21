"""
HIA: Hierarchical Interactive Attention Framework
for Residual Hierarchical Pronunciation Assessment (AAAI 2026)

Architecture:
  1. GOP features (84-dim) + canonical phone one-hot embedding
  2. Input projection layers → embed_dim
  3. Transformer Encoder (3 layers)
  4. Interactive Attention Module (multi-granularity learnable queries)
     - Phone-level, Word-level, Utterance-level query vectors
     - Self-attention among queries (granularity interaction)
     - Cross-attention with acoustic features
  5. Residual Hierarchical Predictions:
     - Phone level  : X + H_phn → Conv1D → score (accuracy)
     - Word level   : X + S_phn + H_word → Conv1D → scores (acc, stress, total)
     - Utt level    : X + S_word + H_utt → TransformerDecoder → Conv1D → scores
                      (accuracy, completeness, fluency, prosodic, total)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Interactive Attention Module
# ─────────────────────────────────────────────

class InteractiveAttentionModule(nn.Module):
    """
    Multi-granularity Interactive Attention Module.

    Learns separate query vectors for phoneme, word, and utterance levels.
    - Concatenates them and runs self-attention (granularity interaction).
    - Then runs cross-attention against the acoustic feature sequence H.
    - Splits the output back into per-level representations.
    """

    def __init__(self, embed_dim: int, num_heads: int, seq_len: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Learnable query matrices for each granularity
        self.query_phn  = nn.Parameter(torch.empty(1, seq_len, embed_dim))
        self.query_word = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.query_utt  = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query_phn,  std=0.02)
        nn.init.trunc_normal_(self.query_word, std=0.02)
        nn.init.trunc_normal_(self.query_utt,  std=0.02)

        # Self-attention among concatenated queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_sa = nn.LayerNorm(embed_dim)

        # Cross-attention: queries → acoustic features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ca = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, H: torch.Tensor,
                src_key_padding_mask: torch.Tensor | None = None):
        """
        Args:
            H:                    acoustic features  [B, L, d]
            src_key_padding_mask: True for pad tokens [B, L]
        Returns:
            H_phn  [B, L, d], H_word [B, 1, d], H_utt [B, 1, d]
        """
        B = H.size(0)

        # Expand queries to batch size
        Q_phn  = self.query_phn.expand(B, -1, -1)   # [B, L, d]
        Q_word = self.query_word.expand(B, -1, -1)  # [B, 1, d]
        Q_utt  = self.query_utt.expand(B, -1, -1)   # [B, 1, d]
        Q = torch.cat([Q_phn, Q_word, Q_utt], dim=1)  # [B, L+2, d]

        # Self-attention (granularity interaction)
        out, _ = self.self_attn(Q, Q, Q)
        Q = self.norm_sa(Q + out)

        # Cross-attention: queries attend to acoustic features H
        out, _ = self.cross_attn(Q, H, H, key_padding_mask=src_key_padding_mask)
        Q = self.norm_ca(Q + out)

        # Feed-forward
        Q = self.norm_ffn(Q + self.ffn(Q))

        # Split back into per-level representations
        L = self.seq_len
        H_phn  = Q[:, :L, :]       # [B, L, d]
        H_word = Q[:, L:L+1, :]    # [B, 1, d]
        H_utt  = Q[:, L+1:L+2, :]  # [B, 1, d]
        return H_phn, H_word, H_utt


# ─────────────────────────────────────────────
# Main HIA Model
# ─────────────────────────────────────────────

class HIA(nn.Module):
    """
    Hierarchical Interactive Attention model.

    Returns (in the same order as GOPT for drop-in compatibility):
        u_acc, u_comp, u_flu, u_pros, u_total  — utterance-level scores  [B]
        p_acc                                  — phone-level accuracy     [B, L]
        w_acc, w_stress, w_total               — word-level scores        [B, L]
    """

    def __init__(self, embed_dim: int = 48, num_heads: int = 1, depth: int = 3,
                 gop_dim: int = 84, seq_len: int = 50, dropout: float = 0.1,
                 num_phones: int = 42):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len   = seq_len

        # ── Input projections ──────────────────────────────────
        self.gop_proj = nn.Linear(gop_dim, embed_dim)
        self.phn_proj = nn.Linear(num_phones, embed_dim)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.input_dropout = nn.Dropout(dropout)

        # ── Transformer Encoder ────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # ── Interactive Attention Module ───────────────────────
        self.iam = InteractiveAttentionModule(
            embed_dim, num_heads, seq_len=seq_len, dropout=dropout
        )

        # ── Phone-level branch ─────────────────────────────────
        self.phn_conv  = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.phn_norm  = nn.LayerNorm(embed_dim)
        self.phn_head  = nn.Linear(embed_dim, 1)   # accuracy

        # ── Word-level branch ──────────────────────────────────
        self.phn_res_proj  = nn.Linear(1, embed_dim)   # project phone score residual
        self.word_conv     = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.word_norm     = nn.LayerNorm(embed_dim)
        self.word_head     = nn.Linear(embed_dim, 3)   # acc, stress, total

        # ── Utterance-level branch ─────────────────────────────
        self.word_res_proj = nn.Linear(3, embed_dim)   # project word score residual
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.utt_decoder = nn.TransformerDecoder(dec_layer, num_layers=depth)
        self.utt_conv    = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.utt_norm    = nn.LayerNorm(embed_dim)
        self.utt_head    = nn.Linear(embed_dim, 5)   # acc, comp, flu, pros, total

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, gop: torch.Tensor, phn: torch.Tensor):
        """
        Args:
            gop: GOP features           [B, L, 84]
            phn: canonical phone IDs    [B, L]  (-1 = padding, 0-40 = phones)
        Returns:
            (u_acc, u_comp, u_flu, u_pros, u_total,
             p_acc,
             w_acc, w_stress, w_total)
        """
        B, L = phn.shape

        # Padding mask: True where phone is padded (phn == -1)
        pad_mask = (phn < 0)  # [B, L]

        # ── Input encoding ──────────────────────────────────────
        # One-hot phone encoding: shift by +1 (-1 → 0, 0-40 → 1-41)
        phn_shifted = (phn + 1).clamp(0, 41).long()
        phn_onehot  = F.one_hot(phn_shifted, num_classes=42).float()  # [B, L, 42]
        phn_embed   = self.phn_proj(phn_onehot)  # [B, L, d]

        gop_embed = self.gop_proj(gop)  # [B, L, d]

        X = gop_embed + phn_embed + self.pos_embed[:, :L, :]
        X = self.input_dropout(X)

        # ── Transformer Encoder ─────────────────────────────────
        H = self.transformer_encoder(X, src_key_padding_mask=pad_mask)

        # ── Interactive Attention Module ────────────────────────
        H_phn, H_word, H_utt = self.iam(H, src_key_padding_mask=pad_mask)

        # ── Phone-level branch ──────────────────────────────────
        F_phn = X + H_phn
        F_phn = self.phn_conv(F_phn.transpose(1, 2)).transpose(1, 2)
        F_phn = self.phn_norm(F_phn)
        p_acc = self.phn_head(F_phn).squeeze(-1)  # [B, L]

        # ── Word-level branch (residual from phone) ─────────────
        phn_residual = self.phn_res_proj(p_acc.unsqueeze(-1))  # [B, L, d]
        F_word = X + phn_residual + H_word  # H_word [B,1,d] broadcasts
        F_word = self.word_conv(F_word.transpose(1, 2)).transpose(1, 2)
        F_word = self.word_norm(F_word)
        word_out = self.word_head(F_word)  # [B, L, 3]

        # ── Utterance-level branch (residual from word) ─────────
        word_residual = self.word_res_proj(word_out)  # [B, L, d]
        memory = X + word_residual + H_utt            # H_utt [B,1,d] broadcasts
        dec_out = self.utt_decoder(
            H_utt, memory, memory_key_padding_mask=pad_mask
        )  # [B, 1, d]
        dec_out = self.utt_conv(dec_out.transpose(1, 2)).transpose(1, 2)
        dec_out = self.utt_norm(dec_out)
        utt_out = self.utt_head(dec_out.squeeze(1))  # [B, 5]

        # Return in GOPT-compatible order
        return (
            utt_out[:, 0],    # utt accuracy
            utt_out[:, 1],    # utt completeness
            utt_out[:, 2],    # utt fluency
            utt_out[:, 3],    # utt prosodic
            utt_out[:, 4],    # utt total
            p_acc,            # [B, L] phone accuracy
            word_out[:, :, 0],  # [B, L] word accuracy
            word_out[:, :, 1],  # [B, L] word stress
            word_out[:, :, 2],  # [B, L] word total
        )
