"""
Dual-Path Clinical Transformer (DPCT) — PyTorch model definition.

This module mirrors the architecture defined in `sepsis_transformer_paper.ipynb`.
The webapp loads a saved state_dict (.pth) produced by the Kaggle training run
and uses this class definition to reconstruct the full model for inference.

Architecture:
    VITAL PATH   : Linear projection + SinusoidalPE + TransformerEncoder
    LAB PATH     : TimeDecayLabEmbedding + SinusoidalPE + TransformerEncoder
    FUSION       : BidirectionalCrossAttention
    GATE         : ClinicalThresholdGate
    OUTPUT       : ClinicalAttentionPooling -> Linear -> Sigmoid
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Default hyper-parameters (must match training config) ─────────────────────
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LAB_COLS = [
    "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_direct", "Bilirubin_total", "TroponinI",
    "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
]

N_VITALS = len(VITAL_COLS)   # 7
N_LABS   = len(LAB_COLS)     # 27

D_VITAL      = 64
D_LAB        = 64
N_HEADS      = 4
N_LAYERS     = 2
DIM_FF       = 128
DROPOUT      = 0.1
DECAY_LAMBDA = 0.05
MAX_SEQ_LEN  = 72


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — Time-Decay Lab Embedding
# ══════════════════════════════════════════════════════════════════════════════
class TimeDecayLabEmbedding(nn.Module):
    """
    Encodes sparse lab measurements with three information sources:
      1. The lab value itself (zeroed-out when not measured)
      2. An exponential decay weight exp(-λ·Δt) where Δt = hours since last measurement
      3. A binary was-measured indicator embedding

    This explicitly communicates lab staleness to the Transformer rather than
    silently treating a 3-day-old Lactate reading the same as a fresh one.
    """

    def __init__(self, n_labs: int = N_LABS, d_out: int = D_LAB, decay_lambda: float = DECAY_LAMBDA):
        super().__init__()
        self.decay_lambda = decay_lambda
        self.value_proj   = nn.Linear(n_labs, d_out // 2)
        self.meta_proj    = nn.Linear(n_labs * 2, d_out // 2)
        self.norm         = nn.LayerNorm(d_out)
        self.act          = nn.GELU()

    def forward(self, lab_values: torch.Tensor, lab_delta: torch.Tensor, lab_measured: torch.Tensor) -> torch.Tensor:
        decay_weight = torch.exp(-self.decay_lambda * lab_delta)          # (B, T, n_labs)
        val_embed    = self.value_proj(lab_values * decay_weight)         # (B, T, d_out//2)
        meta         = torch.cat([decay_weight, lab_measured], dim=-1)   # (B, T, n_labs*2)
        meta_embed   = self.meta_proj(meta)                               # (B, T, d_out//2)
        out          = self.act(self.norm(torch.cat([val_embed, meta_embed], dim=-1)))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — Sinusoidal Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = DROPOUT):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — Bidirectional Cross-Attention Fusion
# ══════════════════════════════════════════════════════════════════════════════
class BidirectionalCrossAttention(nn.Module):
    """
    Two parallel cross-attention heads:
      V→L : vitals-as-query,  labs-as-key/value
      L→V : labs-as-query,    vitals-as-key/value
    Outputs are fused into (B, T, d_v + d_l).
    """

    def __init__(self, d_vital: int = D_VITAL, d_lab: int = D_LAB, n_heads: int = N_HEADS, dropout: float = DROPOUT):
        super().__init__()
        assert d_vital == d_lab, "d_vital must equal d_lab for cross-attention"
        d = d_vital
        self.v_to_l = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.l_to_v = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm_v = nn.LayerNorm(d)
        self.norm_l = nn.LayerNorm(d)
        self.fuse   = nn.Linear(d * 2, d * 2)
        self.norm_f = nn.LayerNorm(d * 2)
        self.act    = nn.GELU()

    def forward(self, vital_enc: torch.Tensor, lab_enc: torch.Tensor, pad_mask=None) -> torch.Tensor:
        v_ctx, _ = self.v_to_l(query=vital_enc, key=lab_enc,   value=lab_enc,   key_padding_mask=pad_mask)
        v_out    = self.norm_v(vital_enc + v_ctx)
        l_ctx, _ = self.l_to_v(query=lab_enc,   key=vital_enc, value=vital_enc, key_padding_mask=pad_mask)
        l_out    = self.norm_l(lab_enc + l_ctx)
        fused    = torch.cat([v_out, l_out], dim=-1)
        fused    = self.act(self.norm_f(self.fuse(fused)))
        return fused


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — Clinical Threshold Gate
# ══════════════════════════════════════════════════════════════════════════════
class ClinicalThresholdGate(nn.Module):
    """
    Computes a per-hour clinical alert score from soft versions of:
      MAP < 65 mmHg, HR > 100 bpm, Resp > 22 bpm, O2Sat < 94%, Lactate ↑

    Rule weights are *learned*, not hard-coded, so the model can down-weight
    any rule that is not useful for a given patient population.
    """

    def __init__(self):
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(5))
        self.hr_idx   = VITAL_COLS.index("HR")
        self.o2_idx   = VITAL_COLS.index("O2Sat")
        self.map_idx  = VITAL_COLS.index("MAP")
        self.resp_idx = VITAL_COLS.index("Resp")
        self.lac_idx  = LAB_COLS.index("Lactate")

    def forward(self, vitals_raw: torch.Tensor, labs_measured: torch.Tensor, lab_measured_mask: torch.Tensor) -> torch.Tensor:
        w          = F.softplus(self.log_weights)
        map_alert  = torch.sigmoid(-(vitals_raw[..., self.map_idx]  - 65) * 0.2)
        hr_alert   = torch.sigmoid( (vitals_raw[..., self.hr_idx]   - 100) * 0.1)
        resp_alert = torch.sigmoid( (vitals_raw[..., self.resp_idx] - 22) * 0.3)
        o2_alert   = torch.sigmoid(-(vitals_raw[..., self.o2_idx]   - 94) * 0.3)
        lac_vals   = labs_measured[..., self.lac_idx]
        lac_meas   = lab_measured_mask[..., self.lac_idx]
        lac_alert  = torch.sigmoid(lac_vals * 0.5) * lac_meas
        score = (w[0] * map_alert + w[1] * hr_alert + w[2] * resp_alert + w[3] * o2_alert + w[4] * lac_alert) / w.sum()
        return score


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — Clinical Attention Pooling
# ══════════════════════════════════════════════════════════════════════════════
class ClinicalAttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, gate_score: torch.Tensor, pad_mask=None):
        learned_scores = self.query(x).squeeze(-1)
        combined       = learned_scores + gate_score
        if pad_mask is not None:
            combined = combined.masked_fill(pad_mask, float("-inf"))
        attn    = F.softmax(combined, dim=-1)
        attn    = torch.nan_to_num(attn, nan=0.0)
        context = (attn.unsqueeze(-1) * x).sum(dim=1)
        return context, attn


# ══════════════════════════════════════════════════════════════════════════════
#  FULL MODEL — Dual-Path Clinical Transformer (DPCT)
# ══════════════════════════════════════════════════════════════════════════════
class DualPathClinicalTransformer(nn.Module):
    """
    Full DPCT model combining all five modules.

    Input  : vitals (B,T,7), vitals_raw (B,T,7), labs (B,T,27),
             lab_delta (B,T,27), lab_measured (B,T,27), pad_mask (B,T)
    Output : logit (B,), attn_weights (B,T)
    """

    def __init__(
        self,
        n_vitals: int      = N_VITALS,
        n_labs:   int      = N_LABS,
        d_vital:  int      = D_VITAL,
        d_lab:    int      = D_LAB,
        n_heads:  int      = N_HEADS,
        n_layers: int      = N_LAYERS,
        dim_ff:   int      = DIM_FF,
        dropout:  float    = DROPOUT,
        decay_lambda: float = DECAY_LAMBDA,
        max_len:  int      = MAX_SEQ_LEN,
    ):
        super().__init__()
        assert d_vital == d_lab

        # Vital path
        self.vital_proj = nn.Linear(n_vitals, d_vital)
        self.vital_pe   = SinusoidalPE(d_vital, max_len, dropout)
        vital_layer     = nn.TransformerEncoderLayer(
            d_model=d_vital, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.vital_enc  = nn.TransformerEncoder(vital_layer, num_layers=n_layers)

        # Lab path
        self.lab_embed  = TimeDecayLabEmbedding(n_labs, d_lab, decay_lambda)
        self.lab_pe     = SinusoidalPE(d_lab, max_len, dropout)
        lab_layer       = nn.TransformerEncoderLayer(
            d_model=d_lab, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.lab_enc    = nn.TransformerEncoder(lab_layer, num_layers=n_layers)

        # Fusion + gate + output
        self.fusion     = BidirectionalCrossAttention(d_vital, d_lab, n_heads, dropout)
        self.gate       = ClinicalThresholdGate()
        d_fused         = d_vital + d_lab
        self.pool       = ClinicalAttentionPooling(d_fused)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_fused, d_fused // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fused // 2, 1),
        )

    def forward(
        self,
        vitals:       torch.Tensor,
        vitals_raw:   torch.Tensor,
        labs:         torch.Tensor,
        lab_delta:    torch.Tensor,
        lab_measured: torch.Tensor,
        pad_mask=None,
    ):
        v      = self.vital_pe(self.vital_proj(vitals))
        v      = self.vital_enc(v, src_key_padding_mask=pad_mask)
        l      = self.lab_pe(self.lab_embed(labs, lab_delta, lab_measured))
        l      = self.lab_enc(l, src_key_padding_mask=pad_mask)
        fused  = self.fusion(v, l, pad_mask)
        gate   = self.gate(vitals_raw, labs, lab_measured)
        ctx, attn = self.pool(fused, gate, pad_mask)
        logit  = self.classifier(self.dropout(ctx)).squeeze(-1)
        return logit, attn
