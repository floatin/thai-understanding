"""
XLSR-Thai model loader.

The official XLSR-Thai checkpoint is in Fairseq format. Fairseq 0.12.2 has Python 3.12
dataclass compatibility issues, so we load weights into torchaudio.models.Wav2Vec2Model
which has the same architecture but works on Python 3.12.
"""
import torch
import warnings
import os
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore")

from torchaudio.models import wav2vec2_model


def _map_fairseq_key(fs_k: str) -> str | None:
    """Map a Fairseq wav2vec2 key to torchaudio Wav2Vec2Model key.

    Returns None for SSL-specific keys (quantizer, project_q, mask_emb, final_proj)
    that don't exist in the inference model.
    """
    # feature_extractor.conv_layers.X.0.weight/bias -> conv
    # OR feature_extractor.conv_layers.X.2.1.weight/bias -> layer_norm
    if fs_k.startswith("feature_extractor.conv_layers."):
        parts = fs_k.split(".")
        layer_idx = parts[2]
        if parts[3] == "0" and len(parts) == 5:
            return f"feature_extractor.conv_layers.{layer_idx}.conv.{parts[4]}"
        if parts[3] == "2" and parts[4] == "1" and len(parts) == 6:
            return f"feature_extractor.conv_layers.{layer_idx}.layer_norm.{parts[5]}"
        return None

    # post_extract_proj + layer_norm (after feature extractor)
    mapping_simple = {
        "post_extract_proj.weight": "encoder.feature_projection.projection.weight",
        "post_extract_proj.bias": "encoder.feature_projection.projection.bias",
        "layer_norm.weight": "encoder.feature_projection.layer_norm.weight",
        "layer_norm.bias": "encoder.feature_projection.layer_norm.bias",
    }
    if fs_k in mapping_simple:
        return mapping_simple[fs_k]

    # Encoder transformer layers
    if fs_k.startswith("encoder.layers."):
        rest = fs_k[len("encoder.layers."):]
        layer_idx = rest.split(".")[0]
        suffix = ".".join(rest.split(".")[1:])
        if suffix.startswith("self_attn."):
            sub = suffix[len("self_attn."):]
            return f"encoder.transformer.layers.{layer_idx}.attention.{sub}"
        if suffix == "self_attn_layer_norm.weight":
            return f"encoder.transformer.layers.{layer_idx}.layer_norm.weight"
        if suffix == "self_attn_layer_norm.bias":
            return f"encoder.transformer.layers.{layer_idx}.layer_norm.bias"
        if suffix == "fc1.weight":
            return f"encoder.transformer.layers.{layer_idx}.feed_forward.intermediate_dense.weight"
        if suffix == "fc1.bias":
            return f"encoder.transformer.layers.{layer_idx}.feed_forward.intermediate_dense.bias"
        if suffix == "fc2.weight":
            return f"encoder.transformer.layers.{layer_idx}.feed_forward.output_dense.weight"
        if suffix == "fc2.bias":
            return f"encoder.transformer.layers.{layer_idx}.feed_forward.output_dense.bias"
        if suffix == "final_layer_norm.weight":
            return f"encoder.transformer.layers.{layer_idx}.final_layer_norm.weight"
        if suffix == "final_layer_norm.bias":
            return f"encoder.transformer.layers.{layer_idx}.final_layer_norm.bias"
        return None

    if fs_k.startswith("encoder.layer_norm."):
        return "encoder.transformer.layer_norm." + fs_k.split(".")[-1]

    # pos_conv: fairseq uses weight_g, weight_v (weight_norm parametrization).
    # torchaudio uses parametrizations.weight.original0 (g) / original1 (v).
    if fs_k == "encoder.pos_conv.0.weight_g":
        return "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original0"
    if fs_k == "encoder.pos_conv.0.weight_v":
        return "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original1"
    if fs_k == "encoder.pos_conv.0.bias":
        return "encoder.transformer.pos_conv_embed.conv.bias"
    if fs_k.startswith("encoder.pos_conv.0."):
        return None

    # SSL-specific (skip)
    if fs_k in {
        "final_proj.weight", "final_proj.bias",
        "quantizer.vars", "quantizer.weight_proj.weight", "quantizer.weight_proj.bias",
        "project_q.weight", "project_q.bias", "mask_emb",
    }:
        return None

    return None


def load_xlsr_thai(checkpoint_path: str, device: str = "cuda") -> Tuple[torch.nn.Module, dict]:
    """Load XLSR-Thai from a Fairseq checkpoint into torchaudio Wav2Vec2Model.

    Returns (model, cfg) where cfg is the original fairseq config dict.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fs_state = ckpt["model"]
    cfg = ckpt["cfg"]["model"]

    # cfg["conv_feature_layers"] may be stored as a Python expression string
    if isinstance(cfg["conv_feature_layers"], str):
        conv_layers = eval(cfg["conv_feature_layers"])
    else:
        conv_layers = cfg["conv_feature_layers"]

    # Build model with XLSR-53 large config
    model = wav2vec2_model(
        extractor_mode=cfg["extractor_mode"],
        extractor_conv_layer_config=conv_layers,
        extractor_conv_bias=cfg["conv_bias"],
        encoder_embed_dim=cfg["encoder_embed_dim"],
        encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=cfg["conv_pos"],
        encoder_pos_conv_groups=cfg["conv_pos_groups"],
        encoder_num_layers=cfg["encoder_layers"],
        encoder_num_heads=cfg["encoder_attention_heads"],
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=cfg["encoder_ffn_embed_dim"],
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.0,
        encoder_layer_norm_first=cfg["layer_norm_first"],
        encoder_layer_drop=0.0,
        aux_num_out=None,
    )

    ta_state = model.state_dict()

    # Map weights
    new_state = {}
    unmapped = []
    for fs_k, fs_v in fs_state.items():
        ta_k = _map_fairseq_key(fs_k)
        if ta_k is None:
            unmapped.append(fs_k)
            continue
        if ta_k in ta_state and tuple(ta_state[ta_k].shape) == tuple(fs_v.shape):
            new_state[ta_k] = fs_v
        else:
            unmapped.append(fs_k)

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        print(f"[load_xlsr_thai] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  missing keys: {missing[:3]}")
        if unexpected:
            print(f"  unexpected keys: {unexpected[:3]}")

    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)

    return model, cfg


def extract_features(model, audio: torch.Tensor, lengths: torch.Tensor | None = None):
    """Extract contextualized features from raw audio.

    Args:
        model: XLSR-Thai wav2vec2 model
        audio: (B, T) float tensor in [-1, 1] at 16kHz
        lengths: (B,) lengths in samples; if None, all assumed same length
    Returns:
        features: (B, T', 1024) tensor
        output_lengths: (B,) lengths in frames
    """
    model.eval()
    if lengths is None:
        lengths = torch.tensor([audio.shape[1]] * audio.shape[0], dtype=torch.long)
    # Ensure same device as model
    audio = audio.to(next(model.parameters()).device)
    lengths = lengths.to(next(model.parameters()).device)
    with torch.no_grad():
        # torchaudio Wav2Vec2Model.forward(waveforms, lengths) where waveforms is (B, T)
        # It returns (output, output_lengths) - or just output depending on version
        try:
            out = model(audio, lengths)
        except Exception:
            # Some versions don't accept lengths arg
            out = model(audio)
    # torchaudio Wav2Vec2Model returns (features, lengths) when called as model(audio)
    # If called with lengths, returns (features, lengths)
    if isinstance(out, tuple):
        features, output_lengths = out
    else:
        features = out
        # Compute output lengths: T' = T / 320 (stride)
        output_lengths = torch.tensor([audio.shape[1] // 320] * audio.shape[0])
    return features, output_lengths


if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt"
    print(f"Loading {ckpt_path}...")
    model, cfg = load_xlsr_thai(ckpt_path)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"Config: {cfg.get('_name')}, {cfg.get('encoder_layers')} layers, dim={cfg.get('encoder_embed_dim')}")

    # Quick test
    audio = torch.randn(1, 16000 * 5)
    feats, lens = extract_features(model, audio)
    print(f"Input audio: {audio.shape}, Output features: {feats.shape}, output_lengths: {lens}")
    # Feature dim should be 1024 (encoder_embed_dim)
    print(f"Feature dim: {feats.shape[-1]} (expected 1024)")
