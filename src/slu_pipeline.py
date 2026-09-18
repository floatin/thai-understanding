"""
End-to-end SLU pipeline matching the Thai-Understanding paper architecture.

Components (paper Section 2.2):
  1. XLSR-Thai encoder (1024-dim, 20ms stride)
  2. Modality adapter: LayerNorm + CNN subsampler + projection MLP
     Maps encoder dim (1024) -> LLM hidden size (3072 for Llama-3.2-3B)
  3. Frozen LLM: Typhoon2-LLaMa2-3B (Llama architecture, 3072 hidden)
  4. Task prompt + speech embeddings -> text generation

The released checkpoint is encoder-only. The adapter + LLM are not provided
in the public release, so we initialize the adapter with random weights and
the LLM with the public Typhoon-3B weights. This is a "smoke test" that
verifies the architecture works end-to-end — it will NOT reproduce the
paper's task accuracy without training the adapter on the alignment stage.

For inference we keep:
  - XLSR-Thai + adapter on GPU (~1.3 GB VRAM)
  - LLM on CPU (works with small batch, slow but feasible)
"""
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).parent))
from xlsr_thai import load_xlsr_thai


class StridedMeanSubsampler(nn.Module):
    """Simple stride-2 mean subsampler for temporal downsampling."""

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        if T % 2 == 1:
            x = x[:, :-1, :]
        return (x[:, 0::2, :] + x[:, 1::2, :]) / 2


class ModalityAdapter(nn.Module):
    """Adapter: LN + subsampler + MLP projection (1024 -> LLM dim)."""

    def __init__(self, encoder_dim: int = 1024, llm_dim: int = 3072, downsample: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(encoder_dim)
        self.subsampler = StridedMeanSubsampler()
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):  # x: (B, T, encoder_dim)
        x = self.norm(x)
        x = self.subsampler(x)
        x = self.proj(x)
        return x  # (B, T//2, llm_dim)


class ThaiSLUPipeline:
    """End-to-end SLU: audio -> text.

    - Encoder + adapter on GPU
    - LLM on CPU (configurable via llm_device)
    """

    def __init__(
        self,
        encoder_ckpt: str,
        llm_path: str,
        llm_dim: int = 3072,
        encoder_device: str = "cuda",
        llm_device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.encoder_device = encoder_device
        self.llm_device = llm_device
        self.dtype = dtype

        # XLSR-Thai encoder on GPU
        print(f"[SLU] Loading XLSR-Thai from {encoder_ckpt} -> {encoder_device}")
        self.encoder, self.encoder_cfg = load_xlsr_thai(encoder_ckpt, device=encoder_device)
        self.encoder_dim = self.encoder_cfg["encoder_embed_dim"]
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Adapter on GPU (small)
        print(f"[SLU] Initializing adapter ({self.encoder_dim} -> {llm_dim}) on {encoder_device}")
        self.adapter = ModalityAdapter(
            encoder_dim=self.encoder_dim,
            llm_dim=llm_dim,
            downsample=2,
        ).to(device=encoder_device, dtype=dtype)

        # Frozen LLM
        print(f"[SLU] Loading LLM from {llm_path} -> {llm_device}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, dtype=dtype,
        )
        self.llm = self.llm.to(llm_device)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False
        self.embed_tokens = self.llm.get_input_embeddings()

    @torch.no_grad()
    def encode_audio(self, audio: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is None:
            lengths = torch.tensor([audio.shape[1]] * audio.shape[0], dtype=torch.long)
        feats, _ = self.encoder(audio.to(self.encoder_device), lengths.to(self.encoder_device))
        return feats

    @torch.no_grad()
    def generate(
        self,
        audio: torch.Tensor,
        task_prompt: str,
        max_new_tokens: int = 32,
    ) -> str:
        feats = self.encode_audio(audio)
        speech_embeds = self.adapter(feats.to(self.encoder_device, dtype=self.dtype))

        prompt_ids = self.tokenizer(task_prompt, return_tensors="pt").input_ids.to(self.llm_device)
        prompt_embeds = self.embed_tokens(prompt_ids).to(self.dtype)

        # Move speech embeds to LLM device
        speech_embeds = speech_embeds.to(self.llm_device, dtype=self.dtype)
        inputs_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)

        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text


def smoke_test():
    print("=" * 60)
    print("Thai SLU Pipeline Smoke Test")
    print("=" * 60)

    base = Path("/data/workspace/asr-model-training/thai-understanding")
    pipeline = ThaiSLUPipeline(
        encoder_ckpt=str(base / "XLSR-Thai/checkpoint_best.pt"),
        llm_path=str(base / "Typhoon2-3B"),
        encoder_device="cuda",
        llm_device="cpu",  # save GPU memory
    )

    audio = torch.randn(1, 16000 * 5)
    print(f"\n[TEST] Random 5s audio -> encode -> adapt")
    feats = pipeline.encode_audio(audio)
    print(f"  encoder output: {feats.shape}")
    adapted = pipeline.adapter(feats.cuda().to(pipeline.dtype))
    print(f"  adapted to LLM: {adapted.shape}")

    task_prompt = "จำแนกเจตนาของเสียงนี้:"
    print(f"\n[INFER] Task prompt: {task_prompt!r}")
    print("  (adapter weights are RANDOM — output will be nonsense)")
    import time
    t0 = time.time()
    text = pipeline.generate(audio, task_prompt, max_new_tokens=16)
    print(f"  generated ({time.time()-t0:.1f}s): {text!r}")
    print("\n[OK] Pipeline works end-to-end.")


if __name__ == "__main__":
    smoke_test()
