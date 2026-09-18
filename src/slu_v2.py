"""
SLU pipeline v2: pluggable encoder (XLSR-Thai or ZikXewen) + adapter + LLM.

Two architectures:
  - U-Align style: audio → encoder → adapter → LLM token embeddings → LLM
  - ASR-based style: audio → encoder → CTC → text → tokenize → LLM

Both use the same encoder, adapter, and LLM. The difference is where speech
becomes text/embeddings.
"""
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from xlsr_thai import load_xlsr_thai


class StridedMeanSubsampler(nn.Module):
    def forward(self, x):  # (B, T, D)
        B, T, D = x.shape
        if T % 2 == 1:
            x = x[:, :-1, :]
        return (x[:, 0::2, :] + x[:, 1::2, :]) / 2


class ModalityAdapter(nn.Module):
    def __init__(self, encoder_dim=1024, llm_dim=3072, downsample=2):
        super().__init__()
        self.norm = nn.LayerNorm(encoder_dim)
        self.subsampler = StridedMeanSubsampler()
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):
        return self.proj(self.subsampler(self.norm(x)))


class SLUPipelineV2:
    """Pluggable-encoder SLU pipeline with TWO inference modes:

    mode='speech_embedding': U-Align style (audio → embeddings → LLM)
    mode='asr_then_text':    ASR-based (audio → text → tokenize → LLM)
    """

    def __init__(
        self,
        encoder_type: str,  # 'xlsr_thai' or 'zikxewen'
        xlsr_thai_path: str = None,
        zikxewen_path: str = None,
        llm_path: str = None,
        llm_dim: int = 3072,
        encoder_device: str = 'cuda',
        llm_device: str = 'cpu',
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.encoder_type = encoder_type
        self.encoder_device = encoder_device
        self.llm_device = llm_device
        self.dtype = dtype

        # === Load encoder ===
        if encoder_type == 'xlsr_thai':
            assert xlsr_thai_path, "xlsr_thai_path required"
            print(f"[SLU-v2] Loading XLSR-Thai SSL encoder from {xlsr_thai_path}...")
            self.encoder, self.encoder_cfg = load_xlsr_thai(xlsr_thai_path, device=encoder_device)
            self.encoder_dim = self.encoder_cfg['encoder_embed_dim']
            self.encoder_mode = 'ssl'  # SSL encoder, no CTC head
        elif encoder_type == 'zikxewen':
            assert zikxewen_path, "zikxewen_path required"
            print(f"[SLU-v2] Loading ZikXewen XLSR-53-Thai (encoder + CTC head)...")
            self.ctc_model = Wav2Vec2ForCTC.from_pretrained(zikxewen_path)
            self.ctc_model = self.ctc_model.cuda().eval()
            self.encoder_dim = self.ctc_model.config.hidden_size  # 1024
            self.encoder_mode = 'ctc'  # has CTC head available
            # CTC tokenizer/processor for ASR
            self.tokenizer_asr = Wav2Vec2CTCTokenizer(
                vocab_file=str(Path(zikxewen_path) / 'vocab.json'),
                unk_token='<unk>', pad_token='<pad>', word_delimiter_token='|',
            )
            self.feature_extractor_asr = Wav2Vec2FeatureExtractor(
                feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False,
            )
            self.processor_asr = Wav2Vec2Processor(
                feature_extractor=self.feature_extractor_asr, tokenizer=self.tokenizer_asr
            )
            self.encoder = self.ctc_model.wav2vec2  # raw encoder
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Adapter
        print(f"[SLU-v2] Initializing adapter ({self.encoder_dim} -> {llm_dim}) on {encoder_device}")
        self.adapter = ModalityAdapter(self.encoder_dim, llm_dim).to(encoder_device, dtype=dtype)

        # LLM
        print(f"[SLU-v2] Loading LLM from {llm_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=dtype).to(llm_device)
        self.llm.eval()
        self.embed_tokens = self.llm.get_input_embeddings()
        for p in self.llm.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _encode_audio_raw(self, audio: torch.Tensor) -> torch.Tensor:
        """Get encoder features (1024-dim)."""
        if self.encoder_mode == 'ssl':
            return self.encoder(audio)
        else:
            return self.encoder(audio).last_hidden_state

    @torch.no_grad()
    def _transcribe_with_ctc(self, audio_np: 'np.ndarray') -> str:
        """For ZikXewen: transcribe to text via CTC head."""
        inputs = self.processor_asr(audio_np, sampling_rate=16000, return_tensors='pt').input_values.cuda()
        logits = self.ctc_model(inputs).logits
        return self.processor_asr.batch_decode(torch.argmax(logits, dim=-1))[0]

    @torch.no_grad()
    def generate_u_align(self, audio: torch.Tensor, task_prompt: str, max_new_tokens: int = 32) -> str:
        """Mode 1: U-Align style. Audio → encoder → adapter → LLM."""
        feats = self._encode_audio_raw(audio)  # (1, T, 1024)
        if isinstance(feats, tuple):
            feats = feats[0]
        speech_embeds = self.adapter(feats.to(self.encoder_device, dtype=self.dtype))
        prompt_ids = self.tokenizer(task_prompt, return_tensors='pt').input_ids.to(self.llm_device)
        prompt_embeds = self.embed_tokens(prompt_ids).to(self.dtype)
        speech_embeds = speech_embeds.to(self.llm_device, dtype=self.dtype)
        inputs_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_asr_then_text(self, audio_np: 'np.ndarray', task_prompt_template: str, max_new_tokens: int = 32) -> str:
        """Mode 2: ASR-based. Audio → text via CTC → tokenize → LLM text pipeline."""
        if self.encoder_mode != 'ctc':
            raise NotImplementedError("ASR mode requires a CTC head (use encoder_type='zikxewen')")
        transcript = self._transcribe_with_ctc(audio_np)
        # Strip the word delimiter | for cleaner prompt
        transcript = transcript.replace('|', ' ').strip()
        prompt = task_prompt_template.format(transcript=transcript)
        ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.llm_device)
        out = self.llm.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True), transcript


def demo_both_modes():
    """Run both pipelines on the same SR sample for comparison."""
    import io
    import numpy as np
    import soundfile as sf
    import pyarrow.parquet as pq

    base = Path("/data/workspace/asr-model-training/thai-understanding")

    # Load a sample
    print("Loading sample from SR/dev...")
    table = pq.read_table(str(base / "Thai-SUP/SR/dev/dev-00000.parquet"))
    df = table.to_pandas()
    row = df.iloc[0]
    audio_bytes = row['audio_flac']
    audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).cuda()
    print(f"Sample: text={row['text']!r}")
    print(f"        duration={row['duration_s']:.1f}s")

    # === Pipeline A: XLSR-Thai + adapter + LLM (U-Align style, random adapter) ===
    print(f"\n{'='*60}")
    print("Pipeline A: XLSR-Thai SSL encoder + random adapter + LLM (U-Align style)")
    print(f"{'='*60}")
    pipe_a = SLUPipelineV2(
        encoder_type='xlsr_thai',
        xlsr_thai_path=str(base / 'XLSR-Thai/checkpoint_best.pt'),
        llm_path=str(base / 'Typhoon2-3B'),
        encoder_device='cuda', llm_device='cpu',
    )
    import time
    t0 = time.time()
    out_a = pipe_a.generate_u_align(audio_tensor, "มันเป็นข้อความเสียงโปรดเขียนมันใหม่:", max_new_tokens=24)
    print(f"  Out ({time.time()-t0:.1f}s): {out_a!r}")
    print(f"  Note: random adapter → nonsense output (expected smoke test)")

    # === Pipeline B: ZikXewen + CTC + LLM (ASR-based style) ===
    print(f"\n{'='*60}")
    print("Pipeline B: ZikXewen (CTC) + LLM (ASR-based style)")
    print(f"{'='*60}")
    pipe_b = SLUPipelineV2(
        encoder_type='zikxewen',
        zikxewen_path=str(base / 'ZikXewen-thai-ctc'),
        llm_path=str(base / 'Typhoon2-3B'),
        encoder_device='cuda', llm_device='cpu',
    )
    # ASR-based: transcribe first, then feed to LLM with transcript as context
    t0 = time.time()
    out_b, transcript = pipe_b.generate_asr_then_text(
        audio_np,
        task_prompt_template="ข้อความนี้คือ: {transcript}\nโปรดเขียนใหม่:",
        max_new_tokens=24,
    )
    print(f"  ASR transcript ({time.time()-t0:.1f}s): {transcript!r}")
    print(f"  LLM output: {out_b!r}")
    print(f"  Note: ASR is real (pretrained), LLM is actually rewriting the transcript")

    # === Pipeline C: ZikXewen encoder + random adapter + LLM (U-Align w/ ZikXewen) ===
    print(f"\n{'='*60}")
    print("Pipeline C: ZikXewen encoder (drop CTC) + random adapter + LLM")
    print(f"{'='*60}")
    # Reuse pipe_b's encoder (just don't use CTC)
    t0 = time.time()
    out_c = pipe_b.generate_u_align(audio_tensor, "มันเป็นข้อความเสียงโปรดเขียนมันใหม่:", max_new_tokens=24)
    print(f"  Out ({time.time()-t0:.1f}s): {out_c!r}")
    print(f"  Note: ZikXewen encoder features + random adapter → also nonsense (smoke test)")


if __name__ == "__main__":
    demo_both_modes()
