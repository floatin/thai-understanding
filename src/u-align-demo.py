"""
U-Align 语音-文本对齐方法复现代码
基于论文 arXiv:2509.14804v1: Towards Building Speech Large Language Models for Multitask Understanding in Low-Resource Languages

实现内容：
1. 基于余弦距离的批量DTW对齐损失（对应论文U-Align阶段一）
2. 模态适配器：LayerNorm + CNN下采样 + MLP投影层（对应论文适配器模块）
3. 两阶段训练流程：阶段一DTW对齐，阶段二多任务微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchaudio


class BatchDTWLoss(nn.Module):
    """
    批量DTW损失，基于余弦距离，支持变长序列padding
    对应论文U-Align阶段一的对齐损失
    """
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, speech_feats, text_embeds, speech_lens, text_lens):
        """
        参数：
            speech_feats: [B, T_s, D] 语音特征，padding后的批次
            text_embeds: [B, T_t, D] 文本嵌入，padding后的批次
            speech_lens: [B] 每个样本的真实语音长度
            text_lens: [B] 每个样本的真实文本长度
        返回：
            平均DTW损失
        """
        B, T_s, D = speech_feats.shape
        T_t = text_embeds.shape[1]

        # 归一化特征，计算余弦距离矩阵
        speech_norm = F.normalize(speech_feats, p=2, dim=-1)
        text_norm = F.normalize(text_embeds, p=2, dim=-1)
        cosine_sim = torch.bmm(speech_norm, text_norm.transpose(1, 2))
        cost_matrix = 1 - cosine_sim

        # 初始化DTW矩阵，填充inf
        dtw_matrix = torch.full((B, T_s + 1, T_t + 1), float('inf'), device=speech_feats.device)
        dtw_matrix[:, 0, 0] = 0

        # 动态规划计算DTW
        for i in range(1, T_s + 1):
            for j in range(1, T_t + 1):
                cost = cost_matrix[:, i - 1, j - 1]
                dtw_matrix[:, i, j] = cost + torch.min(
                    torch.stack([
                        dtw_matrix[:, i - 1, j],
                        dtw_matrix[:, i, j - 1],
                        dtw_matrix[:, i - 1, j - 1]
                    ]),
                    dim=0
                ).values

        # 计算每个样本的归一化损失，排除padding部分
        losses = []
        for b in range(B):
            s_len = speech_lens[b].item()
            t_len = text_lens[b].item()
            dtw_dist = dtw_matrix[b, s_len, t_len]
            if self.normalize:
                dtw_dist = dtw_dist / (s_len + t_len)
            losses.append(dtw_dist)

        return torch.stack(losses).mean()


class ModalAdapter(nn.Module):
    """
    模态适配器：LayerNorm → CNN下采样 → MLP投影层
    将语音编码器输出映射到LLM的文本嵌入空间
    对应论文中的模态适配器模块
    """
    def __init__(self, speech_dim=1024, llm_dim=3200, downsample_factor=2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(speech_dim)
        self.cnn_downsample = nn.Sequential(
            nn.Conv1d(speech_dim, speech_dim, kernel_size=3, stride=downsample_factor, padding=1),
            nn.GELU(),
            nn.Conv1d(speech_dim, speech_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.mlp_proj = nn.Sequential(
            nn.Linear(speech_dim, llm_dim * 2),
            nn.GELU(),
            nn.Linear(llm_dim * 2, llm_dim)
        )

    def forward(self, speech_feats, speech_lens):
        """
        参数：
            speech_feats: [B, T_s, D_s] 语音编码器输出
            speech_lens: [B] 真实语音长度
        返回：
            adapted_feats: [B, T_s', D_llm] 适配后的语音特征
            adapted_lens: [B] 适配后的时间长度
        """
        x = self.layer_norm(speech_feats)
        x = x.transpose(1, 2)
        x = self.cnn_downsample(x)
        x = x.transpose(1, 2)
        x = self.mlp_proj(x)
        adapted_lens = (speech_lens + 1) // 2
        return x, adapted_lens


class UAlignTrainer:
    """
    U-Align两阶段训练流程
    """
    def __init__(self, adapter, llm=None, tokenizer=None, device='cuda'):
        self.adapter = adapter.to(device)
        self.device = device
        self.llm = llm
        self.tokenizer = tokenizer
        if self.llm is not None:
            for param in self.llm.parameters():
                param.requires_grad = False

    def stage1_train_step(self, batch, dtw_loss_fn, optimizer):
        """阶段一训练步骤：DTW对齐，无需LLM参与"""
        self.adapter.train()
        speech_feats = batch['speech_feats'].to(self.device)
        text_embeds = batch['text_embeds'].to(self.device)
        speech_lens = batch['speech_lens'].to(self.device)
        text_lens = batch['text_lens'].to(self.device)

        adapted_feats, adapted_lens = self.adapter(speech_feats, speech_lens)
        loss = dtw_loss_fn(adapted_feats, text_embeds, adapted_lens, text_lens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def stage2_train_step(self, batch, ce_loss_fn, optimizer):
        """阶段二训练步骤：多任务微调，固定LLM"""
        self.adapter.train()
        speech_feats = batch['speech_feats'].to(self.device)
        speech_lens = batch['speech_lens'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        adapted_feats, _ = self.adapter(speech_feats, speech_lens)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds[:, :adapted_feats.shape[1], :] = adapted_feats
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        loss = ce_loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


class UAlignDataset(torch.utils.data.Dataset):
    """示例数据集，可根据实际需求修改"""
    def __init__(self, audio_paths, texts, tokenizer, llm_embedding_layer=None, stage=1):
        self.audio_paths = audio_paths
        self.texts = texts
        self.tokenizer = tokenizer
        self.llm_embedding_layer = llm_embedding_layer
        self.stage = stage

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.audio_paths[idx])
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        # 模拟XLSR-Thai的输出，实际替换为真实编码器输出
        speech_feat = torch.randn(100, 1024)
        speech_len = 100

        if self.stage == 1:
            text_ids = self.tokenizer(self.texts[idx], return_tensors='pt')['input_ids'][0]
            text_embed = self.llm_embedding_layer(text_ids)
            text_len = len(text_ids)
            return {
                'speech_feats': speech_feat,
                'text_embeds': text_embed,
                'speech_lens': speech_len,
                'text_lens': text_len
            }
        else:
            task_prompt = "意图分类："
            full_text = task_prompt + self.texts[idx]
            inputs = self.tokenizer(full_text, return_tensors='pt', padding='max_length', max_length=128)
            labels = inputs['input_ids'].clone()
            prompt_len = len(self.tokenizer(task_prompt)['input_ids'])
            labels[:prompt_len] = -100
            return {
                'speech_feats': speech_feat,
                'speech_lens': speech_len,
                'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0],
                'labels': labels
            }


def collate_fn(batch, stage=1):
    """批次整理函数，处理变长序列padding"""
    if stage == 1:
        speech_feats = pad_sequence([b['speech_feats'] for b in batch], batch_first=True, padding_value=0)
        text_embeds = pad_sequence([b['text_embeds'] for b in batch], batch_first=True, padding_value=0)
        speech_lens = torch.tensor([b['speech_lens'] for b in batch])
        text_lens = torch.tensor([b['text_lens'] for b in batch])
        return {
            'speech_feats': speech_feats,
            'text_embeds': text_embeds,
            'speech_lens': speech_lens,
            'text_lens': text_lens
        }
    else:
        speech_feats = pad_sequence([b['speech_feats'] for b in batch], batch_first=True, padding_value=0)
        speech_lens = torch.tensor([b['speech_lens'] for b in batch])
        input_ids = torch.stack([b['input_ids'] for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        labels = torch.stack([b['labels'] for b in batch])
        return {
            'speech_feats': speech_feats,
            'speech_lens': speech_lens,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    speech_dim = 1024
    llm_dim = 3200
    llm_name = 'llama-2-3b-chat-hf'

    adapter = ModalAdapter(speech_dim=speech_dim, llm_dim=llm_dim)
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16).to(device)
    llm_embedding_layer = llm.get_input_embeddings()

    audio_paths = ['audio1.wav', 'audio2.wav']
    texts = ['เปิดเพลงป๊อป', 'พยากรณ์อากาศวันนี้']

    print("=== 阶段一：DTW对齐训练 ===")
    dataset_stage1 = UAlignDataset(audio_paths, texts, tokenizer, llm_embedding_layer, stage=1)
    dataloader_stage1 = torch.utils.data.DataLoader(
        dataset_stage1, batch_size=2, collate_fn=lambda x: collate_fn(x, stage=1)
    )
    dtw_loss_fn = BatchDTWLoss(normalize=True)
    optimizer_stage1 = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    trainer = UAlignTrainer(adapter, llm, tokenizer, device)

    for epoch in range(3):
        total_loss = 0
        for batch in dataloader_stage1:
            loss = trainer.stage1_train_step(batch, dtw_loss_fn, optimizer_stage1)
            total_loss += loss
        print(f"Epoch {epoch+1}, 平均损失: {total_loss/len(dataloader_stage1):.4f}")

    print("\n=== 阶段二：多任务微调 ===")
    dataset_stage2 = UAlignDataset(audio_paths, texts, tokenizer, stage=2)
    dataloader_stage2 = torch.utils.data.DataLoader(
        dataset_stage2, batch_size=2, collate_fn=lambda x: collate_fn(x, stage=2)
    )
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer_stage2 = torch.optim.AdamW(adapter.parameters(), lr=5e-5)

    for epoch in range(3):
        total_loss = 0
        for batch in dataloader_stage2:
            loss = trainer.stage2_train_step(batch, ce_loss_fn, optimizer_stage2)
            total_loss += loss
        print(f"Epoch {epoch+1}, 平均损失: {total_loss/len(dataloader_stage2):.4f}")

    torch.save(adapter.state_dict(), 'u_align_adapter.pth')
    print("适配器权重已保存为u_align_adapter.pth")


if __name__ == '__main__':
    main()
