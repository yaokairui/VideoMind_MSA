import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    交叉模态注意力层：实现模态间的动态对齐与增强
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_feat, key_value_feat):
        # MultiheadAttention 输入格式: (Seq_len, Batch, Dim)
        q = query_feat.transpose(0, 1)
        kv = key_value_feat.transpose(0, 1)

        attn_output, _ = self.attention(q, kv, kv)
        # 残差连接与层归一化
        output = self.norm(q + self.dropout(attn_output))
        return output.transpose(0, 1)


class MultimodalTransformer(nn.Module):
    """
    VideoMind MSA 核心融合模型
    """

    def __init__(self, config):
        super().__init__()
        d_model = 128  # 统一的特征维度
        nhead = 4

        # 1. 特征投影层：将异构特征映射到同一维度
        self.v_proj = nn.Linear(1404, d_model)  # 视觉 1404 -> 128
        self.a_proj = nn.Linear(20, d_model)  # 音频 20 -> 128 (MFCC)
        self.t_proj = nn.Linear(768, d_model)  # 文本 768 -> 128 (BERT)

        # 2. 交叉注意力层
        self.trans_v_to_t = CrossModalAttention(d_model, nhead)
        self.trans_a_to_t = CrossModalAttention(d_model, nhead)

        # 3. 最终预测头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出 -3 到 +3 的情感得分
        )

    def forward(self, vision, audio, text):
        # 维度投影
        v = F.relu(self.v_proj(vision))
        a = F.relu(self.a_proj(audio))
        t = F.relu(self.t_proj(text))

        # 跨模态融合
        h_vt = self.trans_v_to_t(t, v)
        h_at = self.trans_a_to_t(t, a)

        # 特征聚合 (残差融合)
        fusion_feat = (h_vt + h_at) / 2

        # 全局池化
        out_feat = fusion_feat.mean(dim=1)
        return self.classifier(out_feat)
