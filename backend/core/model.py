import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    核心组件：交叉模态注意力层
    让模态 A 作为 Query，去关注模态 B 的 Key 和 Value
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_feat, key_value_feat):
        # MultiheadAttention 输入格式要求: (Sequence_len, Batch, Dim)
        # 我们之前对齐后的格式是 (Batch, Seq_len, Dim)，需要转置
        q = query_feat.transpose(0, 1)
        kv = key_value_feat.transpose(0, 1)
        
        attn_output, _ = self.attention(q, kv, kv)
        # 残差连接与层归一化
        output = self.norm(q + self.dropout(attn_output))
        return output.transpose(0, 1)

class MultimodalTransformer(nn.Module):
    """
    任务书要求的端到端深度学习模型 
    """
    def __init__(self, config):
        super().__init__()
        d_model = 128  # 统一的内部嵌入维度
        nhead = 4      # 注意力头数
        
        # 1. 投影层：将不同维度的原始特征映射到统一的 d_model 维度
        self.v_proj = nn.Linear(1404, d_model)  # 视觉: MediaPipe 1404维
        self.a_proj = nn.Linear(20, d_model)    # 音频: MFCC 20维
        self.t_proj = nn.Linear(768, d_model)   # 文本: BERT 768维

        # 2. 交叉注意力融合层 (以文本为核心进行跨模态增强)
        self.trans_v_to_t = CrossModalAttention(d_model, nhead)
        self.trans_a_to_t = CrossModalAttention(d_model, nhead)

        # 3. 情感分类头 (Classification Head) 
        # 根据 MOSEI 任务，输出通常为连续的情感得分 (-3 to +3)
        self.fusion_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # 输出情感分值
        )

    def forward(self, vision, audio, text):
        # A. 特征投影
        v = F.relu(self.v_proj(vision))
        a = F.relu(self.a_proj(audio))
        t = F.relu(self.t_proj(text))

        # B. 跨模态注意力融合：让文本语义吸收视觉与音频的上下文
        # 模拟人类感知：一边读字幕(T)，一边看表情(V)和听语气(A)
        h_vt = self.trans_v_to_t(t, v) 
        h_at = self.trans_a_to_t(t, a)

        # C. 特征拼接与全局编码
        # 融合视觉增强文本和音频增强文本
        fusion_feat = (h_vt + h_at) / 2
        
        # D. 池化并输出结果 (取序列的均值作为全局情感表示)
        cls_feat = fusion_feat.mean(dim=1)
        output = self.classifier(cls_feat)
        
        return output