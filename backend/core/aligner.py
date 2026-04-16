import numpy as np


class MultimodalAligner:
    """
    多模态时序对齐器：确保视觉、音频、文本序列长度一致
    """

    def __init__(self, target_len=50):
        self.target_len = target_len

    def align_sequence(self, feature, target_len=None):
        """
        使用线性插值对齐单个模态的特征序列
        """
        if target_len is None:
            target_len = self.target_len

        current_len = feature.shape[0]
        if current_len == target_len:
            return feature

        # 创建原始索引和目标索引
        # 例如：从 10步 插值到 50 步
        indices = np.linspace(0, current_len - 1, num=target_len)

        # 对每一个维度进行插值
        aligned_feature = np.zeros((target_len, feature.shape[1]), dtype=np.float32)
        for i in range(feature.shape[1]):
            aligned_feature[:, i] = np.interp(indices, np.arange(current_len), feature[:, i])

        return aligned_feature

    def align(self, visual_feat, audio_feat, text_feat):
        """
        三模态联合对齐
        """
        v_aligned = self.align_sequence(visual_feat)
        a_aligned = self.align_sequence(audio_feat)

        # 文本如果只是 [CLS] 向量 (1, 768)，我们需要将其广播（Repeat）到 target_len
        if text_feat.ndim == 1 or text_feat.shape[0] == 1:
            t_aligned = np.repeat(text_feat.reshape(1, -1), self.target_len, axis=0)
        else:
            t_aligned = self.align_sequence(text_feat)

        return v_aligned, a_aligned, t_aligned
