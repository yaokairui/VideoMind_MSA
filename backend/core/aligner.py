import numpy as np


class MultimodalAligner:
    """
    多模态时序对齐器：确保视觉、音频、文本序列长度严格一致
    """

    def __init__(self, target_len=50):
        # 目标长度 L，通常根据数据集（如 MOSEI）的平均长度设定
        self.target_len = target_len

    def _interpolate_sequence(self, feature, target_len):
        """
        对单个模态的特征序列进行线性插值
        """
        current_len = feature.shape[0]
        if current_len == target_len:
            return feature

        # 创建原始索引（0 到 current_len-1）和目标采样索引
        indices = np.linspace(0, current_len - 1, num=target_len)

        # 对每一个维度（Column）分别进行插值计算
        aligned_feature = np.zeros((target_len, feature.shape[1]), dtype=np.float32)
        for i in range(feature.shape[1]):
            # np.interp(目标位置, 原始位置, 原始值)
            aligned_feature[:, i] = np.interp(indices, np.arange(current_len), feature[:, i])

        return aligned_feature

    def align(self, visual_feat, audio_feat, text_feat):
        """
        三模态联合对齐核心函数
        """
        # 1. 视觉与音频通常是多帧序列，直接插值
        v_aligned = self._interpolate_sequence(visual_feat, self.target_len)
        a_aligned = self._interpolate_sequence(audio_feat, self.target_len)

        # 2. 文本特征处理逻辑
        # 如果 text_feat 是 (768,) 维度（仅 [CLS]），则通过 Repeat 广播至全序列
        if text_feat.ndim == 1:
            t_aligned = np.tile(text_feat, (self.target_len, 1))
        # 如果是词级别序列 (Words_len, 768)，则进行插值对齐
        else:
            t_aligned = self._interpolate_sequence(text_feat, self.target_len)

        return v_aligned, a_aligned, t_aligned
