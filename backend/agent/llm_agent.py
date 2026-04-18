import os
import yaml
from openai import OpenAI

class SentimentAgent:
    """
    VideoMind AI 情感分析助手
    负责将多模态融合结果转化为自然语言解释
    """
    def __init__(self, config):
        self.config = config
        self.agent_config = config.get('agent', {})
        
        # 初始化客户端 (适配 DeepSeek, 智谱, OpenAI 等)
        self.client = OpenAI(
            api_key=self.agent_config.get('api_key'),
            base_url=self.agent_config.get('base_url')
        )
        self.model_name = self.agent_config.get('model', 'deepseek-chat')

    def generate_explanation(self, analysis_result, user_query="请解释一下这个视频的情感表现"):
        """
        根据融合模型的结果生成解释
        analysis_result: 包含 score 和 label 的字典
        """
        score = analysis_result.get('score', 0)
        label = analysis_result.get('label', '未知')
        
        # 构造系统提示词 (System Prompt) - 确立 Apple 风格的专业语气
        system_prompt = (
            "你是一位精通多模态情感分析的资深专家。你的任务是根据系统提取的数据，"
            "为用户提供专业、简洁、且具有启发性的视频情感分析报告。"
            "语气要符合 Apple 主义风格：极简、高级、克制且温暖。"
        )

        # 构造上下文 (Context)
        context = (
            f"当前视频分析数据如下：\n"
            f"- 融合情感得分：{score} (量表范围：-3.0 到 +3.0)\n"
            f"- 情感极性判定：{label}\n"
            f"- 模态对齐状态：已完成 50 帧时序对齐\n"
            f"\n请结合以上数据回答用户的问题：'{user_query}'"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Agent 暂时无法连接: {str(e)}"

    def chat(self, history, new_query, analysis_result):
        """
        支持多轮对话
        """
        # 这里可以扩展成带记忆的对话，目前先实现基础调用
        return self.generate_explanation(analysis_result, new_query)