# VideoMind_MSA 项目架构设计文档

## 1. 项目愿景 (Vision)

VideoMind_MSA 是一个基于多模态学习的视频情感分析系统。它能够接收一段原始视频，通过并行处理视觉表情、语音声学和文本语义三路特征，利用 Transformer 注意力机制进行时序对齐与融合，最终产出精准的情感倾向评价，并通过 AI Agent 给出具备可解释性的分析报告。

## 2. 核心技术栈 (Technical Stack)

- **前端**: Vue 3 (Vite) + Element Plus + ECharts (可视化)
- **后端**: FastAPI (Python) + Uvicorn (高性能服务器)
- **特征提取**: OpenCV, MediaPipe (视觉), LibROSA (音频), BERT (文本)
- **深度学习**: PyTorch + Transformers (Hugging Face)
- **解说 Agent**: DeepSeek API / 智谱 AI (在线大模型)

## 3. 模块化设计说明 (Module Breakdown)

### 📂 backend/core/extractor.py (特征提取引擎)
- **功能描述**: 系统的"感官"，将原始非结构化视频转化为高维数值特征矩阵。
- **视觉流**: 利用 MediaPipe 捕捉 468 个面部关键点坐标，转化为 $1404$ 维序列。
- **音频流**: 通过 LibROSA 提取 MFCC 和能量特征，捕捉语音的声学起伏。
- **文本流**: 利用 BERT-base-uncased 将字幕文本转化为 $768$ 维的词嵌入向量。

### 📂 backend/core/aligner.py (时序对齐工具)
- **功能描述**: 解决多模态数据频率不一致的"断层"问题。
- **逻辑原理**: 通过线性插值算法（Linear Interpolation），将不同步长的三模态特征强行映射到统一的时间轴长度 $L$（如 50 帧）。
- **数学支撑**:
  $$X_{aligned} = Interpolate(X_{raw}, target\_length = L)$$

### 📂 backend/core/model.py (融合模型核心)
- **功能描述**: 实现任务书中的"模型级融合策略"。
- **技术细节**: 采用多头交叉注意力机制 (Multi-head Cross-Attention)，让模型学会"当文本是讽刺时，多看一眼视觉表情"。
- **输出**: 产出 -3 (极度消极) 到 +3 (极度积极) 的连续情感分值。

### 📂 backend/agent/llm_agent.py (解释性对话机器人)

- **功能描述**: 赋予系统"说话"的能力，解决深度学习模型黑盒不可见的问题。
- **输入**: 将模型计算出的特征统计量、预测分值、关键词喂给 LLM。
- **功能**: 回答用户关于"为什么这段视频被判定为积极"或"视频中哪里最感人"的问题。

### 📂 frontend/src/App.vue (Apple 风格交互界面)
- **功能描述**: 系统的脸面，提供极简、高级的用户体验。
- **功能点**: 文件拖拽上传、分析进度实时反馈、情感波动图表展示、iMessage 风格对话框。

## 4. 系统工作流 (System Workflow)

1. **上传阶段**: 用户在 Vue 界面上传 .mp4 视频，后端 FastAPI 接收并分配唯一的 task_id。
2. **提取阶段**: extractor.py 启动，调用 RTX 30 显卡并行提取视觉、音频和文本特征。
3. **对齐阶段**: aligner.py 介入，确保三路特征矩阵在序列长度维度上完全一致，满足 Transformer 输入要求。
4. **推理阶段**: model.py 加载预训练的融合模型权重，计算出该视频的情感分布曲线。
5. **交互阶段**: llm_agent.py 结合推理结果，在前端聊天窗口中与用户进行对话，解释分析依据。

## 5. 对论文的支撑映射 (Thesis Mapping)

| 任务书章节 | 对应项目代码模块 | 技术亮点 |
|-----------|-----------------|----------|
| 任务 1：预处理与特征提取 | extractor.py | MediaPipe + LibROSA + BERT 三路并行提取 |
| 任务 2：多模态融合策略 | aligner.py & model.py | 基于时序插值的 Cross-modal Attention 融合 |
| 任务 3：模型训练与调优 | backend/checkpoints/ | 模型权重管理与 Ablation (消融) 实验 |
| 任务 4：系统原型开发 | frontend/ & main.py | 前后端分离架构、异步任务管理、AI Agent 交互 |