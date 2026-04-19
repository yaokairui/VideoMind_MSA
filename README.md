# VideoMind_MSA

基于多模态学习的视频情感分析系统

## 项目概述

VideoMind_MSA 是一个基于多模态学习的视频情感分析系统。它能够接收一段原始视频，通过并行处理视觉表情、语音声学和文本语义三路特征，利用 Transformer 注意力机制进行时序对齐与融合，最终产出精准的情感倾向评价，并通过 AI Agent 给出具备可解释性的分析报告。

## 功能特性

### 核心功能

- **三路特征提取**: 并行处理视觉（MediaPipe）、音频（LibROSA）、文本（BERT）特征
- **时序对齐**: 线性插值算法解决多模态数据频率不一致问题
- **多模态融合**: 多头交叉注意力机制（Multi-head Cross-Attention）实现特征融合
- **情感评分**: 产出 -3（极度消极）到 +3（极度积极）的连续情感分值
- **AI 解释**: 基于 DeepSeek/智谱 AI 提供可解释性分析报告

### 技术亮点

- MediaPipe 捕捉 468 个面部关键点坐标，转化为 1404 维序列
- BERT-base-uncased 生成 768 维词嵌入向量
- Cross-modal Attention 融合策略
- 前后端分离架构，支持异步任务管理

## 项目结构

```
VideoMind_MSA/
├── backend/                    # FastAPI 后端
│   ├── main.py                # 程序入口
│   ├── config.yaml            # 配置文件
│   ├── core/                  # 核心模块
│   │   ├── extractor.py       # 特征提取引擎
│   │   ├── aligner.py         # 时序对齐工具
│   │   └── model.py           # 融合模型
│   ├── agent/                 # 对话机器人
│   │   └── llm_agent.py      # LLM Agent
│   └── uploads/               # 用户上传视频目录
└── frontend/                  # Vue 3 前端
    └── src/
        └── App.vue            # 主界面组件
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 前端 | Vue 3 (Vite), Element Plus, ECharts |
| 后端 | FastAPI, Uvicorn, Python |
| 视觉特征 | OpenCV, MediaPipe |
| 音频特征 | LibROSA |
| 文本特征 | BERT (Hugging Face) |
| 深度学习 | PyTorch, Transformers |
| AI Agent | DeepSeek API / 智谱 AI |

## 安装说明

### 环境要求

- Python 3.8+
- Node.js 16+
- CUDA 11.0+ (GPU 支持，可选)
- Windows/Linux/macOS

### 后端安装

```bash
# 克隆项目
git clone https://github.com/your-repo/VideoMind_MSA.git
cd VideoMind_MSA

# 创建虚拟环境
conda create -n videomind python=3.10
conda activate videomind

# 安装 PyTorch (GPU 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装后端依赖
pip install fastapi uvicorn opencv-python mediapipe librosa
pip install transformers torch torchvision huggingface_hub
pip install pyyaml python-multipart

# 安装 LLM 相关
pip install deepseek-sdk zhipuai
```

### 前端安装

```bash
cd frontend

# 安装依赖
npm install

# 或使用 yarn
yarn install
```

## 使用指南

### 启动后端服务

```bash
cd backend

# 启动 FastAPI 服务
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# API 文档地址: http://localhost:8000/docs
```

### 启动前端服务

```bash
cd frontend

# 开发模式
npm run dev

# 生产构建
npm run build
```

### API 调用示例

```bash
# 上传视频进行情感分析
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@video.mp4"

# 获取分析结果
curl -X GET "http://localhost:8000/api/result/{task_id}"

# 与 AI Agent 对话
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "xxx", "question": "为什么这段视频被判定为积极？"}'
```

## 配置方法

配置文件位于 `backend/config.yaml`：

```yaml
# 服务器配置
server:
  host: "0.0.0.0"
  port: 8000

# 模型配置
model:
  vision_dim: 1404      # 视觉特征维度
  audio_dim: 128        # 音频特征维度
  text_dim: 768         # 文本特征维度
  sequence_length: 50   # 对齐后的序列长度
  output_range: [-3, 3] # 情感分值范围

# LLM 配置
llm:
  provider: "deepseek"  # deepseek 或 zhipuai
  api_key: "your-api-key"
  model: "deepseek-chat"

# 路径配置
paths:
  upload_dir: "uploads"
  checkpoint_dir: "checkpoints"

# 显存限制
gpu:
  memory_limit: "4GB"
```

## 系统工作流

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   上传阶段   │ ──> │   提取阶段   │ ──> │   对齐阶段   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               v
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   交互阶段   │ <── │   推理阶段   │ <── │   融合阶段   │
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **上传阶段**: 用户上传 .mp4 视频，后端分配唯一 task_id
2. **提取阶段**: 并行提取视觉、音频、文本特征
3. **对齐阶段**: 三路特征映射到统一时间轴（50帧）
4. **推理阶段**: 融合模型计算情感分布曲线
5. **交互阶段**: AI Agent 与用户对话解释分析结果

## 贡献规范

欢迎提交 Issue 和 Pull Request！

### 提交规范

- Fork 本仓库
- 创建特性分支 (`git checkout -b feature/AmazingFeature`)
- 提交更改 (`git commit -m 'Add some AmazingFeature'`)
- 推送到分支 (`git push origin feature/AmazingFeature`)
- 创建 Pull Request

### 开发规范

- 遵循 PEP 8 代码规范
- 前端代码遵循 ESLint 配置
- 提交前运行测试
- 更新相关文档

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目主页: https://github.com/your-repo/VideoMind_MSA
- 问题反馈: https://github.com/your-repo/VideoMind_MSA/issues
- 邮箱: your-email@example.com

## 致谢

感谢以下开源项目：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Hugging Face Transformers](https://huggingface.co/) - 预训练模型库
- [MediaPipe](https://mediapipe.dev/) - 面部关键点检测
- [LibROSA](https://librosa.org/) - 音频分析
- [Vue 3](https://vuejs.org/) - 前端框架
- [FastAPI](https://fastapi.tiangolo.com/) - 后端框架
