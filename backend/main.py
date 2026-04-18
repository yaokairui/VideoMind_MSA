from __future__ import annotations
import sys

import os

try:
    from eval_type_backport import patch
    patch()
except ImportError:
    pass

import yaml
import uuid
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.extractor import FeatureExtractor
from core.aligner import MultimodalAligner
from core.model import MultimodalTransformer
from agent.llm_agent import SentimentAgent


app = FastAPI(title="VideoMind MSA - 最终集成版")

# 解决跨域：允许 Vue 3 前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 1. 初始化系统核心 ====================
# 加载全局配置 [cite: 2]
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 硬件检测：针对你的 Dell G15 RTX 30 显卡优化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 系统大脑已激活。当前推理设备: {device}")

# 实例化核心组件 (单例模式)
# extractor 会加载 BERT 和 MediaPipe
extractor = FeatureExtractor(config)
# aligner 确保三模态长度统一
aligner = MultimodalAligner(target_len=config['preprocessing']['max_seq_len'])
# 在 extractor, aligner 初始化之后
agent = SentimentAgent(config)
# model 加载 Transformer 融合架构 [cite: 2]
model = MultimodalTransformer(config).to(device)
model.eval() # 开启推理模式


# 任务状态与结果存储 (生产环境建议换成 Redis)
task_status = {}
task_results = {}

# ==================== 2. 核心自动化流水线 ====================
def run_video_mind_pipeline(file_id: str, video_path: str, text: str):
    """
    自动化流程：提取 -> 对齐 -> 融合推理 -> 保存结果
    """
    try:
        # Step A: 特征提取
        task_status[file_id] = "🧠 正在解析视频与语音特征..."
        v_raw, a_raw, t_raw = extractor.process_all(video_path, text)
        
        # Step B: 时序对齐
        task_status[file_id] = "📏 执行多模态时序对齐..."
        v_aln, a_aln, t_aln = aligner.align(v_raw, a_raw, t_raw)
        
        # Step C: 模型推理 (大脑思考)
        task_status[file_id] = "🤖 正在进行深度情感融合计算..."
        # 准备数据 Tensor [Batch, Seq, Dim]
        v_tensor = torch.from_numpy(v_aln).unsqueeze(0).to(device)
        a_tensor = torch.from_numpy(a_aln).unsqueeze(0).to(device)
        t_tensor = torch.from_numpy(t_aln).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_score = model(v_tensor, a_tensor, t_tensor)
            score = output_score.item()
        
        # Step D: 结果归档
        task_results[file_id] = {
            "score": round(score, 4),
            "label": "Positive (积极)" if score > 0 else "Negative (消极)",
            "status": "Success",
            "timestamp": str(uuid.uuid1())
        }
        task_status[file_id] = "✅ 分析完成"
        print(f"🌟 任务 {file_id} 分析成功，分值: {score}")

    except Exception as e:
        task_status[file_id] = f"❌ 崩溃报错: {str(e)}"
        print(f"🚨 报错详情: {e}")

# ==================== 3. API 路由定义 ====================

@app.get("/")
async def index():
    return {"message": "VideoMind MSA 后端服务已就绪"}

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    视频上传入口，采用异步 BackgroundTasks 防止前端卡死
    """
    # 生成短 ID
    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1]
    save_name = f"{file_id}{ext}"
    video_path = os.path.join(config['paths']['upload_dir'], save_name)
    
    # 保存视频文件到磁盘
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    # 异步启动全流程流水线
    # 注意：text 参数在正式版中应来自 ASR (语音转文字) 或字幕文件
    default_text = "The user is expressing their feelings in this video."
    background_tasks.add_task(run_video_mind_pipeline, file_id, video_path, default_text)
    
    task_status[file_id] = "🚀 视频已入库，流水线启动"
    
    return {
        "task_id": file_id,
        "status": "started",
        "message": "Apple-style 分析已开始"
    }

@app.get("/status/{task_id}")
async def check_status(task_id: str):
    """
    供前端 Vue 轮询展示处理进度 [cite: 2]
    """
    return {
        "task_id": task_id,
        "current_step": task_status.get(task_id, "任务不存在")
    }

@app.get("/result/{task_id}")
async def get_final_result(task_id: str):
    """
    获取分析最终分值与标签
    """
    result = task_results.get(task_id)
    if not result:
        return {"error": "结果尚未生成或任务不存在"}
    return result

@app.post("/chat/{task_id}")
async def chat_with_agent(task_id: str, payload: dict):
    """
    与 AI Agent 进行关于特定视频任务的对话
    """
    user_query = payload.get("message")
    result = task_results.get(task_id)
    
    if not result:
        return {"reply": "请先完成视频分析，我才能为您提供深度解读。"}
    
    # 调用 Agent 生成回答
    answer = agent.chat([], user_query, result)
    return {"reply": answer}

if __name__ == "__main__":
    import uvicorn
    # 启动后端，运行在 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)