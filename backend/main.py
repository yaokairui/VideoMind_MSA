import os
import yaml
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# 导入自定义核心模块
from core.extractor import FeatureExtractor
from core.aligner import MultimodalAligner

app = FastAPI(title="VideoMind MSA API")

# 解决前后端分离架构中的跨域问题 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境下允许所有来源
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 加载全局配置
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"未找到配置文件: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 2. 初始化核心组件 (单例模式，常驻内存)
# FeatureExtractor 初始化时会自动检测并占用显卡 (CUDA) 资源
extractor = FeatureExtractor(config)
# Aligner 负责将异构的特征序列对齐至统一长度
aligner = MultimodalAligner(target_len=config['preprocessing']['max_seq_len'])

# 3. 任务状态追踪字典 (内存存储，重启后重置)
task_status = {}


def run_extraction_pipeline(file_id: str, video_path: str, text: str = "This video is amazing"):
    """
    核心自动化流水线：
    Step 1: 特征提取 (Extractor)
    Step 2: 时序对齐 (Aligner)
    Step 3: 结果持久化 (.npz)
    """
    try:
        # A. 执行特征提取
        task_status[file_id] = "正在提取视觉、音频与文本特征..."
        v_raw, a_raw, t_raw = extractor.process_all(video_path, text)

        # B. 执行时序对齐
        task_status[file_id] = "正在执行三模态时序对齐..."
        v_aligned, a_aligned, t_aligned = aligner.align(v_raw, a_raw, t_raw)

        # C. 特征持久化保存
        feature_dir = config['paths']['feature_dir']
        os.makedirs(feature_dir, exist_ok=True)

        save_path = os.path.join(feature_dir, f"{file_id}_aligned.npz")
        np.savez(save_path,
                 vision=v_aligned,
                 audio=a_aligned,
                 text=t_aligned)

        task_status[file_id] = "处理完成，特征已就绪"
        print(f"✅ 任务 {file_id} 完成。特征对齐结果已保存: {save_path}")

    except Exception as e:
        error_info = f"流水线处理失败: {str(e)}"
        task_status[file_id] = error_info
        print(f"❌ 任务 {file_id} 报错: {error_info}")


@app.get("/")
async def health_check():
    return {"status": "online", "service": "VideoMind MSA Backend"}


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: File = File(...)):
    """
    视频上传与异步处理入口
    """
    # 1. 生成任务 ID 并保存原始视频
    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1]

    upload_dir = config['paths']['upload_dir']
    os.makedirs(upload_dir, exist_ok=True)

    video_path = os.path.join(upload_dir, f"{file_id}{ext}")

    with open(video_path, "wb") as f:
        f.write(await file.read())

    # 2. 将计算密集型任务放入后台，立即返回响应
    # 注意：在实际数据集测试中，text 建议来源于 ASR 结果或对应的 .txt 文件
    background_tasks.add_task(run_extraction_pipeline, file_id, video_path)

    task_status[file_id] = "视频已上传，异步流水线启动..."

    return {
        "status": "started",
        "task_id": file_id,
        "message": "视频已成功接收，正在进行多模态深度分析。"
    }


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    前端轮询接口，用于在 UI 上实时展示处理进度
    """
    status = task_status.get(task_id, "任务不存在")
    return {"task_id": task_id, "status": status}


if __name__ == "__main__":
    import uvicorn

    # 启动服务，默认端口 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
