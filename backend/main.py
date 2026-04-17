import os
import yaml
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# 导入我们的核心模块
from core.extractor import FeatureExtractor

app = FastAPI(title="VideoMind MSA")

# 解决跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 加载配置并初始化提取器 (单例模式)
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 这一步会加载 BERT，利用你的 RTX 30 显存
extractor = FeatureExtractor(config)

# 存储任务状态（生产环境中建议使用 Redis）
task_status = {}


def run_extraction_pipeline(file_id: str, video_path: str, text: str = "This video is amazing"):
    """
    后台运行的提取流水线
    """
    try:
        task_status[file_id] = "正在提取多模态特征..."

        # 执行三模态提取
        v_feat, a_feat, t_feat = extractor.process_all(video_path, text)

        # 结果持久化 (保存为 .npy 供后续对齐和模型使用)
        save_path = os.path.join(config['paths']['feature_dir'], f"{file_id}_features.npz")
        np.savez(save_path, vision=v_feat, audio=a_feat, text=t_feat)

        task_status[file_id] = "特征提取完成，等待对齐与融合"
        print(f"✅ 任务 {file_id} 特征已保存至: {save_path}")

    except Exception as e:
        task_status[file_id] = f"处理失败: {str(e)}"


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: File = File(...)):
    # 1. 生成唯一 ID 并保存视频
    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1]
    video_path = os.path.join(config['paths']['upload_dir'], f"{file_id}{ext}")

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # 2. 注册后台任务，不阻塞前端
    # 在 MOSI/MOSEI 数据集中，通常需要 ASR 文本，这里先预留接口 
    background_tasks.add_task(run_extraction_pipeline, file_id, video_path)

    return {
        "status": "started",
        "task_id": file_id,
        "message": "视频上传成功，后端已启动异步特征提取"
    }


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    return {"task_id": task_id, "status": task_status.get(task_id, "未知任务")}


if __name__ == "__main__":
    import uvicorn
    import numpy as np  # 确保导入

    uvicorn.run(app, host="127.0.0.1", port=8000)
