from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

app = FastAPI(title="VideoMind MSA API")

# 解决前后端分离的跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境下允许所有来源
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "VideoMind MSA 后端已启动"}


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: File = File(...)):
    # 1. 保存视频文件
    file_id = str(uuid.uuid4())
    file_path = f"uploads/{file_id}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2. 触发异步任务 (Task 1 + Task 2)
    # background_tasks.add_task(start_pipeline, file_path)

    return {
        "status": "processing",
        "file_id": file_id,
        "message": "视频已上传，后台正在提取多模态特征..."
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)