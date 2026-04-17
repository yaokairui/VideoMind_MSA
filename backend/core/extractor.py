import os
import cv2
import torch
import librosa
import numpy as np

# --- 核心修复：显式子模块导入，解决 Windows 环境下的加载问题 ---
import mediapipe as mp
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except ImportError:
    from mediapipe.solutions import face_mesh as mp_face_mesh
# -----------------------------------------------------------

from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer, AutoModel

class FeatureExtractor:
    def __init__(self, config):
        """
        初始化特征提取引擎
        """
        self.config = config
        # 自动检测 GPU (针对你的 Dell G15 RTX 30 显卡)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 FeatureExtractor 正在运行在: {self.device}")

        # 1. 初始化面部关键点工具 (MediaPipe)
        self.face_mesh_tool = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # 2. 初始化文本语义模型 (BERT)
        print("📥 正在加载 BERT 预训练模型...")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.bert_model.eval()
        print("✅ 预处理引擎初始化完成")

    def extract_vision(self, video_path):
        """
        视觉流提取：强制每一帧输出 1404 维特征 (468点 * 3坐标)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        sample_rate = self.config['preprocessing']['fps']
        interval = max(1, int(fps // sample_rate))

        visual_features = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh_tool.process(rgb_frame)
                
                # 初始化一个全零的 1404 维向量 (float32)
                # 这样做可以确保即使没检出人脸，维度也是对齐的
                current_feat = np.zeros(1404, dtype=np.float32)
                
                if results.multi_face_landmarks:
                    # 提取第一张脸的 468 个 Landmark
                    landmarks = results.multi_face_landmarks[0]
                    # 转换为 (1404,) 的一维数组
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32).flatten()
                    
                    # 严格校验：只有长度正好是 1404 时才进行替换
                    if coords.shape[0] == 1404:
                        current_feat = coords
                
                visual_features.append(current_feat)
            
            frame_count += 1
        
        cap.release()

        # 核心修复：使用 np.stack 确保所有帧特征拼接成一个规整的 (N, 1404) 矩阵
        if len(visual_features) > 0:
            return np.stack(visual_features)
        else:
            return np.zeros((1, 1404), dtype=np.float32)

    def extract_audio(self, video_path):
        """
        音频流提取：提取 20 维 MFCC 特征
        """
        temp_wav = video_path.replace(".mp4", ".wav")
        try:
            # 1. 临时提取音轨
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(temp_wav, codec='pcm_s16le', verbose=False, logger=None)
            
            # 2. 提取特征
            y, sr = librosa.load(temp_wav, sr=self.config['preprocessing']['sr'])
            # 确保跳窗步长与视觉 FPS 对应，便于后续对齐
            hop_len = int(sr / self.config['preprocessing']['fps'])
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_len)
            
            video.close()
            if os.path.exists(temp_wav): os.remove(temp_wav)
            
            # 返回形状为 (时间步, 20) 的数组
            return np.array(mfcc.T, dtype=np.float32)
        except Exception as e:
            print(f"❌ 音频提取失败: {e}")
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return np.zeros((1, 20), dtype=np.float32)

    def extract_text(self, text):
        """
        文本流提取：提取 768 维 BERT Embedding
        """
        if not text:
            return np.zeros((1, 768), dtype=np.float32)

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # 取最后一层的所有 Token 嵌入
            embeddings = outputs.last_hidden_state.cpu().numpy()
            
        return np.array(embeddings.squeeze(0), dtype=np.float32)

    def process_all(self, video_path, text):
        """
        全自动化特征流水线
        """
        print(f"🎬 开始处理视频: {os.path.basename(video_path)}")
        
        # 1. 视觉特征 (Frames, 1404)
        v_feat = self.extract_vision(video_path)
        # 2. 音频特征 (Time, 20)
        a_feat = self.extract_audio(video_path)
        # 3. 文本特征 (Words, 768)
        t_feat = self.extract_text(text)
        
        # 在这里进行最后的打印，方便你调试维度
        print(f"📊 特征提取完成 -> 视觉: {v_feat.shape}, 音频: {a_feat.shape}, 文本: {t_feat.shape}")
        
        return v_feat, a_feat, t_feat