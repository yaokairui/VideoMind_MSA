<template>
  <div class="app-container" :class="{ 'light-theme': isLightMode }">
    <ParticleBackground />

    <button class="theme-toggle" @click="toggleTheme" title="切换黑白模式">
      <svg v-if="!isLightMode" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
      <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    </button>

    <div class="glass-panel main-dashboard">
      <h1 class="title">VideoMind MSA <span class="subtitle">多模态情感分析中枢</span></h1>

      <div class="content-wrapper">
        
        <div class="left-section">
          <div 
            class="upload-box" 
            :class="{ 'is-dragover': isDragging, 'has-video': videoPreviewUrl }"
            @dragenter.prevent="isDragging = true"
            @dragover.prevent="isDragging = true"
            @dragleave.prevent="isDragging = false"
            @drop.prevent="handleDrop"
            @click="triggerFileInput"
          >
            <input type="file" ref="fileInput" @change="handleFileSelect" accept="video/mp4,video/quicktime" hidden />
            
            <video 
              v-if="videoPreviewUrl" 
              :src="videoPreviewUrl" 
              class="video-player" 
              controls 
              preload="metadata"
              @click.stop
            ></video>
            
            <div v-else class="upload-content">
              <svg v-if="!isUploading" class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
              <div v-if="isUploading" class="spinner"></div>
              <p v-if="!isUploading" class="upload-text">{{ isDragging ? '松开鼠标开始分析' : '拖拽视频至此，或点击选择文件' }}</p>
              <p v-else class="upload-text">正在提取多模态特征...</p>
            </div>
          </div>

          <div class="status-box" v-if="taskId">
            <h3 class="box-title">系统状态</h3>
            <p class="status-text animated-text">{{ currentStatus }}</p>
            
            <transition name="fade">
              <div v-if="finalResult" class="result-card">
                <div class="score-row">
                  <span class="label">情感极性:</span>
                  <span class="value" :class="finalResult.score > 0 ? 'positive' : 'negative'">
                    {{ finalResult.label }}
                  </span>
                </div>
                <div class="score-row">
                  <span class="label">融合分值:</span>
                  <span class="value">{{ finalResult.score }}</span>
                </div>
              </div>
            </transition>
          </div>
        </div>

        <div class="right-section">
          <div class="chat-container">
            <div class="chat-header">
              <h3>SentimentAgent 洞察</h3>
              <span class="status-dot" :class="{ 'active': finalResult }"></span>
            </div>
            
            <div class="chat-history" ref="chatBox">
              <div v-if="chatMessages.length === 0" class="empty-chat">
                <p>上传视频并完成分析后，即可与我对话。</p>
              </div>
              
              <div v-for="(msg, index) in chatMessages" :key="index" :class="['chat-bubble', msg.role]">
                <div class="bubble-content">{{ msg.content }}</div>
              </div>
              
              <div v-if="isAiTyping && !isStreamingText" class="chat-bubble ai typing">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
              </div>
            </div>

            <div class="chat-input-area">
              <input 
                type="text" 
                v-model="userInput" 
                @keyup.enter="sendMessage"
                placeholder="问我关于这个视频的情感细节..." 
                :disabled="!finalResult || isAiTyping || isStreamingText"
              />
              <button @click="sendMessage" :disabled="!finalResult || !userInput.trim() || isAiTyping || isStreamingText">发送</button>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue';
import ParticleBackground from './components/ParticleBackground.vue';

const isDragging = ref(false);
const isUploading = ref(false);
const fileInput = ref(null);
const taskId = ref(null);
const currentStatus = ref('等待上传视频...');
const finalResult = ref(null);
let pollingInterval = null;

const videoPreviewUrl = ref(null);
const chatMessages = ref([]);
const userInput = ref('');
const isAiTyping = ref(false);
const isStreamingText = ref(false);
const chatBox = ref(null);

const isLightMode = ref(false);
const toggleTheme = () => {
  isLightMode.value = !isLightMode.value;
  window.dispatchEvent(new CustomEvent('theme-toggle', { detail: isLightMode.value }));
  // 动态切换网页底色，防止边缘露白
  document.documentElement.style.backgroundColor = isLightMode.value ? '#f0f2f5' : '#08080f';
};

const triggerFileInput = () => {
  if(!videoPreviewUrl.value) fileInput.value.click();
};

const handleDrop = (e) => {
  isDragging.value = false;
  const file = e.dataTransfer.files[0];
  if (file && file.type.includes('video')) {
    generateVideoPreview(file);
    uploadVideo(file);
  } else {
    alert("请拖拽有效的视频文件 (MP4/MOV)！");
  }
};

const handleFileSelect = (e) => {
  const file = e.target.files[0];
  if (file) {
    generateVideoPreview(file);
    uploadVideo(file);
  }
};

const generateVideoPreview = (file) => {
  if (videoPreviewUrl.value) URL.revokeObjectURL(videoPreviewUrl.value);
  videoPreviewUrl.value = URL.createObjectURL(file);
};

const uploadVideo = async (file) => {
  isUploading.value = true;
  taskId.value = null;
  finalResult.value = null;
  chatMessages.value = [];
  currentStatus.value = '正在将视频流推送到大脑...';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('http://127.0.0.1:8000/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    taskId.value = data.task_id;
    isUploading.value = false;
    startPolling(data.task_id);
  } catch (err) {
    currentStatus.value = '❌ 上传失败，请检查后端是否启动';
    isUploading.value = false;
  }
};

const startPolling = (id) => {
  if(pollingInterval) clearInterval(pollingInterval);
  pollingInterval = setInterval(async () => {
    try {
      const res = await fetch(`http://127.0.0.1:8000/status/${id}`);
      const data = await res.json();
      currentStatus.value = data.current_step;
      if (data.current_step.includes('✅') || data.current_step.includes('分析完成')) {
        clearInterval(pollingInterval);
        fetchFinalResult(id);
      } else if (data.current_step.includes('❌')) {
        clearInterval(pollingInterval);
      }
    } catch (err) {}
  }, 1000);
};

const fetchFinalResult = async (id) => {
  try {
    const res = await fetch(`http://127.0.0.1:8000/result/${id}`);
    finalResult.value = await res.json();
    streamTextResponse(`分析已完成！该视频呈现出 ${finalResult.value.label} 的情绪特征。你可以向我提问关于这段视频的细节。`);
  } catch (err) {
    currentStatus.value = '❌ 获取最终结果失败';
  }
};

const scrollToBottom = () => {
  nextTick(() => {
    if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight;
  });
};

const streamTextResponse = (fullText) => {
  return new Promise((resolve) => {
    isStreamingText.value = true;
    chatMessages.value.push({ role: 'ai', content: '' });
    const targetIndex = chatMessages.value.length - 1;
    let charIndex = 0;

    const typeNextChar = () => {
      if (charIndex < fullText.length) {
        chatMessages.value[targetIndex].content += fullText.charAt(charIndex);
        charIndex++;
        scrollToBottom();
        setTimeout(typeNextChar, 40); 
      } else {
        isStreamingText.value = false;
        resolve();
      }
    };
    typeNextChar(); 
  });
};

const sendMessage = async () => {
  if (!userInput.value.trim() || !finalResult.value || isAiTyping.value || isStreamingText.value) return;
  
  const msg = userInput.value;
  chatMessages.value.push({ role: 'user', content: msg });
  userInput.value = '';
  isAiTyping.value = true;
  scrollToBottom();

  try {
    const res = await fetch(`http://127.0.0.1:8000/chat/${taskId.value}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    });
    const data = await res.json();
    isAiTyping.value = false;
    await streamTextResponse(data.reply); 
  } catch (err) {
    isAiTyping.value = false;
    chatMessages.value.push({ role: 'ai', content: "网络波动，请重试。" });
  }
};
</script>

<style>
:root {
  background-color: #08080f; /* 强行垫底的暗色，防闪屏泛白 */
}
body, html, #app {
  margin: 0 !important;
  padding: 0 !important;
  width: 100% !important;
  max-width: none !important; /* 打破 Vite 默认的 1280px 宽度禁锢 */
  height: 100% !important;
  overflow-x: hidden;
}
</style>

<style scoped>
.app-container {
  --bg-gradient: radial-gradient(circle at center, #13131f 0%, #08080f 100%);
  --panel-bg: rgba(30, 30, 35, 0.25); 
  --text-main: #ffffff;
  --text-sub: #a0a0b0;
  --border-color: rgba(255, 255, 255, 0.12);
  --chat-ai-bg: rgba(0, 0, 0, 0.15); 
  --input-bg: rgba(0, 0, 0, 0.2);
  
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  width: 100vw; /* 强行铺满屏幕 */
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text-main);
  background: var(--bg-gradient);
  padding: 20px;
  box-sizing: border-box; /* 防止 padding 撑破容器 */
  transition: all 0.5s ease;
}

.app-container.light-theme {
  --bg-gradient: radial-gradient(circle at center, #f0f2f5 0%, #e4e7ed 100%);
  --panel-bg: rgba(255, 255, 255, 0.35); 
  --text-main: #1d1d1f;
  --text-sub: #86868b;
  --border-color: rgba(255, 255, 255, 0.6);
  --chat-ai-bg: rgba(255, 255, 255, 0.25);
  --input-bg: rgba(255, 255, 255, 0.5);
}

.theme-toggle { position: absolute; top: 30px; right: 30px; width: 44px; height: 44px; border-radius: 50%; border: 1px solid var(--border-color); background: var(--panel-bg); color: var(--text-main); cursor: pointer; z-index: 50; display: flex; justify-content: center; align-items: center; backdrop-filter: blur(10px); transition: all 0.3s; }
.theme-toggle:hover { transform: scale(1.1); }
.theme-toggle svg { width: 20px; height: 20px; }

.glass-panel {
  background: var(--panel-bg);
  backdrop-filter: blur(35px);
  -webkit-backdrop-filter: blur(35px);
  border: 1px solid var(--border-color);
  border-radius: 24px;
  padding: 30px;
  width: 100%;
  max-width: 950px;
  box-shadow: 0 24px 50px rgba(0, 0, 0, 0.3);
  position: relative;
  z-index: 10;
  transition: all 0.5s ease;
}

.title { font-size: 28px; font-weight: 700; margin-bottom: 25px; text-align: center; color: var(--text-main); }
.subtitle { font-size: 13px; font-weight: 400; color: var(--text-sub); margin-left: 10px; background: var(--border-color); padding: 4px 10px; border-radius: 12px; }
.content-wrapper { display: flex; gap: 25px; height: 480px; }
.left-section, .right-section { flex: 1; display: flex; flex-direction: column; gap: 20px; }

.upload-box {
  flex: 1; border: 2px dashed var(--border-color); border-radius: 16px;
  display: flex; justify-content: center; align-items: center; cursor: pointer;
  transition: all 0.3s; background: rgba(0, 0, 0, 0.02);
  position: relative; overflow: hidden;
}
.upload-box:hover, .upload-box.is-dragover { border-color: #007aff; background: rgba(0, 122, 255, 0.05); }
.upload-box.has-video { border: 1px solid var(--border-color); cursor: default; }
.video-player { width: 100%; height: 100%; object-fit: cover; z-index: 5; background: #000; }

.upload-content { text-align: center; color: var(--text-sub); }
.icon { width: 48px; height: 48px; margin-bottom: 10px; color: var(--text-sub); }
.spinner { width: 40px; height: 40px; border: 3px solid var(--border-color); border-top-color: #007aff; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 15px; }
@keyframes spin { to { transform: rotate(360deg); } }

.status-box { background: var(--chat-ai-bg); border-radius: 16px; padding: 20px; border: 1px solid var(--border-color); }
.box-title { font-size: 14px; color: var(--text-sub); margin-bottom: 10px; }
.animated-text { font-size: 15px; color: #007aff; font-weight: 500; margin-bottom: 15px; }
.result-card { background: var(--border-color); border-radius: 12px; padding: 15px; }
.score-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }
.positive { color: #34c759; font-weight: bold; }
.negative { color: #ff3b30; font-weight: bold; }

.chat-container { flex: 1; background: var(--chat-ai-bg); border-radius: 16px; display: flex; flex-direction: column; overflow: hidden; border: 1px solid var(--border-color); }
.chat-header { padding: 15px 20px; background: var(--border-color); display: flex; align-items: center; gap: 10px; border-bottom: 1px solid var(--border-color);}
.chat-header h3 { font-size: 15px; margin: 0; font-weight: 500; color: var(--text-main); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #ff3b30; }
.status-dot.active { background: #34c759; box-shadow: 0 0 8px #34c759; }

.chat-history { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
.empty-chat { margin: auto; color: var(--text-sub); font-size: 14px; }
.chat-bubble { max-width: 85%; display: flex; }
.chat-bubble.user { align-self: flex-end; }
.chat-bubble.user .bubble-content { background: #007aff; color: white; border-radius: 18px 18px 4px 18px; }
.chat-bubble.ai { align-self: flex-start; }
.chat-bubble.ai .bubble-content { background: var(--panel-bg); color: var(--text-main); border: 1px solid var(--border-color); border-radius: 18px 18px 18px 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }

.bubble-content { padding: 10px 16px; font-size: 14px; line-height: 1.6; text-align: left; word-wrap: break-word; }
.typing { padding: 12px 16px; background: var(--panel-bg); border-radius: 18px; display: flex; gap: 4px; align-self: flex-start; border: 1px solid var(--border-color); }
.dot { width: 6px; height: 6px; background: var(--text-sub); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }
@keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }

.chat-input-area { padding: 15px; background: var(--border-color); display: flex; gap: 10px; }
.chat-input-area input { flex: 1; background: var(--input-bg); border: 1px solid var(--border-color); border-radius: 20px; padding: 10px 15px; color: var(--text-main); outline: none; transition: all 0.3s; }
.chat-input-area input:focus { border-color: #007aff; }
.chat-input-area button { background: #007aff; color: white; border: none; border-radius: 20px; padding: 0 20px; cursor: pointer; font-weight: 500; transition: all 0.3s; }
.chat-input-area button:disabled { background: var(--border-color); color: var(--text-sub); cursor: not-allowed; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
.fade-enter-active, .fade-leave-active { transition: opacity 0.5s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>