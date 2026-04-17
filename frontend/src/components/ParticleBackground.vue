<template>
  <canvas ref="canvasRef" class="particle-canvas"></canvas>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';

const canvasRef = ref(null);
let ctx = null;
let particlesArray = [];
let animationFrameId = null;
let isLightMode = false; // 主题状态

const mouse = { x: null, y: null, radius: 120 };

// 监听 App.vue 发来的主题切换事件
window.addEventListener('theme-toggle', (e) => {
  isLightMode = e.detail;
});

class Particle {
  constructor(canvasWidth, canvasHeight) {
    this.x = Math.random() * canvasWidth;
    this.y = Math.random() * canvasHeight;
    this.size = Math.random() * 1.5 + 0.5; 
    this.density = (Math.random() * 30) + 1;
    this.velocityX = (Math.random() - 0.5) * 0.5;
    this.velocityY = (Math.random() - 0.5) * 0.5;
  }

  update(canvasWidth, canvasHeight) {
    this.x += this.velocityX;
    this.y += this.velocityY;

    if (this.x < 0 || this.x > canvasWidth) this.velocityX = -this.velocityX;
    if (this.y < 0 || this.y > canvasHeight) this.velocityY = -this.velocityY;

    if (mouse.x != null && mouse.y != null) {
      let dx = mouse.x - this.x;
      let dy = mouse.y - this.y;
      let distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance < mouse.radius) {
        const forceDirectionX = dx / distance;
        const forceDirectionY = dy / distance;
        const force = (mouse.radius - distance) / mouse.radius;
        const directionX = forceDirectionX * force * this.density * 0.6;
        const directionY = forceDirectionY * force * this.density * 0.6;
        
        this.x -= directionX;
        this.y -= directionY;
      }
    }
  }

  draw() {
    // 根据主题切换粒子颜色
    ctx.fillStyle = isLightMode ? 'rgba(0, 0, 0, 0.4)' : 'rgba(255, 255, 255, 0.6)';
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.closePath();
    ctx.fill();
  }
}

const initParticles = (width, height) => {
  particlesArray = [];
  const numberOfParticles = (width * height) / 10000;
  for (let i = 0; i < numberOfParticles; i++) {
    particlesArray.push(new Particle(width, height));
  }
};

const animate = () => {
  if (!canvasRef.value) return;
  const canvas = canvasRef.value;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < particlesArray.length; i++) {
    particlesArray[i].update(canvas.width, canvas.height);
    particlesArray[i].draw();
    
    for (let j = i; j < particlesArray.length; j++) {
      let dx = particlesArray[i].x - particlesArray[j].x;
      let dy = particlesArray[i].y - particlesArray[j].y;
      let distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance < 120) {
        ctx.beginPath();
        let opacity = 1 - (distance / 120);
        // 根据主题切换连线颜色
        ctx.strokeStyle = isLightMode 
          ? `rgba(0, 0, 0, ${opacity * 0.15})` 
          : `rgba(255, 255, 255, ${opacity * 0.2})`;
        ctx.lineWidth = 1;
        ctx.moveTo(particlesArray[i].x, particlesArray[i].y);
        ctx.lineTo(particlesArray[j].x, particlesArray[j].y);
        ctx.stroke();
      }
    }
  }
  animationFrameId = requestAnimationFrame(animate);
};

onMounted(() => {
  const canvas = canvasRef.value;
  ctx = canvas.getContext('2d');
  
  const resizeCanvas = () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    initParticles(canvas.width, canvas.height);
  };
  
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  window.addEventListener('mousemove', (event) => { mouse.x = event.x; mouse.y = event.y; });
  window.addEventListener('mouseout', () => { mouse.x = undefined; mouse.y = undefined; });

  animate();
});

onBeforeUnmount(() => {
  cancelAnimationFrame(animationFrameId);
});
</script>

<style scoped>
.particle-canvas {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 0;
  background: transparent; /* 背景颜色移交到主页面控制 */
  pointer-events: none;
}
</style>