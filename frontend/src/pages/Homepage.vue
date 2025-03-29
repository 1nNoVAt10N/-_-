<script setup lang="ts">
import { ReportMedical, ArrowRight } from '@vicons/tabler';
import { useRouter } from 'vue-router';
// import { NAffix } from 'naive-ui';
import eyeImage from '@/assets/eye.png';
import p1 from '@/assets/yuanli1.png';
import p2 from '@/assets/yuanli2.png';
import p3 from '@/assets/yuanli3.png';
import backgroundImage from '@/assets/设计眼科系统封面.png';
import p31 from '@/assets/图片1.png';
import p32 from '@/assets/图片2.png';
import p33 from '@/assets/图片3.png';
import p23 from '@/assets/图片4.png';
import p11 from '@/assets/图片5.jpg';
import p13 from '@/assets/图片6.png';
import p12 from '@/assets/图片7.png';
import p22 from '@/assets/图片8.png';
import p21 from '@/assets/图片9.png';

// import { ref } from 'vue';
import { ref, computed, onMounted } from 'vue';
import { NAffix } from 'naive-ui';
// import { LogoGithub } from '@vicons/ionicons5'

// 从GitHub链接提取用户名
const getGithubAvatar = (githubUrl: string) => {
  const username = githubUrl.split('/').pop()
  return `https://github.com/${username}.png`
}
// 导航项配置
const navItems = ['项目背景', '模型优势', '系统涵盖功能', '项目人员'];

// DOM 引用
const contentRef = ref<HTMLElement | null>(null)!;
const activeIndex = ref(0);
const borderLeft = ref('0px');
const borderWidth = ref('0px');

// 边框样式计算
const borderStyle = computed(() => ({
  left: borderLeft.value,
  width: borderWidth.value,
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
}));

// 更新边框位置
const updateBorderPosition = (index: number) => {
  if (!contentRef.value) return;

  const items = contentRef.value.querySelectorAll('.module-nav-item');
  const activeItem = items[index] as HTMLElement;
  if (!activeItem) return;

  borderLeft.value = `${activeItem.offsetLeft}px`;
  borderWidth.value = `${activeItem.offsetWidth}px`;
};

// 导航点击处理
const handleNavClick = (index: number) => {
  activeIndex.value = index;
  updateBorderPosition(index);

  // 滚动到对应区块
  const targetId = `#section${index + 1}`;
  const targetEl = document.querySelector(targetId);
  targetEl?.scrollIntoView({ behavior: 'smooth' });
};

// 初始化位置
onMounted(() => {
  updateBorderPosition(0);
});
const router = useRouter();

const startButton1 = () => {
  router.push('/diagnosis');
};

const startButton2 = () => {
  router.push('/multiRecognition');
};

const containerRef = ref(<HTMLElement | undefined>undefined);

import { NAvatar, NButton, NTag, NIcon } from 'naive-ui'
import { LogoGithub } from '@vicons/ionicons5'

type TagType = 'default' | 'error' | 'success' | 'warning' | 'primary' | 'info';

const teamMembers: Array<{
  name: string;
  initials: string;
  role: string;
  github: string;
  major: string;
  color: string;
  tag: string;
  tagType: TagType;
}> = [
  {
    name: '姜星宇',
    initials: 'JXY',
    role: '前端页面设计',
    github: 'https://github.com/Reviahh',
    major: '软件工程',
    color: '#f59e0b',
    tag: '队员',
    tagType: 'warning'
  },
  {
    name: '陈非洋',
    initials: 'CFY',
    role: '算法设计实现与文档撰写',
    github: 'https://github.com/1nNoVAt10N',
    major: '软件工程',
    color: '#10b981',
    tag: '副队长',
    tagType: 'success'
  },
  {
    name: '翁笑阳',
    initials: 'WXY',
    role: '算法设计与文档撰写',
    github: 'https://github.com/7Spike35',
    major: '软件工程',
    color: '#4f46e5',
    tag: '队长',
    tagType: 'primary'
  },
  {
    name: '高佳烨',
    initials: 'MeGuRu',
    role: '前端页面设计',
    github: 'https://github.com/MeguruForever',
    major: '软件工程',
    color: '#ec4899',
    tag: '队员',
    tagType: 'error'
  },
  {
    name: '袁天博',
    initials: 'YTB',
    role: '算法设计与测试',
    github: 'https://github.com/icecreamforu',
    major: '计算机科学与技术',
    color: '#3b82f6',
    tag: '队员',
    tagType: 'info'
  }
];
</script>

<template>
  <div class="module-header" :style="{
    background: `
      linear-gradient(
        rgba(255, 255, 255, 0.3), /* 白色半透明叠加 */
        rgba(0, 0, 0, 0.5)        /* 黑色半透明叠加 */
      ),
      url(${backgroundImage}) 
      center/cover 
      no-repeat
      rgba(25, 25, 25, 0.8)       /* 备用背景色带透明度 */
    `,
    backdropFilter: 'blur(2px)' /* 可选背景模糊 */,
  }">
    <div class="module-header-container">
      <div class="module-header-content">
        <div class="module-header-title">诊断系统</div>
        <div class="module-header-info">基于深度学习的眼底图像分析系统，为医疗诊断提供智能辅助决策支持</div>

        <div class="module-header-link">
          <n-button type="primary" size="large" @click="startButton1">
            <n-icon class="mr-2">
              <ArrowRight />
            </n-icon>
            开始单模预测
          </n-button>
          <div class="module-header-empty"></div>
          <n-button type="primary" size="large" @click="startButton2">
            <n-icon class="mr-2">
              <ArrowRight />
            </n-icon>
            开始多模预测
          </n-button>
        </div>
      </div>
      <div class="module-header-empty"></div>
    </div>
  </div>
  
  <!-- <n-affix :trigger-top="0" :listen-to="() => contentRef" position="absolute"> -->

  <n-affix :trigger-top="0" position="absolute">
    <div class="module-nav-container">
      <ul class="module-nav-tab">
        <li v-for="(item, index) in navItems" :key="index" class="module-nav-item"
          :class="{ 'module-nav-item-active': activeIndex === index }" @click="handleNavClick(index)">
          {{ item }}
        </li>
      </ul>
      <i class="module-nav-border" :style="borderStyle"></i>
    </div>
  </n-affix>

  <!-- 第一部分 -->
<div id="section1" class="module-section">
  <n-card :bordered="false">
    <h2 class="section-title">项目背景</h2>
    <div class="background-grid">
      <!-- 每个背景项 -->
      <div class="background-item">
        <img :src="p11" class="background-image" alt="疾病威胁">
        <h3>眼科疾病威胁较大</h3>
        <p>全球约有22亿视力受损患者，致盲性眼病已成为重大公共卫生问题</p>
      </div>
      
      <div class="background-item">
        <img :src="p12" class="background-image" alt="诊断难点">
        <h3>眼科疾病诊断较难</h3>
        <p>传统诊断依赖医生经验，眼底病变早期症状不明显，容易漏诊误诊</p>
      </div>

      <div class="background-item">
        <img :src="p13" class="background-image" alt="政策支持">
        <h3>国家政策鼓励建设</h3>
        <p>"十四五"全国眼健康规划明确要求加强眼科医疗服务体系建设</p>
      </div>
    </div>
  </n-card>
</div>

  <!-- 第二部分 -->
<div id="section2" class="py-16">
  <n-card :bordered="false">
    <h2 class="section-title">模型优势</h2>
    
    <!-- 双列布局 -->
    <div class="max-w-6xl mx-auto grid md:grid-cols-2 gap-8 px-4">
      <!-- ResNet101 -->
      <!-- <div class="bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow">
        <img :src="p1" alt="ResNet101" class="w-full h-60 object-contain bg-gray-50 p-4 rounded-t-xl">
        <div class="p-6">
          <h3 class="text-lg font-semibold mb-4 text-gray-700">ResNet101 视觉模型</h3>
          <ul class="space-y-2 text-gray-600">
            <li class="flex items-center">
              <span class="text-green-500 mr-2">✔</span>
              101层深度残差结构
            </li>
            <li class="flex items-center">
              <span class="text-green-500 mr-2">✔</span>
              ImageNet预训练权重
            </li>
            <li class="flex items-center">
              <span class="text-green-500 mr-2">✔</span>
              病灶识别准确率98.2%
            </li>
          </ul>
        </div>
      </div> -->
      <div class="rounded-xl shadow-lg">
        <img :src="p21" alt="BioBERT" class="w-full h-60 object-contain bg-gray-50 p-4 rounded-t-xl">
        <div class="p-6">
          <h3 class="text-lg font-semibold mb-4">ResNet101 视觉模型</h3>
          <ul class="space-y-2">
            <li class="flex items-center">
              <span class="mr-2">✔</span>
              101层深度残差结构
            </li>
            <li class="flex items-center">
              <span class="mr-2">✔</span>
              ImageNet预训练权重
            </li>
            <li class="flex items-center">
              <span class="mr-2">✔</span>
              病灶识别准确率98.2%
            </li>
          </ul>
        </div>
      </div>

      <!-- BioBERT -->
      <div class="rounded-xl shadow-lg">
        <img :src="p22" alt="BioBERT" class="w-full h-60 object-contain bg-gray-50 p-4 rounded-t-xl">
        <div class="p-6">
          <h3 class="text-lg font-semibold mb-4">BioBERT 文本模型</h3>
          <ul class="space-y-2">
            <li class="flex items-center">
              <span class="mr-2">✔</span>
              生物医学领域预训练
            </li>
            <li class="flex items-center">
              <span class="mr-2">✔</span>
              临床文本理解准确率92%
            </li>
            <li class="flex items-center">
              <span class="mr-2">✔</span>
              症状关联分析模型
            </li>
          </ul>
        </div>
      </div>
    </div>

    <!-- 融合说明 -->
    <div class="max-w-4xl mx-auto mt-12 from-gray-50 to-gray-100 rounded-xl p-8 shadow-lg">
      <div class="flex flex-col md:flex-row gap-8 items-center">
        <img :src="p23" alt="双模态融合" class="md:w-1/2 rounded-lg">
        <div class="space-y-4">
          <h3 class="text-xl font-semibold">双模态融合架构</h3>
          <p class="leading-relaxed">
            通过跨模态注意力机制实现图像与文本特征对齐，融合准确率提升15.6%
          </p>
          <div class="flex flex-wrap gap-2">
            <span class="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">交叉注意力</span>
            <span class="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">特征对齐</span>
            <span class="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm">决策融合</span>
          </div>
        </div>
      </div>
    </div>
  </n-card>
</div>

<!-- 第三部分 -->
<div id="section3" class="py-16">
  <n-card :bordered="false">
    <h2 class="section-title">系统涵盖功能</h2>

    <div class="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6 px-4">
      <!-- 预处理 -->
      <div class="space-y-4">
        <div class="aspect-[2/1] bg-gray-50 rounded-lg overflow-hidden shadow-sm">
          <img :src="p31" class="w-full h-full object-cover" alt="预处理与拼接">
        </div>
        <div class="text-center space-y-2">
          <h3 class="text-lg font-semibold">图像预处理与拼接</h3>
          <p class="text-sm">解决图像亮度不均衡的问题并拼接左右眼图像</p>
        </div>
      </div>

      <!-- 增强 -->
      <div class="space-y-4">
        <div class="aspect-[2/1] bg-gray-50 rounded-lg overflow-hidden shadow-sm">
          <img :src="p32" class="w-full h-full object-cover" alt="图像增强">
        </div>
        <div class="text-center space-y-2">
          <h3 class="text-lg font-semibold">图像增强</h3>
          <p class="text-sm">对少数类别进行数据增强以提高类别平衡性</p>
        </div>
      </div>

      <!-- 分割 -->
      <div class="space-y-4">
        <div class="aspect-[2/1] bg-gray-50 rounded-lg overflow-hidden shadow-sm">
          <img :src="p33" class="w-full h-full object-cover" alt="图像分割">
        </div>
        <div class="text-center space-y-2">
          <h3 class="text-lg font-semibold">图像分割</h3>
          <p class="text-sm">分割出血管等病灶特征利于疾病识别</p>
        </div>
      </div>
    </div>
  </n-card>
</div>

  <!-- 第四部分 -->
<div id="section4" class="py-16 from-gray-50 to-blue-50">
    <n-card :bordered="false">
      <h2 class="section-title">
        项目团队
      </h2>

      <div class="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-5 gap-6 px-4">
        <div 
          v-for="(member, index) in teamMembers" 
          :key="index"
          class="group relative transition-all duration-300 hover:-translate-y-2"
        >
          <div class="absolute inset-0 bg-gradient-to-br from-blue-100 to-purple-100 rounded-xl blur opacity-0 group-hover:opacity-100 transition-opacity"></div>
          <n-card class="relative!">
            <div class="flex flex-col items-center space-y-4">
              <!-- 头像 -->
              <n-avatar
                round
                :size="120"
                class="ring-4 ring-white shadow-lg hover:ring-2 hover:ring-blue-400 transition-all"
                :src="getGithubAvatar(member.github)"
                :fallback-text="member.initials"
              />

              <!-- 基本信息 -->
              <div class="text-center">
                <h3 class="text-xl font-bold">{{ member.name }}</h3>
                <p class="text-sm mt-1">{{ member.role }}</p>
              </div>

              <!-- GitHub链接 -->
              <n-button
                tag="a"
                :href="member.github"
                target="_blank"
                type="primary"
                ghost
                class="!px-4"
              >
                <template #icon>
                  <n-icon :component="LogoGithub" />
                </template>
                GitHub
              </n-button>

              <!-- 个人介绍 -->
              <p class="text-center text-sm leading-relaxed">
                {{ member.major }}
              </p>

              <!-- 技术标签 -->
              <n-tag :type="member.tagType" size="small" round>
                {{ member.tag }}
              </n-tag>
            </div>
          </n-card>
        </div>
      </div>
    </n-card>
  </div>


</template>

<style scoped>
.module-header {
  height: 450px;
  color: #fff;
  background: #000;
}

.module-header-container {}

.module-header-container {
  position: relative;
  margin: 0 auto;
  width: 1180px;
  height: 100%;
  color: #fff;
  overflow: hidden;
}

.module-header-title {
  font-size: 48px;
}

.module-header-info {
  width: 630px;
  margin-top: 30px;
  font-size: 16px;
  line-height: 27px;
}

.module-header-link {
  display: block;
  margin-top: 30px;
  width: 122px;
  font-size: 14px;
  line-height: 36px;
  color: #fff;
  text-align: center;
  -webkit-box-shadow: none;
  -moz-box-shadow: none;
  box-shadow: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.module-header-content,
.module-header-empty {
  display: inline-block;
  vertical-align: middle;
}

.module-header-empty {
  height: 100%;
}

.module-nav {
  height: 60px;
  background: #fff;
  -webkit-box-shadow: 0 2px 10px 0 rgba(0, 0, 0, 0.1);
  -moz-box-shadow: 0 2px 10px 0 rgba(0, 0, 0, 0.1);
  box-shadow: 0 2px 10px 0 rgba(0, 0, 0, 0.1);
}

.module-nav-fixed {
  position: fixed;
  left: 0;
  right: 0;
  top: 0;
  z-index: 9;
}

.module-nav-container {
  position: relative;
  width: 1180px;
  margin: 0 auto;
}

.module-nav-tab {
  height: 60px;
  list-style: none;
  padding: 0;
  margin: 0;
  text-align: center;
  font-size: 0;
}

.module-nav-border {
  position: absolute;
  left: 0;
  bottom: 0;
  width: 120px;
  height: 3px;
  background: #0575eb;
  -webkit-transition: left 0.2s ease-in;
  -moz-transition: left 0.2s ease-in;
  transition: left 0.2s ease-in;
}

.module-nav-item {
  display: inline-block;
  position: relative;
  vertical-align: top;
  margin: 0 56px;
  width: 122px;
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
  font-size: 20px;
  line-height: 60px;
  white-space: nowrap;
  overflow: hidden;
  cursor: pointer;
}

.module-nav-item-active {
  color: #0575eb;
}

.absolute-anchor-container {
  width: 100%;
  height: 200px;
  position: relative;
}

.container {
  height: 200px;
  background-color: rgba(128, 128, 128, 0.3);
  border-radius: 3px;
  overflow: auto;
}

.padding {
  height: 150px;
  width: 100%;
  background-color: rgba(128, 128, 128, 0.15);
}

.content {
  height: 600px;
}

.section-title {
  text-align: center;
  font-size: 2.5rem;
  margin-bottom: 2rem;
}

.background-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  padding: 0 1rem;
  max-width: 1200px;
  margin: 0 auto;
}

.background-item {
  background: #ffffff;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease;
  text-align: center;
}

.background-item:hover {
  transform: translateY(-5px);
}

.background-item h3 {
  color: #2ba863;
  margin: 1rem 0;
  font-size: 1.25rem;
}

.background-item p {
  color: #666;
  font-size: 0.95rem;
  line-height: 1.5;
}

.background-image {
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 8px;
  margin-bottom: 1rem;
}

@media (max-width: 768px) {
  .background-grid {
    grid-template-columns: 1fr;
    max-width: 500px;
  }
  
  .section-title {
    font-size: 2rem;
  }
}

.gradient-text {
  background-image: linear-gradient(45deg, #4f46e5, #ec4899);
  -webkit-background-clip: text;
  background-clip: text;
}
</style>
