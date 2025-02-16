<script setup lang="ts">
  import { ReportMedical, ArrowRight, QuestionMark } from '@vicons/tabler';
  import { useRouter } from 'vue-router';
  import eyeImage from '@/assets/eye.png';
  import p1 from '@/assets/yuanli1.png';
  import p2 from '@/assets/yuanli2.png';
  import p3 from '@/assets/yuanli3.png';

  const router = useRouter();
  const startButton1 = () => {
    router.push('/singleRecognition');
  };
  const startButton2 = () => {
    router.push('/multiRecognition');
  };

  // 将 Abstract 内容定义为变量
  const abstractContent = `Abstract
Retinal disorders are a major cause of irreversible vision loss, which can be mitigated through accurate and early diagnosis. 
Conventionally, fundus images are used as the gold diagnosis standard in detecting retinal diseases. In recent years, more and 
more researchers have employed deep learning methods for diagnosing ophthalmic diseases using fundus photography datasets. 
Among the studies, most of them focus on diagnosing a single disease in fundus images, making it still challenging for the 
diagnosis of multiple diseases. In this paper, we propose a framework that combines ResNet and Transformer for multi-label 
classification of retinal disease. This model employs ResNet to extract image features, utilizes Transformer to capture global 
information, and enhances the relationships between categories through learnable label embedding. On the publicly available 
Ocular Disease Intelligent Recognition (ODIR-5 k) dataset, the proposed method achieves a mean average precision of 92.86%, 
an area under the curve (AUC) of 97.27%, and a recall of 90.62%, which outperforms other state-of-the-art approaches for the 
multi-label classification. The proposed method represents a significant advancement in the field of retinal disease diagnosis, 
offering a more accurate, efficient, and comprehensive model for the detection of multiple retinal conditions.`;
</script>

<template>
  <div class="flex flex-col p-2 mt-5 mx-10 space-y-20">
    <!-- 第一层 Box -->
    <div class="flex justify-between items-start">
      <!-- 左侧 Logo -->
      <div class="w-1/4">
        <n-icon class="text-[12rem]">
          <ReportMedical />
        </n-icon>
      </div>
      
      <!-- 中间内容 -->
      <div class="w-2/4 flex flex-col items-center">
        <n-text class="text-8xl font-bold mb-8">诊断系统</n-text>
        <n-text class="text-gray-500 text-lg mb-8 text-center">
          基于深度学习的眼底图像分析系统，为医疗诊断提供智能辅助决策支持
        </n-text>
        <div class="flex space-x-4">
          <n-button type="primary" size="large" @click="startButton1">
            <n-icon class="mr-2"><ArrowRight /></n-icon>
            开始单模预测
          </n-button>
          <n-button type="primary" size="large" @click="startButton2">
            <n-icon class="mr-2"><ArrowRight /></n-icon>
            开始多模预测
          </n-button>
        </div>
      </div>
      <!-- 右侧图片 -->
      <div class="w-1/4 flex justify-center">
        <n-image 
          :src="eyeImage" 
          width="200"
          class="object-contain"
        />
      </div>
    </div>

    <!-- 分隔线 -->
    <div class="w-full border-t border-gray-200"></div>

    <!-- 第二层 Box -->
    <div class="flex justify-between">
      <!-- 左侧三张纵向图片 -->
      <div class="w-1/2 flex flex-col space-y-4">
        
        <div>
          <n-image :src="p1" />
        </div>
        <!-- <div>
          <n-image :src="p2" />
        </div> -->
      </div>
      
      <!-- 右侧论文介绍 -->
      <div class="w-1/2 ml-8">
        <n-text class="text-2xl font-bold mb-4">研究背景</n-text>
        <n-text class="text-gray-600">{{ abstractContent }}</n-text>
      </div>
    </div>

    <!-- 分隔线 -->
    <div class="w-full border-t border-gray-200"></div>

    <!-- 第三层 Box -->
    <div class="flex justify-between">
      <!-- 左侧参考文献 -->
      <div class="w-1/3">
        <n-text class="text-xl font-bold mb-4">参考文献</n-text>
        <ul class="list-disc pl-5 space-y-2">
          <li>论文引用 1</li>
          <li>论文引用 2</li>
          <li>论文引用 3</li>
        </ul>
      </div>
      
      <!-- 中间团队成员 -->
      <div class="w-1/3 flex flex-col items-center">
        <n-text class="text-xl font-bold mb-4">团队成员</n-text>
        <div class="flex flex-col space-y-1">
          <div class="flex justify-center space-x-4">
            <div>陈非洋</div>
            <div>袁天博</div>
          </div>
          <div class="flex justify-center space-x-4">
            <div>翁笑阳</div>
            <div>高佳烨</div>
          </div>
          <div class="flex justify-start">
            <div>姜星宇</div>
          </div>
        </div>
      </div>
      
      <!-- 右侧学校信息 -->
      <div class="w-1/3 flex flex-col items-end">
        <n-text class="text-xl font-bold mb-4">苏州大学计算机科学与技术学院</n-text>
        <div class="text-right">
          <div>联系方式：0512-65113107</div>
          <div>联系地址：苏州市干将东路333号</div>
          <div>邮编：215008</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bg-gray-200 {
  background-color: #E5E7EB;
}
</style>
