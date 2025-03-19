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
  const abstractContent = `
眼科疾病如今已成为紧追肿瘤和心脑血管疾病之后，对人类健康及生存质量构成严重威胁的第三大疾病类型。随着全球人口老龄化进程的加速，眼科疾病的患病人数呈现出显著的上升趋势。与此同时，在我国，医疗资源分布不均衡的状况较为突出。高水平医院集中了大量优质医疗资源，吸引众多患者前往就诊，这使得这些医院的眼科医生不得不花费大量时间和精力处理常见眼科疾病，工作任务愈发繁重。在此情况下，医师漏判、误诊的情况时有发生，诊断效率低下，且专业眼科人员短缺的问题日益凸显。仅依靠人类视觉系统去仔细观察眼底图像来精准发现病症，并维持较高的正确诊断率，在实际操作中难度极大。因此，开发一款高效、精准的眼科疾病智能诊断系统迫在眉睫，这将有效缓解当前医疗体系在眼科疾病诊断方面面临的诸多问题。
然而，眼科疾病的诊断面临着诸多挑战。一方面，眼科疾病种类繁多，不同疾病的症状和表现形式复杂多样，给准确诊断带来了困难。另一方面，眼底图像的解读需要专业的知识和经验，且图像质量受到多种因素的影响，进一步增加了诊断的难度。此外，传统的诊断方法主要依赖医生的主观判断，缺乏客观的量化指标，容易导致误诊和漏诊。`
;
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
<!-- 左侧参考文献 -->
      <div >
        <n-text class="text-xl font-bold mb-4">参考文献</n-text>
        <ul class="list-disc pl-5 space-y-2">
<li>[1]Lee, Jinhyuk et al. “BioBERT: a pre-trained biomedical language representation model for biomedical text mining.” Bioinformatics 36 (2019): 1234 - 1240. </li>
<li>[2]Dosovitskiy, Alexey et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.” ArXiv abs/2010.11929 (2020): n. pag.
</li>
<li>[3]He, Kaiming et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2015): 770-778.
</li>
<li>[4]R. Prawira, A. Bustamam and P. Anki, "Multi Label Classification Of Retinal Disease On Fundus Images Using AlexNet And VGG16 Architectures," 2021 4th International Seminar on Research of Information Technology and Intelligent Systems (ISRITI), Yogyakarta, Indonesia, 2021, pp. 464-468, doi: 10.1109/ISRITI54043.2021.9702817.
</li>
<li>[5]Li, Z., Xu, M., Yang, X., & Han, Y. (2022). Multi-Label Fundus Image Classification Using Attention Mechanisms and Feature Fusion. Micromachines, 13(6), 947. https://doi.org/10.3390/mi13060947
</li>
        </ul>
      </div>
    <!-- 分隔线 -->
    <div class="w-full border-t border-gray-200"></div>

    <!-- 第三层 Box -->
    <div class="flex justify-between">
      
      <!-- 中间团队成员 -->
<div class="w-1/2">
  <n-text class="text-xl font-bold mb-4">团队成员</n-text>
  <div class="text-left">
          <div>陈非洋 袁天博</div>
          <div>翁笑阳 高佳烨</div>
          <div>姜星宇</div>
        </div>
</div>
      
      <!-- 右侧学校信息 -->
      <div class="w-1/2 flex flex-col items-end">
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
