<script setup lang="ts">
  import { ReportMedical, ArrowRight, QuestionMark } from '@vicons/tabler';
  import { useRouter } from 'vue-router';

  const router = useRouter();
  const startButton1 = () => {
    router.push('/singleRecognition');
  };
  const startButton2 = () => {
    router.push('/multiRecognition');
  };
</script>

<template>
  <div class="flex justify-center flex-col p-2 mt-5 mx-10">
    <div class="flex justify-center items-center flex-col">
      <n-icon class="text-9xl">
        <ReportMedical />
      </n-icon>
      <div class="flex items-center">
        <n-text class="text-3xl my-5">让我们开始预测</n-text>
      </div>
    </div>
    <div class="w-full flex items-center justify-end space-x-5">
      <n-button type="primary" size="large" class="mt-5 mr-5" @click="startButton1">
        <n-icon class="mr-3">
          <ArrowRight />
        </n-icon>
        <span>开始单模预测</span>
      </n-button>
      <n-button type="primary" size="large" class="mt-5 ml-5" @click="startButton2">
        <n-icon class="mr-3">
          <ArrowRight />
        </n-icon>
        <span>开始多模预测</span>
      </n-button>
    </div>
  </div>
</template>

<style scoped></style>
