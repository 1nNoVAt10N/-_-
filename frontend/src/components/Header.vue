<script setup lang="ts">
  import { Home, Stethoscope, ZoomQuestion } from '@vicons/tabler';
  import { useRouter } from 'vue-router';

  const router = useRouter();

  const backButton = () => {
    router.push('/');
  };
</script>

<template>
  <NLayoutHeader bordered>
    <div class="p-4 w-full">
      <div class="flex items-center w-full">
        <div class="text-3xl items-center flex">
          <n-icon>
            <Stethoscope />
          </n-icon>
        </div>
        <div class="w-full ml-4 text-xl">
          <n-text> 眼球疾病预测 </n-text>
        </div>
        <div class="w-full flex justify-end space-x-5">
          <n-button type="primary" size="large" class="mt-5 ml-5" @click="backButton">
            <n-icon class="mr-3">
              <Home />
            </n-icon>
            <span>首页</span>
          </n-button>
          <DarkModeSwitchButton />
        </div>
      </div>
    </div>
  </NLayoutHeader>
</template>
