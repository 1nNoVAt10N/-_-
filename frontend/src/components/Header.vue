<script setup lang="ts">
  import { Home, Stethoscope, ZoomQuestion } from '@vicons/tabler';
  import { RemoveRedEyeSharp } from '@vicons/material';
  import { ReminderMedical } from '@vicons/carbon';
  import { useRouter } from 'vue-router';
  import { useMessage } from 'naive-ui';
  import { NButton } from 'naive-ui';

  const message = useMessage();
  const router = useRouter();
  const startButton1 = () => {
    router.push('/diagnosis');
  };
  const startButton2 = () => {
    router.push('/multiRecognition');
  };
  const backButton = () => {
    router.push('/');
  };

  const options = [
    {
      label: '单模检测',
      key: 'single',
      style: 'min-width: 120px'
    },
    {
      label: '多模检测',
      key: 'multi',
      style: 'min-width: 120px'
    }
  ];

  const handleSelect = (key: string | number) => {
    switch (key) {
      case 'single':
        startButton1();
        break;
      case 'multi':
        startButton2();
        break;
      default:
        message.info(String(key));
    }
  };
</script>

<template>
  <NLayoutHeader bordered>
    <div class="p-4 w-full">
      <div class="flex items-center w-full">
        <div class="text-3xl items-center flex">
          <n-icon>
            <RemoveRedEyeSharp />
          </n-icon>
        </div>
        <div class="w-full ml-4 text-xl">
          <n-text> 智能眼疾医疗诊断平台 </n-text>
        </div>
        <div class="w-full flex justify-end space-x-5">
          <n-dropdown 
            trigger="hover" 
            :options="options" 
            @select="handleSelect"
            :style="{ minWidth: '120px' }"
          >
            <n-button type="primary" size="large" class="mt-5 ml-5">
              <n-icon class="mr-3">
                <ReminderMedical />
              </n-icon>
              <span>检测疾病</span>
            </n-button>
          </n-dropdown>
          <n-button type="primary" size="large" class="mt-5 ml-5 min-w-[120px]" @click="backButton">
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
