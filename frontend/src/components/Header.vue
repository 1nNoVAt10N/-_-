<script setup lang="ts">
  import { ZoomQuestion } from '@vicons/tabler';
</script>

<template>
  <NLayoutHeader bordered>
    <div class="p-4 w-full">
      <div class="flex items-center w-full">
        <div class="text-3xl items-center flex">
          <n-icon>
            <ZoomQuestion />
          </n-icon>
        </div>
        <div class="ml-4 text-xl">
          <n-text> 预测 </n-text>
        </div>
        <div class="ml-auto">
          <DarkModeSwitchButton />
        </div>
      </div>
    </div>
  </NLayoutHeader>
</template>
