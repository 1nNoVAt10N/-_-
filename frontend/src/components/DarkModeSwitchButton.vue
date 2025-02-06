<script setup lang="ts">
  import { inject } from 'vue';
  import type { Ref } from 'vue';
  import { Sun, Moon } from '@vicons/tabler';
  import { NButton, NIcon } from 'naive-ui';

  const isDarkMode = inject('isDarkMode') as Ref<boolean>;
  const switchDarkMode = inject('switchDarkMode') as () => void;
</script>

<template>
  <NButton tertiary circle @click="switchDarkMode">
    <template #icon>
      <NIcon v-if="isDarkMode">
        <Sun />
      </NIcon>
      <NIcon v-else>
        <Moon />
      </NIcon>
    </template>
  </NButton>
</template>
