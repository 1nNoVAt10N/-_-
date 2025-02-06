<script setup lang="ts">
  import { zhCN, dateZhCN, useOsTheme, darkTheme, lightTheme } from 'naive-ui';

  const osThemeRef = useOsTheme();
  const isDarkMode = ref(osThemeRef.value === 'dark');

  const theme = computed(() => {
    return isDarkMode.value ? darkTheme : lightTheme;
  });

  watchEffect(() => {
    if (isDarkMode.value) {
      document.documentElement.style.setProperty(
        '--background-color',
        'var(--background-color-dark)',
      );
    } else {
      document.documentElement.style.setProperty(
        '--background-color',
        'var(--background-color-light)',
      );
    }
  });

  const switchDarkMode = () => {
    isDarkMode.value = !isDarkMode.value;
    document.documentElement.classList.toggle('dark', isDarkMode.value);
  };

  provide('isDarkMode', isDarkMode);
  provide('switchDarkMode', switchDarkMode);
</script>

<template>
  <NConfigProvider :locale="zhCN" :date-locale="dateZhCN" :theme="theme">
    <NLoadingBarProvider>
      <NDialogProvider>
        <NMessageProvider>
          <NNotificationProvider>
            <NLayout content-style="min-height: 100vh">
              <Header />
              <NLayoutContent>
                <router-view />
              </NLayoutContent>
            </NLayout>
          </NNotificationProvider>
        </NMessageProvider>
      </NDialogProvider>
    </NLoadingBarProvider>
  </NConfigProvider>
</template>

<style scoped>
  body {
    background-color: var(--background-color);
    transition: background-color 0.3s ease;
  }
</style>
