<script setup lang="ts">
import { zhCN, dateZhCN, useOsTheme, darkTheme, lightTheme } from 'naive-ui';
import { ref, computed, watchEffect, provide } from 'vue';
import Sidebar from './components/Sidebar.vue';
import Login from './pages/Login.vue';

const showLogin = ref(true);

const handleLoginSuccess = () => {
  showLogin.value = false;
};
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
              <NLayout has-sider position="absolute">
                <Sidebar />
                <NLayout>
                  <Header />
                  <NLayoutContent>
                    <router-view />
                    <Login v-if="showLogin" @loginSuccess="handleLoginSuccess" />
                  </NLayoutContent>
                </NLayout>
              </NLayout>
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
  margin: 0;
  padding: 0;
}

:deep(.n-layout) {
  position: relative;
}

:deep(.n-layout-scroll-container) {
  position: relative;
}
</style>
