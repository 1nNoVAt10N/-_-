<script setup lang="ts">
import { ref, inject } from 'vue';
import { useRouter } from 'vue-router';
import {
  HomeOutline,
  SettingsOutline,
  MedicalOutline,
  BarChartOutline,
  BusinessOutline,
  MegaphoneOutline,
  ReceiptOutline,
  EyeOutline,
  ChevronBackOutline,
  ChevronForwardOutline,
  FolderOpenOutline
} from '@vicons/ionicons5';
import { NIcon } from 'naive-ui';
import { NLayoutSider } from 'naive-ui';

const router = useRouter();
const activeMenuIndex = ref(0); // 默认选中仪表盘
const isCollapsed = ref(false);
const isDarkMode = inject('isDarkMode');

const menuItems = [
  { id: 0, icon: HomeOutline, name: '仪表盘', route: '/dashboard' },
  { id: 1, icon: MedicalOutline, name: '诊断管理', route: '/diagnosis' },
  { id: 2, icon: FolderOpenOutline, name: '病历管理', route: '/medical-records' },
  { id: 3, icon: BarChartOutline, name: '数据图表', route: '/charts' },
  { id: 4, icon: BusinessOutline, name: '网络管理', route: '/network' },
  { id: 5, icon: MegaphoneOutline, name: '公共服务', route: '/services' },
  { id: 6, icon: ReceiptOutline, name: '收费评单', route: '/billing' },
  { id: 7, icon: SettingsOutline, name: '系统管理', route: '/system' }
];

const navigateTo = (item:any) => {
  activeMenuIndex.value = item.id;
  router.push(item.route);
};
</script>

<template>
  <NLayoutSider
    bordered
    collapse-mode="width"
    :collapsed-width="64"
    :width="200"
    show-trigger
    @collapse="isCollapsed = true"
    @expand="isCollapsed = false"
    class="sidebar"
  >
    <div class="logo-container">
      <div class="logo-icon">
        <NIcon size="20">
          <EyeOutline />
        </NIcon>
      </div>
      <div class="logo-text" v-show="!isCollapsed">IEDD</div>
    </div>
    
    <ul class="nav-menu">
      <li v-for="item in menuItems" :key="item.id" class="nav-item">
        <a @click.prevent="navigateTo(item)" href="#" 
           class="nav-link" 
           :class="{ active: activeMenuIndex === item.id }">
          <NIcon size="18">
            <component :is="item.icon" />
          </NIcon>
          <span v-show="!isCollapsed">{{ item.name }}</span>
        </a>
      </li>
    </ul>
  </NLayoutSider>
</template>

<style scoped>
.sidebar {
  height: 100vh !important;
  overflow: visible !important;
  background-color: var(--n-color);
}

.sidebar :deep(.n-layout-sider-scroll-container) {
  overflow: visible !important;
}

.sidebar :deep(.n-layout-sider-trigger) {
  width: 100% !important;
  height: 100% !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  z-index: 100 !important;
}

.sidebar :deep(.n-layout-sider-trigger-placeholder) {
  display: none !important;
}

.logo-container {
  display: flex;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid var(--n-border-color);
  height: 60px;
}

.logo-icon {
  width: 32px;
  height: 32px;
  margin-right: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--n-color-hover);
  border-radius: 50%;
  color: var(--n-text-color);
  flex-shrink: 0;
}

.logo-text {
  font-size: 16px;
  font-weight: 600;
  color: var(--n-text-color);
  white-space: nowrap;
  overflow: hidden;
}

.nav-menu {
  /* margin-top: 20px; */
  list-style-type: none;
  padding: 0;
  flex: 1;
  height: calc(100vh - 60px);
  overflow: hidden;
}

.nav-item {
  padding: 0 15px;
}

.nav-link {
  display: flex;
  align-items: center;
  padding: 12px 15px;
  color: var(--n-text-color-3);
  text-decoration: none;
  transition: all 0.3s;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
  overflow: hidden;
}

.nav-link:hover, .nav-link.active {
  color: var(--n-text-color);
  background-color: var(--n-color-hover);
}

.nav-link :deep(.n-icon) {
  margin-right: 10px;
  width: 20px;
  text-align: center;
  flex-shrink: 0;
}

.nav-link span {
  font-size: 14px;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 响应式设计 */
@media screen and (max-width: 768px) {
  .logo-text, .nav-link span {
    display: none;
  }
}
</style> 