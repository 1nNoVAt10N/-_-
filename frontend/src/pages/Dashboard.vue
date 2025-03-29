<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  EyeOutline,
  MedicalOutline,
  ServerOutline,
  BasketOutline,
  CashOutline,
  SyncOutline,
  PersonOutline,
  TimeOutline,
  CheckmarkCircleOutline,
  TrendingUpOutline,
  TrendingDownOutline,
  InformationCircleOutline,
  FilterOutline,
  BarChartOutline,
  ArchiveOutline
} from '@vicons/ionicons5'
import { NIcon, NCard, NGrid, NGridItem, NStatistic, NProgress, NButton } from 'naive-ui'

const router = useRouter()

// 数据加载状态
const loading = ref(true)

// 功能卡片数据
const functionCards = [
  {
    id: 1,
    icon: EyeOutline,
    title: '单眼诊断',
    description: '针对单张眼底图像的智能诊断分析',
    route: '/diagnosis',
    color: '#409EFF'
  },
  {
    id: 2,
    icon: MedicalOutline,
    title: '批量诊断',
    description: '针对多张眼底图像的智能诊断分析',
    route: '/multiRecognition',
    color: '#67C23A'
  },
  {
    id: 3,
    icon: BasketOutline,
    title: '诊断记录',
    description: '查看历史诊断记录和结果',
    route: '/diagnosis',
    color: '#E6A23C'
  },
  {
    id: 4,
    icon: BarChartOutline,
    title: '数据分析',
    description: '查看诊断数据统计和分析',
    route: '/charts',
    color: '#909399'
  }
]

// 系统状态数据
const systemStatus = ref([
  { id: 1, icon: ServerOutline, title: '服务器状态', value: '正常', status: 'success' },
  { id: 2, icon: BasketOutline, title: '数据库连接', value: '正常', status: 'success' },
  { id: 3, icon: CashOutline, title: 'AI引擎', value: '运行中', status: 'success' },
  { id: 4, icon: SyncOutline, title: '模型版本', value: 'v1.0.0', status: 'info' }
])

// 诊断统计数据
const diagnosisStats = ref([
  { id: 1, icon: MedicalOutline, title: '总诊断数', value: '1,234', increase: '12%', isPositive: true },
  { id: 2, icon: PersonOutline, title: '今日诊断', value: '45', increase: '8%', isPositive: true },
  { id: 3, icon: TimeOutline, title: '平均耗时', value: '2.5秒', increase: '15%', isPositive: false },
  { id: 4, icon: CheckmarkCircleOutline, title: '诊断准确率', value: '92.86%', increase: '0.5%', isPositive: true }
])

// 导航到特定页面

const navigateTo = (route: any) => {


  router.push(route)
}

// 组件挂载时加载数据
onMounted(() => {
  // 模拟数据加载
  setTimeout(() => {
    loading.value = false
  }, 1000)
})
</script>

<template>
  <div class="p-6">
    <!-- 页面标题 -->
    <div class="mb-8">
      <h1 class="text-2xl font-bold flex items-center">
        <NIcon size="24" class="mr-2">
          <TrendingUpOutline />
        </NIcon>
        系统概览
      </h1>
      <p class="text-gray-500 mt-2">基于深度学习的眼底图像分析系统，为医疗诊断提供智能辅助决策支持</p>
    </div>

    <!-- 功能卡片区域 -->
    <NGrid :x-gap="16" :y-gap="16" :cols="4" class="mb-8">
      <NGridItem v-for="card in functionCards" :key="card.id">
        <NCard hoverable @click="navigateTo(card.route)" class="cursor-pointer">
          <div class="flex items-center">
            <div class="w-12 h-12 rounded-full flex items-center justify-center mr-4" :style="{ backgroundColor: card.color + '20' }">
              <NIcon size="24" :style="{ color: card.color }">
                <component :is="card.icon" />
              </NIcon>
            </div>
            <div>
              <h3 class="text-lg font-medium">{{ card.title }}</h3>
              <p class="text-gray-500 text-sm">{{ card.description }}</p>
            </div>
          </div>
        </NCard>
      </NGridItem>
    </NGrid>

    <!-- 系统状态和统计信息 -->
    <NGrid :x-gap="16" :y-gap="16" :cols="2" class="mb-8">
      <!-- 系统状态 -->
      <NGridItem>
        <NCard title="系统状态">
          <div class="grid grid-cols-2 gap-4">
            <div v-for="item in systemStatus" :key="item.id" class="flex items-center">
              <NIcon size="20" class="mr-2">
                <component :is="item.icon" />
              </NIcon>
              <div>
                <div class="text-sm text-gray-500">{{ item.title }}</div>
                <div :class="['text-base font-medium', item.status === 'success' ? 'text-green-500' : 'text-blue-500']">
                  {{ item.value }}
                </div>
              </div>
            </div>
          </div>
        </NCard>
      </NGridItem>

      <!-- 诊断统计 -->
      <NGridItem>
        <NCard title="诊断统计">
          <div class="grid grid-cols-2 gap-4">
            <div v-for="stat in diagnosisStats" :key="stat.id" class="flex items-center">
              <NIcon size="20" class="mr-2">
                <component :is="stat.icon" />
              </NIcon>
              <div>
                <div class="text-sm text-gray-500">{{ stat.title }}</div>
                <div class="text-base font-medium">{{ stat.value }}</div>
                <div :class="['text-sm', stat.isPositive ? 'text-green-500' : 'text-red-500']">
                  <NIcon size="14" class="mr-1">
                    <component :is="stat.isPositive ? TrendingUpOutline : TrendingDownOutline" />
                  </NIcon>
                  {{ stat.increase }}
                </div>
              </div>
            </div>
          </div>
        </NCard>
      </NGridItem>
    </NGrid>

    <!-- 系统性能指标 -->
    <NCard title="系统性能" class="mb-8">
      <div class="grid grid-cols-4 gap-4">
        <NStatistic label="CPU使用率">
          <NProgress type="line" :percentage="45" :indicator-placement="'inside'" />
        </NStatistic>
        <NStatistic label="内存使用率">
          <NProgress type="line" :percentage="60" :indicator-placement="'inside'" />
        </NStatistic>
        <NStatistic label="GPU使用率">
          <NProgress type="line" :percentage="30" :indicator-placement="'inside'" />
        </NStatistic>
        <NStatistic label="磁盘使用率">
          <NProgress type="line" :percentage="75" :indicator-placement="'inside'" />
        </NStatistic>
      </div>
    </NCard>

    <div class="diagnosis-options">
      <NCard class="diagnosis-card">
        <template #header>
          <div class="card-header">
            <h3>眼底诊断</h3>
          </div>
        </template>
        <div class="diagnosis-buttons">
          <NButton
            type="primary"
            size="large"
            @click="router.push('/diagnosis')"
          >
            <template #icon>
              <NIcon><EyeOutline /></NIcon>
            </template>
            单眼诊断
          </NButton>
          <NButton
            type="info"
            size="large"
            @click="router.push('/multiRecognition')"
          >
            <template #icon>
              <NIcon><ArchiveOutline /></NIcon>
            </template>
            批量诊断
          </NButton>
        </div>
      </NCard>
    </div>
  </div>
</template>

<style scoped>
.n-card {
  transition: all 0.3s;
}

.n-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.diagnosis-options {
  margin-top: 20px;
}

.diagnosis-card {
  margin-bottom: 20px;
}

.diagnosis-buttons {
  display: flex;
  gap: 20px;
  justify-content: center;
  padding: 20px 0;
}

.diagnosis-buttons .n-button {
  min-width: 200px;
}
</style> 