<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  ServerOutline,
  CloudUploadOutline,
  CloudDownloadOutline,
  PulseOutline,
  TimeOutline,
  CheckmarkCircleOutline,
  WarningOutline,
  StopCircleOutline,
  PlayCircleOutline,
  PauseCircleOutline
} from '@vicons/ionicons5'
import { 
  NCard, 
  NGrid, 
  NGridItem, 
  NIcon,
  NButton,
  NSpace,
  NProgress,
  NTag,
  NBadge,
  NTable,
  NDataTable,
  NInput,
  NSelect
} from 'naive-ui'

// 服务状态统计
const serviceStats = ref({
  total: 12,
  running: 8,
  stopped: 3,
  error: 1
})

// 服务列表
const services = ref([
  {
    id: 1,
    name: 'Web服务器',
    type: 'web',
    status: 'running',
    uptime: '23天15小时',
    cpu: 45,
    memory: 60,
    port: 8080
  },
  {
    id: 2,
    name: '数据库服务',
    type: 'database',
    status: 'running',
    uptime: '30天',
    cpu: 35,
    memory: 75,
    port: 3306
  },
  {
    id: 3,
    name: '缓存服务',
    type: 'cache',
    status: 'stopped',
    uptime: '0',
    cpu: 0,
    memory: 0,
    port: 6379
  },
  {
    id: 4,
    name: '消息队列',
    type: 'queue',
    status: 'error',
    uptime: '2小时',
    cpu: 85,
    memory: 90,
    port: 5672
  }
])

// 服务类型选项
const serviceTypes = [
  { label: '全部', value: 'all' },
  { label: 'Web服务', value: 'web' },
  { label: '数据库', value: 'database' },
  { label: '缓存', value: 'cache' },
  { label: '消息队列', value: 'queue' }
]

// 状态选项
const statusOptions = [
  { label: '全部', value: 'all' },
  { label: '运行中', value: 'running' },
  { label: '已停止', value: 'stopped' },
  { label: '错误', value: 'error' }
]

// 搜索和筛选
const searchText = ref('')
const selectedType = ref('all')
const selectedStatus = ref('all')

// 表格列定义
const columns = [
  { title: '服务名称', key: 'name' },
  { title: '类型', key: 'type' },
  { title: '状态', key: 'status', render: (row: { status: 'running' | 'stopped' | 'error' }) => {
    const statusMap = {
      running: { type: 'success' as const, text: '运行中' },
      stopped: { type: 'warning' as const, text: '已停止' },
      error: { type: 'error' as const, text: '错误' }
    }
    const status = statusMap[row.status]
    return h(NTag, { type: status.type }, { default: () => status.text })
  }},
  { title: '运行时间', key: 'uptime' },
  { title: 'CPU使用率', key: 'cpu', render: (row: any) => {
    return h(NProgress, {
      type: 'line',
      percentage: row.cpu,
      height: 12,
      indicatorPlacement: 'inside'
    })
  }},
  { title: '内存使用率', key: 'memory', render: (row: any) => {
    return h(NProgress, {
      type: 'line',
      percentage: row.memory,
      height: 12,
      indicatorPlacement: 'inside'
    })
  }},
  { title: '端口', key: 'port' },
  { title: '操作', key: 'actions', render: (row: any) => {
    return h(NSpace, null, {
      default: () => [
        h(NButton, {
          size: 'small',
          type: row.status === 'running' ? 'warning' : 'success',
          onClick: () => handleServiceAction(row)
        }, {
          default: () => row.status === 'running' ? '停止' : '启动'
        })
      ]
    })
  }}
]

// 处理服务操作
const handleServiceAction = (service: any) => {
  console.log('操作服务:', service.name, service.status === 'running' ? '停止' : '启动')
}

onMounted(() => {
  // 初始化逻辑
})
</script>

<template>
  <div class="p-6">
    <!-- 页面标题 -->
    <div class="mb-8">
      <h1 class="text-2xl font-bold flex items-center">
        <NIcon size="24" class="mr-2">
          <ServerOutline />
        </NIcon>
        服务管理
      </h1>
      <p class="text-gray-500 mt-2">监控和管理系统各项服务的运行状态</p>
    </div>

    <!-- 服务状态概览 -->
    <NGrid :x-gap="16" :y-gap="16" :cols="4" class="mb-6">
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-blue-500">
              <ServerOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">总服务数</div>
              <div class="text-lg font-bold">{{ serviceStats.total }}</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-green-500">
              <PlayCircleOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">运行中</div>
              <div class="text-lg font-bold text-green-500">{{ serviceStats.running }}</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-orange-500">
              <PauseCircleOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">已停止</div>
              <div class="text-lg font-bold text-orange-500">{{ serviceStats.stopped }}</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-red-500">
              <StopCircleOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">错误</div>
              <div class="text-lg font-bold text-red-500">{{ serviceStats.error }}</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
    </NGrid>

    <!-- 搜索和筛选 -->
    <NCard class="mb-6">
      <NSpace align="center">
        <NInput
          v-model:value="searchText"
          placeholder="搜索服务名称"
          style="width: 200px"
        />
        <NSelect
          v-model:value="selectedType"
          :options="serviceTypes"
          style="width: 150px"
        />
        <NSelect
          v-model:value="selectedStatus"
          :options="statusOptions"
          style="width: 150px"
        />
        <NButton type="primary">
          刷新
        </NButton>
      </NSpace>
    </NCard>

    <!-- 服务列表 -->
    <NCard>
      <NDataTable
        :columns="columns"
        :data="services"
        :pagination="{ pageSize: 10 }"
        :bordered="false"
        :single-line="false"
      />
    </NCard>
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
</style>