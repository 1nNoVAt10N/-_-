<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  GitNetworkOutline,
  SpeedometerOutline,
  ServerOutline,
  WifiOutline,
  TimeOutline,
  AlertCircleOutline,
  SwapHorizontalOutline,
  CloudUploadOutline,
  CloudDownloadOutline,
  PulseOutline
} from '@vicons/ionicons5'
import { 
  NCard, 
  NGrid, 
  NGridItem, 
  NDatePicker, 
  NSelect, 
  NIcon,
  NButton,
  NSpace,
  NProgress,
  NTag,
  NBadge
} from 'naive-ui'

// 网络状态
const networkStatus = ref({
  status: 'online',
  uptime: '23天15小时',
  latency: '32ms',
  packetLoss: '0.1%'
})

// 带宽使用统计
const bandwidthStats = ref({
  upload: '125 Mbps',
  download: '256 Mbps',
  totalTransferred: '1.2 TB',
  activeConnections: 256
})

// 网络设备状态
const devices = ref([
  { 
    id: 1, 
    name: '主路由器',
    ip: '192.168.1.1',
    status: 'online',
    load: 45,
    type: 'router'
  },
  { 
    id: 2, 
    name: '交换机-01',
    ip: '192.168.1.2',
    status: 'online',
    load: 32,
    type: 'switch'
  },
  { 
    id: 3, 
    name: '防火墙',
    ip: '192.168.1.3',
    status: 'online',
    load: 28,
    type: 'firewall'
  },
  { 
    id: 4, 
    name: '备用路由器',
    ip: '192.168.1.4',
    status: 'standby',
    load: 5,
    type: 'router'
  }
])

// 流量数据（模拟）
const trafficData = ref({
  labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '现在'],
  upload: [20, 15, 45, 75, 60, 45, 55],
  download: [30, 25, 65, 125, 90, 75, 85]
})

// 告警信息
const alerts = ref([
  {
    id: 1,
    type: 'warning' as const,
    device: '交换机-01',
    message: '带宽使用率超过80%',
    time: '10分钟前'
  },
  {
    id: 2,
    type: 'info' as const,
    device: '防火墙',
    message: '检测到异常连接尝试',
    time: '25分钟前'
  },
  {
    id: 3,
    type: 'error' as const,
    device: '备用路由器',
    message: '设备离线',
    time: '1小时前'
  }
])

// 时间范围选择
const dateRange = ref(null)

// 筛选选项
const filterOptions = [
  { label: '所有设备', value: 'all' },
  { label: '路由器', value: 'router' },
  { label: '交换机', value: 'switch' },
  { label: '防火墙', value: 'firewall' }
]
const selectedFilter = ref('all')

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
          <NetworkOutline />
        </NIcon>
        网络管理
      </h1>
      <p class="text-gray-500 mt-2">监控和管理网络设备、流量及性能</p>
    </div>

    <!-- 网络状态概览 -->
    <NGrid :x-gap="16" :y-gap="16" :cols="4" class="mb-6">
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-green-500">
              <PulseOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">网络状态</div>
              <div class="text-lg font-bold text-green-500">正常运行中</div>
              <div class="text-sm text-gray-400">运行时间: {{ networkStatus.uptime }}</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-blue-500">
              <SpeedometerOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">网络延迟</div>
              <div class="text-lg font-bold">{{ networkStatus.latency }}</div>
              <div class="text-sm text-gray-400">丢包率: {{ networkStatus.packetLoss }}</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-purple-500">
              <SwapHorizontalOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">当前带宽使用</div>
              <div class="text-lg font-bold">
                ↑ {{ bandwidthStats.upload }} / ↓ {{ bandwidthStats.download }}
              </div>
              <div class="text-sm text-gray-400">
                活动连接: {{ bandwidthStats.activeConnections }}
              </div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-orange-500">
              <ServerOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">总传输量</div>
              <div class="text-lg font-bold">{{ bandwidthStats.totalTransferred }}</div>
              <div class="text-sm text-gray-400">今日累计</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
    </NGrid>

    <!-- 筛选器 -->
    <NCard class="mb-6">
      <NSpace align="center">
        <NDatePicker
          v-model:value="dateRange"
          type="daterange"
          clearable
          :shortcuts="{ '最近24小时': [Date.now() - 24 * 60 * 60 * 1000, Date.now()] }"
        />
        <NSelect
          v-model:value="selectedFilter"
          :options="filterOptions"
          style="width: 200px"
        />
        <NButton type="primary">
          刷新数据
        </NButton>
      </NSpace>
    </NCard>

    <!-- 流量图表 -->
    <NCard title="网络流量监控" class="mb-6">
      <div class="h-80">
        <div class="flex items-end h-64 space-x-2">
          <div v-for="(value, index) in trafficData.labels" :key="index" class="flex-1">
            <div class="flex flex-col space-y-1">
              <div
                class="bg-green-500 hover:bg-green-600 transition-all"
                :style="{ height: `${trafficData.upload[index]}px` }"
              ></div>
              <div
                class="bg-blue-500 hover:bg-blue-600 transition-all"
                :style="{ height: `${trafficData.download[index]}px` }"
              ></div>
            </div>
          </div>
        </div>
        <div class="flex justify-between mt-4">
          <span
            v-for="label in trafficData.labels"
            :key="label"
            class="text-sm text-gray-500"
          >{{ label }}</span>
        </div>
      </div>
      <div class="flex justify-center gap-4 mt-4">
        <div class="flex items-center">
          <div class="w-4 h-4 bg-green-500 rounded mr-2"></div>
          <span>上传流量</span>
        </div>
        <div class="flex items-center">
          <div class="w-4 h-4 bg-blue-500 rounded mr-2"></div>
          <span>下载流量</span>
        </div>
      </div>
    </NCard>

    <!-- 设备状态和告警 -->
    <NGrid :x-gap="16" :y-gap="16" :cols="2">
      <!-- 设备状态 -->
      <NGridItem>
        <NCard title="设备状态">
          <div class="space-y-4">
            <div v-for="device in devices" :key="device.id" class="flex items-center justify-between p-3 border rounded">
              <div class="flex items-center">
                <NBadge :dot="true" :type="device.status === 'online' ? 'success' : 'warning'" />
                <div class="ml-3">
                  <div class="font-medium">{{ device.name }}</div>
                  <div class="text-sm text-gray-500">{{ device.ip }}</div>
                </div>
              </div>
              <div class="text-right">
                <NProgress
                  type="line"
                  :percentage="device.load"
                  :indicator-placement="'inside'"
                  :height="12"
                  :border-radius="4"
                />
                <div class="text-sm text-gray-500">负载: {{ device.load }}%</div>
              </div>
            </div>
          </div>
        </NCard>
      </NGridItem>

      <!-- 告警信息 -->
      <NGridItem>
        <NCard title="告警信息">
          <div class="space-y-4">
            <div v-for="alert in alerts" :key="alert.id" class="flex items-center justify-between p-3 border rounded">
              <div class="flex items-center">
                <NTag :type="alert.type">{{ alert.type.toUpperCase() }}</NTag>
                <div class="ml-3">
                  <div class="font-medium">{{ alert.device }}</div>
                  <div class="text-sm text-gray-500">{{ alert.message }}</div>
                </div>
              </div>
              <div class="text-sm text-gray-500">
                {{ alert.time }}
              </div>
            </div>
          </div>
        </NCard>
      </NGridItem>
    </NGrid>
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