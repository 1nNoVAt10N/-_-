<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  SettingsOutline,
  ShieldCheckmarkOutline,
  KeyOutline,
  PeopleOutline,
  LogOutOutline,
  ServerOutline,
  CloudUploadOutline,
  CloudDownloadOutline,
  TimeOutline
} from '@vicons/ionicons5'
import { 
  NCard, 
  NGrid, 
  NGridItem, 
  NIcon,
  NButton,
  NSpace,
  NInput,
  NSelect,
  NForm,
  NFormItem,
  NInputNumber,
  NSwitch,
  useMessage
} from 'naive-ui'

// 系统信息
const systemInfo = ref({
  version: 'v1.0.0',
  uptime: '30天15小时',
  lastBackup: '2024-03-20 10:30',
  diskUsage: 75,
  memoryUsage: 60,
  cpuUsage: 45
})

// 系统设置
const systemSettings = ref({
  autoBackup: true,
  backupInterval: 24,
  maxLogDays: 30,
  maintenanceMode: false,
  emailNotification: true
})

// 保存设置
const message = useMessage()
const handleSaveSettings = () => {
  message.success('设置已保存')
}

// 系统状态
const systemStatus = ref({
  database: '正常',
  cache: '正常',
  storage: '正常',
  network: '正常'
})

// 备份选项
const backupOptions = [
  { label: '12小时', value: 12 },
  { label: '24小时', value: 24 },
  { label: '48小时', value: 48 },
  { label: '72小时', value: 72 }
]

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
          <SettingsOutline />
        </NIcon>
        系统管理
      </h1>
      <p class="text-gray-500 mt-2">配置系统参数和监控系统状态</p>
    </div>

    <!-- 系统状态 -->
    <NGrid :x-gap="16" :y-gap="16" :cols="4" class="mb-6">
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-blue-500">
              <ServerOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">系统版本</div>
              <div class="text-lg font-bold">{{ systemInfo.version }}</div>
              <div class="text-sm text-gray-400">运行时间: {{ systemInfo.uptime }}</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-green-500">
              <CloudUploadOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">最后备份</div>
              <div class="text-lg font-bold">{{ systemInfo.lastBackup }}</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-orange-500">
              <TimeOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">系统负载</div>
              <div class="text-lg font-bold">CPU: {{ systemInfo.cpuUsage }}%</div>
              <div class="text-sm text-gray-400">内存: {{ systemInfo.memoryUsage }}%</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-purple-500">
              <ShieldCheckmarkOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">安全状态</div>
              <div class="text-lg font-bold text-green-500">正常</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
    </NGrid>

    <!-- 系统设置 -->
    <NGrid :x-gap="16" :y-gap="16" :cols="2">
      <!-- 基本设置 -->
      <NGridItem>
        <NCard title="基本设置">
          <NForm label-placement="left" label-width="120">
            <NFormItem label="自动备份">
              <NSwitch v-model:value="systemSettings.autoBackup" />
            </NFormItem>
            <NFormItem label="备份间隔">
              <NSelect
                v-model:value="systemSettings.backupInterval"
                :options="backupOptions"
                style="width: 200px"
              />
            </NFormItem>
            <NFormItem label="日志保留天数">
              <NInputNumber
                v-model:value="systemSettings.maxLogDays"
                :min="7"
                :max="90"
                style="width: 200px"
              />
            </NFormItem>
            <NFormItem label="维护模式">
              <NSwitch v-model:value="systemSettings.maintenanceMode" />
            </NFormItem>
            <NFormItem label="邮件通知">
              <NSwitch v-model:value="systemSettings.emailNotification" />
            </NFormItem>
          </NForm>
          <div class="flex justify-end mt-4">
            <NButton type="primary" @click="handleSaveSettings">
              保存设置
            </NButton>
          </div>
        </NCard>
      </NGridItem>

      <!-- 系统状态 -->
      <NGridItem>
        <NCard title="系统状态">
          <div class="space-y-4">
            <div v-for="(status, key) in systemStatus" :key="key" class="flex items-center justify-between p-3 border rounded">
              <div class="flex items-center">
                <NIcon size="20" class="mr-2" :class="status === '正常' ? 'text-green-500' : 'text-red-500'">
                  <ShieldCheckmarkOutline />
                </NIcon>
                <div>
                  <div class="font-medium">{{ key }}</div>
                  <div class="text-sm text-gray-500">{{ status }}</div>
                </div>
              </div>
              <NButton size="small" type="primary">
                详情
              </NButton>
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