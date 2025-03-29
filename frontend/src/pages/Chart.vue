<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  BarChartOutline,
  PieChartOutline,
  TrendingUpOutline,
  CalendarOutline,
  FilterOutline
} from '@vicons/ionicons5'
import { 
  NCard, 
  NGrid, 
  NGridItem, 
  NDatePicker, 
  NSelect, 
  NIcon,
  NButton,
  NSpace
} from 'naive-ui'

// 时间范围选择
const dateRange = ref(null)

// 筛选选项
const filterOptions = [
  { label: '全部类型', value: 'all' },
  { label: '正常', value: 'normal' },
  { label: '异常', value: 'abnormal' }
]
const selectedFilter = ref('all')

// 统计数据
const statistics = ref({
  totalDiagnosis: 1234,
  normalCount: 986,
  abnormalCount: 248,
  accuracyRate: '92.86%'
})

// 诊断趋势数据（模拟）
const trendData = ref({
  labels: ['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
  values: [30, 45, 25, 60, 35, 25, 40]
})

// 异常类型分布（模拟）
const abnormalTypeData = ref([
  { type: '视网膜病变', value: 120 },
  { type: '青光眼', value: 68 },
  { type: '黄斑病变', value: 45 },
  { type: '其他', value: 15 }
])

// 准确率趋势（模拟）
const accuracyData = ref({
  labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
  values: [90, 91, 89, 92, 93, 92.86]
})

// 处理日期变化
const handleDateChange = (value: any) => {
  console.log('日期范围变更:', value)
  // 这里可以添加根据日期筛选数据的逻辑
}

// 处理筛选条件变化
const handleFilterChange = (value: string) => {
  console.log('筛选条件变更:', value)
  // 这里可以添加根据条件筛选数据的逻辑
}

onMounted(() => {
  // 这里可以添加初始化图表的逻辑
})
</script>

<template>
  <div class="p-6">
    <!-- 页面标题 -->
    <div class="mb-8">
      <h1 class="text-2xl font-bold flex items-center">
        <NIcon size="24" class="mr-2">
          <BarChartOutline />
        </NIcon>
        数据分析
      </h1>
      <p class="text-gray-500 mt-2">查看系统诊断数据的统计分析和趋势图表</p>
    </div>

    <!-- 筛选器 -->
    <NCard class="mb-6">
      <NSpace align="center">
        <NDatePicker
          v-model:value="dateRange"
          type="daterange"
          clearable
          :shortcuts="{ '最近一周': [Date.now() - 7 * 24 * 60 * 60 * 1000, Date.now()] }"
          @update:value="handleDateChange"
        />
        <NSelect
          v-model:value="selectedFilter"
          :options="filterOptions"
          style="width: 200px"
          @update:value="handleFilterChange"
        />
        <NButton type="primary">
          <template #icon>
            <NIcon><FilterOutline /></NIcon>
          </template>
          应用筛选
        </NButton>
      </NSpace>
    </NCard>

    <!-- 统计卡片 -->
    <NGrid :x-gap="16" :y-gap="16" :cols="4" class="mb-6">
      <NGridItem>
        <NCard>
          <div class="text-center">
            <div class="text-gray-500">总诊断数</div>
            <div class="text-2xl font-bold mt-2">{{ statistics.totalDiagnosis }}</div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="text-center">
            <div class="text-gray-500">正常样本</div>
            <div class="text-2xl font-bold mt-2 text-green-500">{{ statistics.normalCount }}</div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="text-center">
            <div class="text-gray-500">异常样本</div>
            <div class="text-2xl font-bold mt-2 text-red-500">{{ statistics.abnormalCount }}</div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="text-center">
            <div class="text-gray-500">准确率</div>
            <div class="text-2xl font-bold mt-2 text-blue-500">{{ statistics.accuracyRate }}</div>
          </div>
        </NCard>
      </NGridItem>
    </NGrid>

    <!-- 图表区域 -->
    <NGrid :x-gap="16" :y-gap="16" :cols="2" class="mb-6">
      <!-- 诊断趋势图 -->
      <NGridItem>
        <NCard title="诊断趋势">
          <div class="h-80">
            <!-- 这里可以使用 ECharts 或其他图表库 -->
            <div class="flex items-end h-64 space-x-2">
              <div
                v-for="(value, index) in trendData.values"
                :key="index"
                class="flex-1 bg-blue-500 hover:bg-blue-600 transition-all"
                :style="{ height: `${value}%` }"
              ></div>
            </div>
            <div class="flex justify-between mt-4">
              <span
                v-for="label in trendData.labels"
                :key="label"
                class="text-sm text-gray-500"
              >{{ label }}</span>
            </div>
          </div>
        </NCard>
      </NGridItem>

      <!-- 异常类型分布 -->
      <NGridItem>
        <NCard title="异常类型分布">
          <div class="h-80">
            <div class="space-y-4">
              <div
                v-for="item in abnormalTypeData"
                :key="item.type"
                class="flex items-center"
              >
                <span class="w-24 text-sm">{{ item.type }}</span>
                <div class="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    class="h-full bg-blue-500"
                    :style="{ width: `${(item.value / statistics.abnormalCount) * 100}%` }"
                  ></div>
                </div>
                <span class="ml-2 text-sm">{{ item.value }}</span>
              </div>
            </div>
          </div>
        </NCard>
      </NGridItem>
    </NGrid>

    <!-- 准确率趋势 -->
    <NCard title="准确率趋势" class="mb-6">
      <div class="h-64">
        <div class="flex items-end h-full space-x-2">
          <div
            v-for="(value, index) in accuracyData.values"
            :key="index"
            class="flex-1 bg-green-500 hover:bg-green-600 transition-all"
            :style="{ height: `${value}%` }"
          ></div>
        </div>
        <div class="flex justify-between mt-4">
          <span
            v-for="label in accuracyData.labels"
            :key="label"
            class="text-sm text-gray-500"
          >{{ label }}</span>
        </div>
      </div>
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