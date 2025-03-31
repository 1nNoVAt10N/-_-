<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
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
  NSpace,
  NSpin
} from 'naive-ui'
import axios from 'axios'
import * as echarts from 'echarts'

// 声明ECharts类型
type EChartsInstance = echarts.ECharts

// 加载状态
const loading = ref(true)

// 时间范围选择
const dateRange = ref(null)

// 筛选选项
const filterOptions = [
  { label: '正常', value: 'N' },
  { label: '糖尿病', value: 'D' },
  { label: '青光眼', value: 'G' },
  { label: '白内障', value: 'C' },
  { label: 'AMD', value: 'A' },
  { label: '高血压', value: 'H' },
  { label: '近视', value: 'M' },
  { label: '其他疾病/异常', value: 'O' },
]
const selectedFilter = ref('N')

// 统计数据
const statistics = ref({
  total_diagnoses: 0,
  normal_samples: 0,
  abnormal_samples: 0,
  accuracy: "0%",
})

// 年龄饼图引用
const ageChartRef = ref<HTMLElement | null>(null)
let ageChart: EChartsInstance | null = null

// 性别饼图引用
const genderChartRef = ref<HTMLElement | null>(null)
let genderChart: EChartsInstance | null = null

// 修改类型定义为数字
class age_pct {
  '0-18': number
  '19-30': number
  '31-45': number
  '46-60': number
  '61-75': number
  '76+': number
}

class gender_pct {
  "Female": number
  "Male": number
}

// 初始化为数字类型
const age = ref<age_pct>({
  '0-18': 0,
  '19-30': 0,
  '31-45': 0,
  '46-60': 0,
  '61-75': 0,
  '76+': 0
})

const gender = ref<gender_pct>({
  "Female": 0,
  "Male": 0
})  

// 格式化百分比，保留两位小数
const formatPercent = (value: number): string => {
  return value.toFixed(2) + '%'
}

// 处理日期变化
const handleDateChange = (value: any) => {
  console.log('日期范围变更:', value)
  // 这里可以添加根据日期筛选数据的逻辑
  refreshData()
}

// 处理筛选条件变化
const handleFilterChange = (value: string) => {
  console.log('筛选条件变更:', value)
}

// 初始化年龄饼图
const initAgeChart = () => {
  if (ageChartRef.value) {
    if (ageChart) {
      ageChart.dispose()
    }
    ageChart = echarts.init(ageChartRef.value)
    
    // 颜色设置
    const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272']
    
    const ageData = Object.entries(age.value).map(([key, value], index) => {
      return {
        name: key,
        value: Number(value.toFixed(2)),
        itemStyle: {
          color: colors[index % colors.length]
        }
      }
    })
    
    const option = {
      color: colors,
      title: {
        text: '年龄分布',
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c}%'
      },
      legend: {
        orient: 'vertical',
        right: 10,
        top: 'middle',
        data: Object.keys(age.value),
        formatter: (name: string) => {
          const value = age.value[name as keyof age_pct]
          return `${name}: ${value.toFixed(2)}%`
        }
      },
      series: [
        {
          name: '年龄分布',
          type: 'pie',
          radius: ['40%', '70%'],
          center: ['40%', '50%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: {
            show: true,
            formatter: '{b}: {d}%'
          },
          emphasis: {
            label: {
              show: true,
              fontSize: '18',
              fontWeight: 'bold'
            }
          },
          labelLine: {
            show: true
          },
          data: ageData
        }
      ]
    }
    
    ageChart.setOption(option)
  }
}

// 初始化性别饼图
const initGenderChart = () => {
  if (genderChartRef.value) {
    if (genderChart) {
      genderChart.dispose()
    }
    genderChart = echarts.init(genderChartRef.value)
    
    // 性别饼图颜色
    const genderColors = ['#ff7e88', '#5387cc']
    
    const genderData = Object.entries(gender.value).map(([key, value], index) => {
      return {
        name: key,
        value: Number(value.toFixed(2)),
        itemStyle: {
          color: genderColors[index % genderColors.length]
        }
      }
    })
    
    const option = {
      color: genderColors,
      title: {
        text: '性别分布',
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c}%'
      },
      legend: {
        orient: 'vertical',
        right: 10,
        top: 'middle',
        data: Object.keys(gender.value),
        formatter: (name: string) => {
          const value = gender.value[name as keyof gender_pct]
          return `${name}: ${value.toFixed(2)}%`
        }
      },
      series: [
        {
          name: '性别分布',
          type: 'pie',
          radius: ['40%', '70%'],
          center: ['40%', '50%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: {
            show: true,
            formatter: '{b}: {d}%'
          },
          emphasis: {
            label: {
              show: true,
              fontSize: '18',
              fontWeight: 'bold'
            }
          },
          labelLine: {
            show: true
          },
          data: genderData
        }
      ]
    }
    
    genderChart.setOption(option)
  }
}

// 更新图表
const updateCharts = () => {
  // 初始化或更新图表
  initAgeChart()
  initGenderChart()
}

// 刷新数据方法
const refreshData = () => {
  loading.value = true
  
  // 模拟API请求延迟
  setTimeout(() => {
    // 在实际应用中，这里应该是真实的数据请求
    loading.value = false
  }, 1000)
}

// 应用筛选按钮
const applyFilter = () => {
  loading.value = true

  axios.post('http://127.0.0.1:5000/get_plot_gender',{flag:selectedFilter.value}).then(response => {
    const data = response.data
    console.log('plot data:', data);

    // 确保从后端获取的数据被正确赋值
    if (data && data.age_pct) {
      age.value = data.age_pct
    }
    
    if (data && data.gender_pct) {
      gender.value = data.gender_pct
    }
    
    // 调试日志，检查数据是否正确
    console.log('年龄数据:', age.value);
    console.log('性别数据:', gender.value);
    
    // 更新图表
    updateCharts()
    
  }).catch(error => {
    console.error('获取数据失败:', error)
  }).finally(() => {
    loading.value = false
  })
}

// 窗口大小变化时重绘图表
const handleResize = () => {
  if (ageChart) {
    ageChart.resize()
  }
  if (genderChart) {
    genderChart.resize()
  }
}

onMounted(() => {
  // 初始化数据加载
  loading.value = true
  
  // 请求图表数据
  axios.post('http://127.0.0.1:5000/get_plot').then(response => {
    const data = response.data
    console.log('plot data:', data);
    statistics.value = {
      total_diagnoses: data.total_diagnoses,
      normal_samples: data.normal_samples,
      abnormal_samples: data.abnormal_samples,
      accuracy: "95.46%"
    }
    
    // 如果初始数据中有年龄和性别数据，也进行更新
    if (data.age_pct) {
      age.value = data.age_pct
    }
    if (data.gender_pct) {
      gender.value = data.gender_pct
    }
    
    // 初始调用get_plot_gender获取数据
    applyFilter();
    
  }).catch(error => {
    console.error('获取数据失败:', error)
  }).finally(() => {
    loading.value = false
  })
  
  // 添加窗口大小变化的监听器
  window.addEventListener('resize', handleResize)
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

    <!-- 包裹整个内容的加载状态 -->
    <NSpin :show="loading" description="加载中...">
      <div class="charts-container">
        <!-- 统计卡片 -->
        <NGrid :x-gap="16" :y-gap="16" :cols="4" class="mb-6">
          <NGridItem>
            <NCard>
              <div class="text-center">
                <div class="text-gray-500">总诊断数</div>
                <div class="text-2xl font-bold mt-2">{{ statistics.total_diagnoses }}</div>
              </div>
            </NCard>
          </NGridItem>
          <NGridItem>
            <NCard>
              <div class="text-center">
                <div class="text-gray-500">正常样本</div>
                <div class="text-2xl font-bold mt-2 text-green-500">{{ statistics.normal_samples}}</div>
              </div>
            </NCard>
          </NGridItem>
          <NGridItem>
            <NCard>
              <div class="text-center">
                <div class="text-gray-500">异常样本</div>
                <div class="text-2xl font-bold mt-2 text-red-500">{{ statistics.abnormal_samples }}</div>
              </div>
            </NCard>
          </NGridItem>
          <NGridItem>
            <NCard>
              <div class="text-center">
                <div class="text-gray-500">准确率</div>
                <div class="text-2xl font-bold mt-2 text-blue-500">{{ statistics.accuracy }}</div>
              </div>
            </NCard>
          </NGridItem>
        </NGrid>
        
        <!-- 筛选器 -->
        <NCard class="mb-6">
          <NSpace align="center">
            <NSelect
              v-model:value="selectedFilter"
              :options="filterOptions"
              style="width: 200px"
              @update:value="handleFilterChange"
            />
            <NButton type="primary" @click="applyFilter">
              <template #icon>
                <NIcon><FilterOutline /></NIcon>
              </template>
              选择疾病
            </NButton>
          </NSpace>
        </NCard>
        
        <!-- 图表区域 -->
        <NGrid :x-gap="16" :y-gap="16" :cols="2" class="mb-6">
          <!-- 年龄分布饼图 -->
          <NGridItem>
            <NCard class="chart-card">
              <div ref="ageChartRef" class="chart-container"></div>
              <!-- 调试信息 -->
              <div class="text-xs text-gray-500 mt-2 p-2">
                <div v-for="(value, key) in age" :key="key">
                  {{ key }}: {{ value.toFixed(2) }}%
                </div>
              </div>
            </NCard>
          </NGridItem>

          <!-- 性别分布饼图 -->
          <NGridItem>
            <NCard class="chart-card">
              <div ref="genderChartRef" class="chart-container"></div>
              <!-- 调试信息 -->
              <div class="text-xs text-gray-500 mt-2 p-2">
                <div v-for="(value, key) in gender" :key="key">
                  {{ key }}: {{ value.toFixed(2) }}%
                </div>
              </div>
            </NCard>
          </NGridItem>
        </NGrid>
      </div>
    </NSpin>
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

.charts-container {
  position: relative;
  min-height: 300px;
}

.chart-card {
  height: 450px;
}

.chart-container {
  width: 100%;
  height: 300px;
}
</style>