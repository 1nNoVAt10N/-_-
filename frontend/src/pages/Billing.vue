<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  ReceiptOutline,
  WalletOutline,
  TimeOutline,
  CalendarOutline,
  FilterOutline,
  DownloadOutline,
  PrintOutline
} from '@vicons/ionicons5'
import { 
  NCard, 
  NGrid, 
  NGridItem, 
  NIcon,
  NButton,
  NSpace,
  NDataTable,
  NInput,
  NSelect,
  NDatePicker,
  NTag
} from 'naive-ui'

// 账单统计
const billingStats = ref({
  totalAmount: '¥128,560.00',
  paidAmount: '¥98,450.00',
  pendingAmount: '¥30,110.00',
  totalOrders: 256
})

// 账单列表
const bills = ref([
  {
    id: 'BILL-2024-001',
    customer: '张三',
    amount: '¥1,280.00',
    status: 'paid',
    date: '2024-03-20',
    type: '诊断服务'
  },
  {
    id: 'BILL-2024-002',
    customer: '李四',
    amount: '¥2,560.00',
    status: 'pending',
    date: '2024-03-19',
    type: '批量诊断'
  },
  {
    id: 'BILL-2024-003',
    customer: '王五',
    amount: '¥3,840.00',
    status: 'overdue',
    date: '2024-03-18',
    type: '高级诊断'
  }
])

// 状态选项
const statusOptions = [
  { label: '全部', value: 'all' },
  { label: '已支付', value: 'paid' },
  { label: '待支付', value: 'pending' },
  { label: '已逾期', value: 'overdue' }
]

// 搜索和筛选
const searchText = ref('')
const selectedStatus = ref('all')
const dateRange = ref(null)

// 表格列定义
const columns = [
  { title: '账单编号', key: 'id' },
  { title: '客户名称', key: 'customer' },
  { title: '金额', key: 'amount' },
  { title: '状态', key: 'status', render: (row: { status: 'paid' | 'pending' | 'overdue' }) => {
    const statusMap = {
      paid: { type: 'success' as const, text: '已支付' },
      pending: { type: 'warning' as const, text: '待支付' },
      overdue: { type: 'error' as const, text: '已逾期' }
    }
    const status = statusMap[row.status]
    return h(NTag, { type: status.type }, { default: () => status.text })
  }},
  { title: '日期', key: 'date' },
  { title: '类型', key: 'type' },
  { title: '操作', key: 'actions', render: (row: any) => {
    return h(NSpace, null, {
      default: () => [
        h(NButton, {
          size: 'small',
          type: 'primary',
          onClick: () => handlePrint(row)
        }, { default: () => '打印' }),
        h(NButton, {
          size: 'small',
          type: 'info',
          onClick: () => handleDownload(row)
        }, { default: () => '下载' })
      ]
    })
  }}
]

// 处理打印
const handlePrint = (bill: any) => {
  console.log('打印账单:', bill.id)
}

// 处理下载
const handleDownload = (bill: any) => {
  console.log('下载账单:', bill.id)
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
          <ReceiptOutline />
        </NIcon>
        收费凭单
      </h1>
      <p class="text-gray-500 mt-2">管理和查看系统收费记录及账单信息</p>
    </div>

    <!-- 账单统计 -->
    <NGrid :x-gap="16" :y-gap="16" :cols="4" class="mb-6">
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-blue-500">
              <WalletOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">总金额</div>
              <div class="text-lg font-bold">{{ billingStats.totalAmount }}</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-green-500">
              <ReceiptOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">已支付</div>
              <div class="text-lg font-bold text-green-500">{{ billingStats.paidAmount }}</div>
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
              <div class="text-gray-500">待支付</div>
              <div class="text-lg font-bold text-orange-500">{{ billingStats.pendingAmount }}</div>
            </div>
          </div>
        </NCard>
      </NGridItem>
      <NGridItem>
        <NCard>
          <div class="flex items-center">
            <NIcon size="24" class="mr-3 text-purple-500">
              <CalendarOutline />
            </NIcon>
            <div>
              <div class="text-gray-500">总订单数</div>
              <div class="text-lg font-bold">{{ billingStats.totalOrders }}</div>
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
          placeholder="搜索账单编号或客户名称"
          style="width: 250px"
        />
        <NSelect
          v-model:value="selectedStatus"
          :options="statusOptions"
          style="width: 150px"
        />
        <NDatePicker
          v-model:value="dateRange"
          type="daterange"
          clearable
          :shortcuts="{ '最近30天': [Date.now() - 30 * 24 * 60 * 60 * 1000, Date.now()] }"
        />
        <NButton type="primary">
          <template #icon>
            <NIcon><FilterOutline /></NIcon>
          </template>
          筛选
        </NButton>
      </NSpace>
    </NCard>

    <!-- 账单列表 -->
    <NCard>
      <template #header>
        <div class="flex justify-between items-center">
          <span>账单列表</span>
          <NSpace>
            <NButton type="primary">
              <template #icon>
                <NIcon><DownloadOutline /></NIcon>
              </template>
              导出
            </NButton>
            <NButton>
              <template #icon>
                <NIcon><PrintOutline /></NIcon>
              </template>
              批量打印
            </NButton>
          </NSpace>
        </div>
      </template>
      <NDataTable
        :columns="columns"
        :data="bills"
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