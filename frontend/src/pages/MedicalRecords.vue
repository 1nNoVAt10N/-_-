<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useMessage } from 'naive-ui'
import {
    ArrowBackOutline,
    EyeOutline,
    CalendarOutline,
    MedicalOutline,
    WarningOutline,
    CheckmarkCircleOutline,
    PrintOutline,
    TimeOutline
} from '@vicons/ionicons5'
import { NIcon, NCard, NButton, NSpin, NImage, NTag, NPagination, NInput } from 'naive-ui'
import axios from 'axios'

class fund {
    fund_id: number
    left_fund: string
    left_fund_keyword: string
    right_fund: string
    right_fund_keyword: string
    patient_id: string
    constructor(fund_id: number, left_fund: string, left_fund_keyword: string, right_fund: string, right_fund_keyword: string, patient_id: string) {
        this.fund_id = fund_id
        this.left_fund = left_fund
        this.left_fund_keyword = left_fund_keyword
        this.right_fund = right_fund
        this.right_fund_keyword = right_fund_keyword
        this.patient_id = patient_id
    }
}

class patient{
    patient_id: string
    patient_name: string
    patient_age: number
    patient_gender: string
    constructor(patient_id: string,patient_name: string,    patient_age: number , patient_gender: string){
        this.patient_id = patient_id
        this.patient_name = patient_name
        this.patient_age = patient_age
        this.patient_gender = patient_gender
    }
}

class record {
    diagnosis_date: number
    fund_id: string
    patient_id: string
    record_id: string
    result: string
    suggestion: string
    user_id : string
    constructor(diagnosis_date: number, fund_id: string,  patient_id: string, record_id: string, result: string, suggestion: string, user_id: string) {
        this.diagnosis_date = diagnosis_date
        this.fund_id = fund_id
        this.patient_id = patient_id
        this.record_id = record_id
        this.result = result
        this.suggestion = suggestion
        this.user_id = user_id
    }
}

class Record {
    fund: fund
    patient: patient
    record: record
    constructor(fund: fund, patient: patient, record: record) {
        this.fund = fund
        this.patient = patient
        this.record = record
    }
}

const route = useRoute()
const router = useRouter()
const message = useMessage()
const historyRecords = ref<Record[]>([])
const loading = ref(true)
const currentPage = ref(1)
const pageSize = 5 

const patientId = ref('')
const fundId = ref('')

const paginatedRecords = computed(() => {
    const startIndex = (currentPage.value - 1) * pageSize
    const endIndex = startIndex + pageSize
    return historyRecords.value.slice(startIndex, endIndex)
})

// 总页数
const totalPages = computed(() => {
    return Math.ceil(historyRecords.value.length / pageSize)
})

// 页码变化处理
const handlePageChange = (page: number) => {
    currentPage.value = page
}

// 获取所有记录
onMounted(() => {
    axios.post('http://127.0.0.1:5000/get_recent_record', {limit:10}).then(res => {
        console.log('records:', res.data)
        historyRecords.value = res.data
        loading.value = false
    }).catch(err => {
        console.log('err:', err)
        message.error('获取记录失败')
        loading.value = false
    })
})

const viewHistoryDetail = (fund_id: string) => {
    router.push(`/fund/${fund_id}`)
}

// 跳转到患者诊断页面
const navigateToPatient = () => {
    if (patientId.value.trim()) {
        router.push(`/diagnosis/${patientId.value.trim()}`)
    } else {
        message.warning('请输入有效的患者ID')
    }
}

// 跳转到基金详情页面
const navigateToFund = () => {
    if (fundId.value.trim()) {
        router.push(`/fund/${fundId.value.trim()}`)
    } else {
        message.warning('请输入有效的Fund_ID')
    }
}
const goBack = () => {
        router.go(-1)
    }
</script>


<template>
    <div class="diagnosis-detail">
        <NPageHeader title="病例管理" @back="goBack">
            <template #avatar>
                <NIcon>
                    <MedicalOutline />
                </NIcon>
            </template>
        </NPageHeader>
    <!-- 历史记录区域 -->
    <div class="history-section">
        <h2 class="section-title">
            <NIcon size="20" class="mr-2">
                <TimeOutline />
            </NIcon>
            病例记录
        </h2>
        
        <p class="section-tip">(你可以手动输入或直接点击卡片跳转)</p>

        <!-- 跳转输入框区域 -->
        <div class="search-container">
            <div class="search-item">
                <NInput v-model:value="patientId" placeholder="请输入患者ID" />
                <NButton type="primary" @click="navigateToPatient">跳转到患者病例</NButton>
            </div>
            <div class="search-item">
                <NInput v-model:value="fundId" placeholder="请输入病历ID" />
                <NButton type="primary" @click="navigateToFund">跳转到眼底详情</NButton>
            </div>
        </div>

        <div v-if="loading" class="history-loading">
            <NSpin size="small" /> 加载中...
        </div>
        

        <div v-else-if="historyRecords.length === 0" class="no-history">
            暂无病例记录
        </div>
        

        <template v-else>
            <ul class="history-list">
                <li v-for="record in paginatedRecords" :key="record.record.record_id" class="history-item"
                    @click="viewHistoryDetail(record.record.fund_id)">
                    <div class="history-thumbnail">
                        <NImage :src="'data:image/jpeg;base64,'+record.fund.left_fund" alt="左眼眼底图像" />
                        <NImage :src="'data:image/jpeg;base64,'+record.fund.right_fund" alt="右眼眼底图像" />
                    </div>
                    <div class="history-info">
                        <div class="history-title">患者ID: {{ record.record.patient_id }}</div>
                        <div class="history-meta">
                            <div class="history-meta-item">
                                <NIcon size="14" class="mr-1">
                                    <CalendarOutline />
                                </NIcon>
                                <span>{{ record.record.diagnosis_date }}</span>
                            </div>
                            <div class="history-meta-item">
                                <NIcon size="14" class="mr-1">
                                    <EyeOutline />
                                </NIcon>
                                <span>病例ID: {{ record.fund.fund_id }}</span>
                            </div>
                        </div>
                    </div>
                    <NTag type="success" size="small">已完成</NTag>
                </li>
            </ul>

            <div class="pagination-container">
                <NPagination 
                    v-model:page="currentPage"
                    :page-count="totalPages"
                    :page-sizes="[5]"
                    :page-size="pageSize"
                    @update:page="handlePageChange"
                />
            </div>
        </template>
    </div>
    </div>
</template>

<style scoped>
.diagnosis-detail {
    padding: 20px;
}

/* 面包屑导航 */
.breadcrumb {
    margin-bottom: 20px;
    font-size: 14px;
    color: var(--n-text-color-3);
}

.breadcrumb a {
    color: var(--n-primary-color);
    text-decoration: none;
}

.breadcrumb i {
    margin: 0 8px;
    font-size: 12px;
}

.history-section {
    background-color: var(--n-card-color);
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-top: 20px;
}

.section-title {
    font-size: 18px;
    color: var(--n-text-color);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
}

.history-loading,
.no-history {
    padding: 30px 0;
    text-align: center;
    color: var(--n-text-color-3);
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.history-list {
    list-style-type: none;
    padding: 0;
    margin: 0;
}

.history-item {
    display: flex;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid var(--n-border-color);
    cursor: pointer;
    transition: all 0.3s;
}

.history-item:last-child {
    border-bottom: none;
}

.history-item:hover {
    background-color: var(--n-fill-color);
}

.history-thumbnail {
    width: 160px;
    height: 60px;
    border-radius: 4px;
    overflow: hidden;
    margin-right: 15px;
    background-color: var(--n-fill-color);
    display: flex;
}

.history-info {
    flex: 1;
}

.history-title {
    font-size: 14px;
    color: var(--n-text-color);
    margin-bottom: 5px;
}

.history-meta {
    display: flex;
    font-size: 12px;
    color: var(--n-text-color-3);
}

.history-meta-item {
    margin-right: 12px;
    display: flex;
    align-items: center;
}

.pagination-container {
    margin-top: 20px;
    display: flex;
    justify-content: center;
}

/* 响应式设计 */
@media screen and (max-width: 768px) {
    .history-item {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .history-thumbnail {
        width: 100%;
        margin-bottom: 10px;
    }
}

.search-container {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.search-item {
    display: flex;
    gap: 8px;
    flex: 1;
    min-width: 280px;
}

.section-tip {
    color: var(--n-text-color-3);
    font-size: 12px;
    margin-top: -12px;
    margin-bottom: 16px;
}
</style>
