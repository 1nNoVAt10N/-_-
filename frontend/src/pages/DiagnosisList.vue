<script setup lang="ts">
import { ref, onMounted } from 'vue'
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
import { NIcon, NCard, NButton, NSpin, NImage, NTag, NProgress } from 'naive-ui'
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
class Record {
    diagnosis_date: number
    fund_id: number
    fund_info: fund
    patient_id: string
    record_id: string
    result: string
    suggestion: string
    user_id : string
    constructor(diagnosis_date: number, fund_id: number, fund_info: fund, patient_id: string, record_id: string, result: string, suggestion: string, user_id: string) {
        this.diagnosis_date = diagnosis_date
        this.fund_id = fund_id
        this.fund_info = fund_info
        this.patient_id = patient_id
        this.record_id = record_id
        this.result = result
        this.suggestion = suggestion
        this.user_id = user_id
    }
}
const route = useRoute()
const router = useRouter()
const message = useMessage()
const historyRecords = ref<Record[]>([])
const diagnosisId = route.params.id
console.log('diagnosisId:', diagnosisId);
axios.post('http://127.0.0.1:5000/get_patient_record', { id: diagnosisId }).then(res => {
    console.log('patient:', res.data.patient)
    console.log('record:', res.data.records)
    historyRecords.value = res.data.records
}).catch(err => {
    console.log('err:', err);
})
const viewHistoryDetail = (fund_id: number) => {
    router.push(`/fund/${fund_id}`)
}
const goBack = () => {
        router.go(-1)
    }
</script>

<template>
    <div class="diagnosis-detail">
        <NPageHeader title="病人病例" @back="goBack">
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
            患者ID: {{ historyRecords[0].patient_id }} 的历史记录
        </h2>
        <!-- 历史记录列表 -->
        <ul class="history-list">
            <li v-for="record in historyRecords" :key="record.patient_id" class="history-item"
                @click="viewHistoryDetail(record.fund_id)">
                <div class="history-thumbnail">
                    <NImage :src="'data:image/jpeg;base64,'+record.fund_info.left_fund" alt="左眼眼底图像" />
                    <NImage :src="'data:image/jpeg;base64,'+record.fund_info.right_fund" alt="右眼眼底图像" />
                </div>
                <div class="history-info">
                    <div class="history-title">病例ID: {{ record.fund_id }}</div>
                    <div class="history-meta">
                        <div class="history-meta-item">
                            <NIcon size="14" class="mr-1">
                                <CalendarOutline />
                            </NIcon>
                            <span>{{ record.diagnosis_date }}</span>
                        </div>
                        <div class="history-meta-item">
                            <NIcon size="14" class="mr-1">
                                <EyeOutline />
                            </NIcon>
                        </div>
                    </div>
                </div>
                <NTag type="success" size="small">已完成</NTag>
            </li>
        </ul>
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

/* 页面标题和操作按钮 */
.page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.title-section {
    display: flex;
    align-items: center;
    gap: 20px;
}

.page-title {
    font-size: 22px;
    color: var(--n-text-color);
    margin: 0;
    display: flex;
    align-items: center;
}

/* 加载状态 */
.loading-container {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: var(--n-text-color-3);
}

/* 详情内容 */
.detail-content {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

/* 卡片样式 */
.info-card,
.image-card,
.result-card,
.suggestion-card,
.notes-card {
    background-color: var(--n-card-color);
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.card-title {
    display: flex;
    align-items: center;
    font-size: 16px;
    color: var(--n-text-color);
}

/* 基本信息 */
.info-content {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.info-item {
    display: flex;
    align-items: center;
}

.info-item .label {
    width: 100px;
    color: var(--n-text-color-3);
}

/* 眼底图像 */
.images-container {
    display: flex;
    gap: 20px;
}

.image-item {
    flex: 1;
}

.image-item h3 {
    margin: 0 0 10px 0;
    font-size: 14px;
    color: var(--n-text-color);
}

.eye-image {
    width: 100%;
    border-radius: 4px;
}

/* 分析结果 */
.result-content {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.result-item {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}

.result-label {
    width: 120px;
    color: var(--n-text-color-3);
}

.result-value {
    flex: 1;
    min-width: 150px;
}

.result-value.positive {
    color: var(--n-error-color);
    font-weight: 500;
}

.result-value.negative {
    color: var(--n-success-color);
}

.confidence-bar {
    width: 100%;
}

/* 诊断建议 */
.suggestion-content {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.problems-section,
.recommendations-section {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.problems-section h3,
.recommendations-section h3 {
    margin: 0;
    font-size: 14px;
    color: var(--n-text-color);
}

.problems-section ul,
.recommendations-section ul {
    margin: 0;
    padding-left: 20px;
}

.problems-section li,
.recommendations-section li {
    color: var(--n-text-color);
    line-height: 1.6;
}

/* 医生备注 */
.notes-content {
    padding: 10px 0;
}

.notes-textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid var(--n-border-color);
    border-radius: 4px;
    background-color: var(--n-input-color);
    color: var(--n-text-color);
    resize: vertical;
}

.notes-textarea:focus {
    outline: none;
    border-color: var(--n-primary-color);
}

.history-section {
    background-color: var(--n-card-color);
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-top: 20px;
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

/* 响应式设计 */
@media screen and (max-width: 768px) {
    .images-container {
        flex-direction: column;
    }

    .page-header {
        flex-direction: column;
        gap: 10px;
        align-items: flex-start;
    }
}
</style>