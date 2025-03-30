<script setup lang="ts">
import { ref, onMounted, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useMessage, useDialog } from 'naive-ui';
import { lyla } from '@lylajs/web';
import {
    CloudUploadOutline,
    SearchOutline,
    MedicalOutline,
    TimeOutline,
    EyeOutline,
    CalendarOutline,
    WarningOutline,
    CheckmarkCircleOutline,
    ImageOutline,
    DocumentTextOutline,
    ArchiveOutline,
    ListOutline,
    PrintOutline
} from '@vicons/ionicons5';
import { NIcon, NCard, NButton, NSpin, NUpload, NImage, NTag, NProgress, NPageHeader, NForm, NFormItem, NInput, NRadioGroup, NRadio, NInputNumber, NDataTable, NPagination } from 'naive-ui';
import { read, utils } from 'xlsx';

// 与MedicalRecords.vue中相匹配的数据结构
class Fund {
    fund_id: string;
    left_fund: string;
    left_fund_keyword: string;
    right_fund: string;
    right_fund_keyword: string;
    patient_id: string;
    constructor(fund_id: string, left_fund: string, left_fund_keyword: string, right_fund: string, right_fund_keyword: string, patient_id: string) {
        this.fund_id = fund_id;
        this.left_fund = left_fund;
        this.left_fund_keyword = left_fund_keyword;
        this.right_fund = right_fund;
        this.right_fund_keyword = right_fund_keyword;
        this.patient_id = patient_id;
    }
}

class Patient {
    patient_id: string;
    patient_name: string;
    patient_age: number;
    patient_gender: string;
    constructor(patient_id: string, patient_name: string, patient_age: number, patient_gender: string) {
        this.patient_id = patient_id;
        this.patient_name = patient_name;
        this.patient_age = patient_age;
        this.patient_gender = patient_gender;
    }
}

class Record {
    diagnosis_date: number;
    fund_id: string;
    patient_id: string;
    record_id: string;
    result: string;
    suggestion: string;
    user_id: string;
    constructor(diagnosis_date: number, fund_id: string, patient_id: string, record_id: string, result: string, suggestion: string, user_id: string) {
        this.diagnosis_date = diagnosis_date;
        this.fund_id = fund_id;
        this.patient_id = patient_id;
        this.record_id = record_id;
        this.result = result;
        this.suggestion = suggestion;
        this.user_id = user_id;
    }
}

class BatchRecord {
    fund: Fund;
    patient: Patient;
    record: Record;
    constructor(fund: Fund, patient: Patient, record: Record) {
        this.fund = fund;
        this.patient = patient;
        this.record = record;
    }
}

const router = useRouter();
const message = useMessage();
const dialog = useDialog();
const backendAddr = import.meta.env.VITE_API_URL;

// 引用和状态变量
const zipFileInput = ref<HTMLInputElement | null>(null);
const excelFileInput = ref<HTMLInputElement | null>(null);
const zipFileName = ref('');
const excelFileName = ref('');
const zipFileUploaded = ref(false);
const excelFileUploaded = ref(false);
const zipFile = ref<any>(null);
const excelFile = ref<any>(null);
const analyzeBtn = ref(false);
const isAnalyzing = ref(false);
const analysisCompleted = ref(false);

// 批量处理结果和分页
const batchResults = ref<BatchRecord[]>([]);
const currentPage = ref(1);
const pageSize = ref(5);

// 表单数据
const patientData = ref<any[]>([]);
const patientDataHeaders = ref<string[]>([]);

// 检测结果
const detectionCard = ref({
    status: 'waiting',
    statusText: '等待分析',
    isActive: false,
    results: [] as { name: string; isPositive: boolean; confidence: number }[],
});
const diagnosisCard = ref({
    status: 'waiting',
    statusText: '等待分析',
    isActive: false,
    content: {
        problems: [] as string[],
        recommendations: [] as string[],
    },
});

// 分页计算
const paginatedResults = computed(() => {
    const startIndex = (currentPage.value - 1) * pageSize.value;
    const endIndex = startIndex + pageSize.value;
    return batchResults.value.slice(startIndex, endIndex);
});

const totalPages = computed(() => {
    return Math.ceil(batchResults.value.length / pageSize.value);
});

// 处理页码变化
const handlePageChange = (page: number) => {
    currentPage.value = page;
};

// 处理拖拽上传
const handleDragOver = (e: any, type: string) => {
    e.preventDefault();
    const uploadArea = document.querySelector(`.upload-area.${type}`);
    if (uploadArea) {
        uploadArea.classList.add('drag-over');
    }
};

const handleDragLeave = (type: string) => {
    const uploadArea = document.querySelector(`.upload-area.${type}`);
    if (uploadArea) {
        uploadArea.classList.remove('drag-over');
    }
};

const handleDrop = (e: any, type: string) => {
    e.preventDefault();
    const uploadArea = document.querySelector(`.upload-area.${type}`);
    if (uploadArea) {
        uploadArea.classList.remove('drag-over');
    }

    if (e.dataTransfer.files.length) {
        if (type === 'zip') {
            handleZipFile(e.dataTransfer.files[0]);
        } else if (type === 'excel') {
            handleExcelFile(e.dataTransfer.files[0]);
        }
    }
};

// 文件处理
const handleZipFile = (file: any) => {
    if (file.type !== 'application/zip' && file.type !== 'application/x-zip-compressed') {
        message.error('请上传ZIP压缩文件');
        return;
    }

    zipFile.value = file;
    zipFileName.value = file.name;
    zipFileUploaded.value = true;
    checkAnalyzeButton();
};

const handleExcelFile = async (file: any) => {
    if (!file.type.includes('spreadsheet') && !file.name.endsWith('.xlsx') && !file.name.endsWith('.xls')) {
        message.error('请上传Excel表格文件');
        return;
    }

    excelFile.value = file;
    excelFileName.value = file.name;
    excelFileUploaded.value = true;

    // 解析Excel文件
    try {
        const data = await file.arrayBuffer();
        const workbook = read(data, { type: 'array' });
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        const jsonData = utils.sheet_to_json(worksheet);
        
        patientData.value = jsonData;
        if (jsonData.length > 0) {
            patientDataHeaders.value = Object.keys(jsonData[0] as object);
        }
        
        message.success('Excel数据解析成功');
    } catch (error) {
        console.error('Excel解析错误:', error);
        message.error('Excel数据解析失败');
    }
    
    checkAnalyzeButton();
};

// 文件上传处理
const handleZipFileUpload = (event: any) => {
    const file = event.target.files[0];
    if (file) {
        handleZipFile(file);
    }
};

const handleExcelFileUpload = (event: any) => {
    const file = event.target.files[0];
    if (file) {
        handleExcelFile(file);
    }
};

// 检查是否可以开始分析
const checkAnalyzeButton = () => {
    analyzeBtn.value = zipFileUploaded.value && excelFileUploaded.value;
};

// 分析处理
const startAnalysis = async () => {
    // 检查是否可以开始分析
    if (!analyzeBtn.value) {
        message.error('请先上传ZIP图像文件和Excel表单数据');
        return;
    }

    isAnalyzing.value = true;

    // 设置卡片激活状态
    detectionCard.value.isActive = true;
    diagnosisCard.value.isActive = true;

    // 更新检测卡片状态
    detectionCard.value.status = 'analyzing';
    detectionCard.value.statusText = '分析中';

    try {
        const formData = new FormData();
        formData.append('images_zip', zipFile.value);
        formData.append('patient_data', excelFile.value);
        
        // 打印 FormData 内容
        for (let [key, value] of formData.entries()) {
            console.log(`${key}: ${value}`);
        }
        
        // 更新API路径与数据库结构匹配
        const { json } = await lyla.post('http://localhost:5000/batch_analysis', {
            body: formData,
            onUploadProgress: ({ percent }) => {
                console.log('上传进度:', Math.ceil(percent));
            },
        });

        // 更新检测卡片状态
        detectionCard.value.status = 'completed';
        detectionCard.value.statusText = '分析完成';

        console.log('批量检测结果:', json);

        // 处理返回的批量结果
        if (json && Object.keys(json).length > 0) {
            // 处理批量结果
            const processedResults: BatchRecord[] = [];
            
            // 遍历每个结果记录
            Object.entries(json).forEach(([key, value]: [string, any]) => {
                const recordData = value;
                
                // 创建模型实例
                if (recordData.fund_id && recordData.record_id) {
                    // 创建患者记录 - 使用默认值，因为JSON中没有患者信息
                    const patient = new Patient(
                        `patient_${recordData.fund_id}`,
                        '未知',
                        0,
                        '未知'
                    );
                    
                    // 创建基金记录
                    const fund = new Fund(
                        recordData.fund_id.toString(),
                        '', // 左眼图像暂无
                        '',
                        '', // 右眼图像暂无
                        '',
                        `patient_${recordData.fund_id}`
                    );
                    
                    // 处理预测结果
                    const predictions = Array.isArray(recordData.predictions) 
                        ? recordData.predictions.join(', ') 
                        : '未知';
                    
                    // 创建诊断记录
                    const record = new Record(
                        Date.now(),
                        recordData.fund_id.toString(),
                        `patient_${recordData.fund_id}`,
                        recordData.record_id.toString(),
                        predictions,
                        '请进一步咨询医生',
                        'current_user'
                    );
                    
                    // 添加到结果列表
                    processedResults.push(new BatchRecord(fund, patient, record));
                }
            });
            
            // 更新结果
            batchResults.value = processedResults;
            
            // 为当前选中的记录更新详细信息
            if (processedResults.length > 0) {
                updateDetailForRecord(processedResults[0]);
            }
        }

        // 更新诊断卡片状态
        diagnosisCard.value.status = 'completed';
        diagnosisCard.value.statusText = '分析完成';
        
        // 更新状态
        analysisCompleted.value = true;

    } catch (error) {
        console.error('分析失败:', error);
        message.error('分析失败，请重试');
        dialog.error({
            title: '错误',
            content: '分析过程中发生错误，请重试',
            positiveText: '重试',
            negativeText: '返回首页',
            onPositiveClick: () => {
                isAnalyzing.value = false;
            },
            onNegativeClick: () => {
                router.push('/');
            },
        });
        // 更新状态为错误
        detectionCard.value.status = 'error';
        detectionCard.value.statusText = '分析失败';
        diagnosisCard.value.status = 'error';
        diagnosisCard.value.statusText = '分析失败';
    } finally {
        isAnalyzing.value = false;
    }
};

// 更新详细信息卡
const updateDetailForRecord = (record: BatchRecord) => {
    if (!record) return;
    
    detectionCard.value.results = [{
        name: record.record.result,
        isPositive: true,
        confidence: 85 // 假设值
    }];
    
    // 更新诊断建议
    diagnosisCard.value.content.problems = [
        `患者ID ${record.patient.patient_id} (${record.patient.patient_name}) 的眼底异常`
    ];
    
    diagnosisCard.value.content.recommendations = [
        `患者姓名：${record.patient.patient_name}`,
        `患者性别：${record.patient.patient_gender}`,
        `患者年龄：${record.patient.patient_age}`,
        `诊断结果：${record.record.result}`,
        `诊断建议：${record.record.suggestion}`
    ];
};

// 查看单条记录详情
const viewRecordDetail = (record: BatchRecord) => {
    updateDetailForRecord(record);
};

// 查看详细报告
const viewDetailReport = () => {
    // 跳转到批量报告页面
    if (batchResults.value.length > 0) {
        const fundIds = batchResults.value.map(r => r.fund.fund_id).join(',');
        router.push(`/batch-report?funds=${fundIds}`);
    } else {
        message.warning('没有可查看的报告');
    }
};

// 跳转到基金详情页面
const viewHistoryDetail = (fundId: string) => {
    router.push(`/fund/${fundId}`);
};

// 导出所有分析结果
const exportResults = () => {
    if (batchResults.value.length === 0) {
        message.warning('没有可导出的结果');
        return;
    }
    
    // 创建导出数据
    const exportData = batchResults.value.map(r => ({
        '患者ID': r.patient.patient_id,
        '患者姓名': r.patient.patient_name,
        '患者性别': r.patient.patient_gender,
        '患者年龄': r.patient.patient_age,
        '记录ID': r.record.record_id,
        '基金ID': r.fund.fund_id, 
        '诊断结果': r.record.result,
        '诊断建议': r.record.suggestion,
        '诊断日期': new Date(r.record.diagnosis_date).toLocaleDateString()
    }));
    
    // 创建工作簿
    const workbook = utils.book_new();
    const worksheet = utils.json_to_sheet(exportData);
    
    // 添加工作表到工作簿
    utils.book_append_sheet(workbook, worksheet, "分析结果");
    
    // 生成Excel并下载
    const now = new Date().toISOString().slice(0, 10);
    // utils.writeFile(workbook, `多模态分析结果_${now}.xlsx`);
};

const goBack = () => {
    router.go(-1);
};
</script>

<template>
    <div class="diagnosis-page">
        <!-- 面包屑导航 -->
        <NPageHeader title="多模态批量智能诊断" @back="goBack">
            <template #avatar>
                <NIcon size="24" class="mr-2">
                    <EyeOutline />
                </NIcon>
            </template>
        </NPageHeader>

        <div class="diagnosis-container">
            <!-- 左侧上传和预览区域 -->
            <div class="upload-section">
                <h2 v-if="detectionCard.status !== 'completed'" class="section-title">
                    <NIcon size="20" class="mr-2">
                        <CloudUploadOutline />
                    </NIcon>
                    上传批量诊断数据
                </h2>

                <!-- ZIP文件上传区域 -->
                <NCard class="analysis-card">
                    <h3 class="eye-title">
                        <NIcon size="18" class="mr-2">
                            <ArchiveOutline />
                        </NIcon>
                        图像数据包 (ZIP格式)
                    </h3>
                    <div v-if="!zipFileUploaded" class="upload-area zip" @click="zipFileInput?.click()"
                        @dragover="(e) => handleDragOver(e, 'zip')" @dragleave="() => handleDragLeave('zip')"
                        @drop="(e) => handleDrop(e, 'zip')">
                        <div class="upload-icon">
                            <NIcon size="48">
                                <ArchiveOutline />
                            </NIcon>
                        </div>
                        <div class="upload-text">点击或拖拽ZIP文件到此区域上传</div>
                        <div class="upload-hint">支持ZIP格式，文件大小不超过100MB</div>
                        <div class="upload-hint">
                            推荐目录结构：患者ID_记录ID/左眼.jpg, 右眼.jpg
                        </div>
                        <NButton type="primary">选择文件</NButton>
                        <input type="file" ref="zipFileInput" style="display: none" accept=".zip"
                            @change="handleZipFileUpload" />
                    </div>
                    <div v-else class="file-uploaded">
                        <div class="file-info">
                            <NIcon size="24" class="mr-2">
                                <ArchiveOutline />
                            </NIcon>
                            <span>{{ zipFileName }}</span>
                        </div>
                        <div class="file-actions">
                            <NButton size="small" @click="zipFileUploaded = false; zipFile = null; checkAnalyzeButton()">
                                移除
                            </NButton>
                        </div>
                    </div>
                </NCard>

                <!-- Excel文件上传区域 -->
                <NCard class="analysis-card">
                    <h3 class="eye-title">
                        <NIcon size="18" class="mr-2">
                            <DocumentTextOutline />
                        </NIcon>
                        患者数据表 (Excel格式)
                    </h3>
                    <div v-if="!excelFileUploaded" class="upload-area excel" @click="excelFileInput?.click()"
                        @dragover="(e) => handleDragOver(e, 'excel')" @dragleave="() => handleDragLeave('excel')"
                        @drop="(e) => handleDrop(e, 'excel')">
                        <div class="upload-icon">
                            <NIcon size="48">
                                <DocumentTextOutline />
                            </NIcon>
                        </div>
                        <div class="upload-text">点击或拖拽Excel文件到此区域上传</div>
                        <div class="upload-hint">支持XLSX格式，文件大小不超过20MB</div>
                        <div class="upload-hint">
                            必需字段：patient_id, patient_name, patient_age, patient_gender
                        </div>
                        <NButton type="primary">选择文件</NButton>
                        <input type="file" ref="excelFileInput" style="display: none" accept=".xlsx,.xls"
                            @change="handleExcelFileUpload" />
                    </div>
                    <div v-else class="file-uploaded">
                        <div class="file-info">
                            <NIcon size="24" class="mr-2">
                                <DocumentTextOutline />
                            </NIcon>
                            <span>{{ excelFileName }}</span>
                        </div>
                        <div class="file-actions">
                            <NButton size="small" @click="excelFileUploaded = false; excelFile = null; patientData = []; checkAnalyzeButton()">
                                移除
                            </NButton>
                        </div>
                    </div>

                    <!-- 表格数据预览 -->
                    <div v-if="patientData.length > 0" class="excel-preview">
                        <h4>患者数据预览 <span class="record-count">({{ patientData.length }}条记录)</span></h4>
                        <div class="scroll-wrapper">
                            <div class="table-container">
                                <table>
                                    <thead>
                                        <tr>
                                            <th v-for="header in patientDataHeaders" :key="header" class="table-header">
                                                {{ header }}
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr v-for="(row, index) in patientData.slice(0, 5)" :key="index">
                                            <td v-for="header in patientDataHeaders" :key="`${index}-${header}`" class="table-cell">
                                                {{ row[header] }}
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                                <div v-if="patientData.length > 5" class="more-data-hint">
                                    显示前5条数据，共{{ patientData.length }}条
                                </div>
                            </div>
                        </div>
                        <div class="scroll-hint">← 左右滑动查看全部数据 →</div>
                    </div>
                </NCard>

            </div>

            <!-- 右侧分析结果区域 -->
            <div class="result-section">
                <!-- 分析结果卡片 -->
                <NCard :class="{ inactive: !detectionCard.isActive }" class="analysis-card">
                    <template #header>
                        <div class="card-header">
                            <div class="card-title">
                                <NIcon size="20" class="mr-2">
                                    <SearchOutline />
                                </NIcon>
                                多模态批量分析
                            </div>
                            <NTag :type="detectionCard.status === 'completed'
                                ? 'success'
                                : detectionCard.status === 'analyzing'
                                    ? 'warning'
                                    : detectionCard.status === 'error'
                                        ? 'error'
                                        : 'info'
                                ">
                                {{ detectionCard.statusText }}
                            </NTag>
                        </div>
                    </template>
                    <div class="card-content">
                        <p v-if="detectionCard.status === 'waiting' || detectionCard.status === 'analyzing'"
                            class="placeholder-text">
                            {{
                                detectionCard.status === 'waiting'
                                    ? '请先上传ZIP图像文件和Excel表单数据并点击"开始分析"按钮'
                                    : '正在分析中，请稍候...'
                            }}
                        </p>
                        <div v-if="detectionCard.status === 'completed'" class="result-container">
                            <h3 class="summary-title">批量检测结果摘要</h3>
                            <div class="batch-results-summary">
                                <div class="summary-item">
                                    <div class="summary-label">总记录:</div>
                                    <div class="summary-value">{{ batchResults.length }}</div>
                                </div>
                                <div class="summary-item">
                                    <div class="summary-label">正常:</div>
                                    <div class="summary-value">{{ batchResults.filter((r: BatchRecord) => r.record.result.includes('正常')).length }}</div>
                                </div>
                                <div class="summary-item">
                                    <div class="summary-label">异常:</div>
                                    <div class="summary-value">{{ batchResults.filter((r: BatchRecord) => !r.record.result.includes('正常')).length }}</div>
                                </div>
                            </div>
                            <p class="batch-instruction">请查看下方"批量分析结果"区域了解详情</p>
                        </div>
                        <p v-if="detectionCard.status === 'error'" class="error-text">
                            分析时发生错误，请重试
                        </p>
                    </div>
                </NCard>

                <!-- 诊断建议卡片 -->
                <NCard :class="{ inactive: !diagnosisCard.isActive }" class="analysis-card">
                    <template #header>
                        <div class="card-header">
                            <div class="card-title">
                                <NIcon size="20" class="mr-2">
                                    <ListOutline />
                                </NIcon>
                                批量分析结果
                                <span v-if="batchResults.length > 0" class="record-count">({{ batchResults.length }}条)</span>
                            </div>
                            <NTag :type="diagnosisCard.status === 'completed' ? 'success' :
                                diagnosisCard.status === 'analyzing' ? 'warning' :
                                    diagnosisCard.status === 'error' ? 'error' : 'info'"
                                size="small">
                                {{ diagnosisCard.statusText }}
                            </NTag>
                        </div>
                    </template>
                    <div class="card-content">
                        <p v-if="diagnosisCard.status === 'waiting' || diagnosisCard.status === 'analyzing'"
                            class="placeholder-text">
                            {{ diagnosisCard.status === 'waiting' ? '分析完成后将显示结果' : '正在生成结果，请稍候...' }}
                        </p>
                        <div v-if="diagnosisCard.status === 'completed'" class="diagnosis-content">
                            <div v-if="batchResults.length > 0" class="batch-results-details">
                                <ul class="diagnosis-result-list">
                                    <li v-for="(record, index) in batchResults" :key="record.record.record_id" class="diagnosis-result-item">
                                        <div class="result-header">
                                            <span class="result-index">{{ index + 1 }}</span>
                                            <span class="result-id">记录ID: {{ record.record.record_id }}</span>
                                            <span class="result-fund">ID: {{ record.fund.fund_id }}</span>
                                        </div>
                                        <div class="result-body">
                                            <div class="result-diagnosis">
                                                <span class="truncated-text">
                                                    {{ record.record.result.length > 30 ? record.record.result.slice(0, 30) + '...' : record.record.result }}
                                                </span>
                                                <div class="result-tooltip" v-if="record.record.result.length > 30">
                                                    {{ record.record.result }}
                                                </div>
                                            </div>
                                            <NButton type="primary" size="small" class="view-detail-btn" @click="viewHistoryDetail(record.fund.fund_id)">
                                                详情
                                            </NButton>
                                        </div>
                                    </li>
                                </ul>
                            </div>
                            <p v-else class="no-results">暂无批量分析结果</p>
                        </div>
                        <p v-if="diagnosisCard.status === 'error'" class="error-text">
                            结果生成失败，请重试
                        </p>
                    </div>
                </NCard>

                <!-- 分析按钮 -->
                <div class="action-buttons">
                    <NButton type="primary" size="large" block :disabled="!analyzeBtn || isAnalyzing"
                        @click="startAnalysis()" :loading="isAnalyzing" v-if="!analysisCompleted">
                        {{ isAnalyzing ? '分析中...' : '开始批量分析' }}
                    </NButton>
                </div>
            </div>
        </div>
    </div>
</template>

<style scoped>
.diagnosis-page {
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

/* 页面标题 */
.page-title {
    font-size: 22px;
    color: var(--n-text-color);
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--n-border-color);
    display: flex;
    align-items: center;
}

/* 主要内容区域 */
.diagnosis-container {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}

/* 左侧上传和预览区域 */
.upload-section {
    flex: 1 1 450px;
    max-width: 100%;
    background-color: var(--n-card-color);
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    padding: 20px;
    display: flex;
    flex-direction: column;
}

.section-title {
    font-size: 16px;
    color: var(--n-text-color);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
}

.eye-title {
    font-size: 16px;
    color: var(--n-text-color);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
}

.upload-area.zip,
.upload-area.excel {
    border: 2px dashed var(--n-border-color);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    margin-bottom: 10px;
    cursor: pointer;
    transition: all 0.3s;
    background-color: var(--n-card-color);
}

.upload-area.zip:hover,
.upload-area.zip.drag-over,
.upload-area.excel:hover,
.upload-area.excel.drag-over {
    border-color: var(--n-primary-color);
    background-color: var(--n-primary-color-hover);
}

.upload-icon {
    margin-bottom: 15px;
    color: var(--n-primary-color);
}

.upload-text {
    color: var(--n-text-color);
    margin-bottom: 15px;
}

.upload-hint {
    color: var(--n-text-color-3);
    font-size: 12px;
    margin-bottom: 15px;
}

.file-uploaded {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    border: 1px solid var(--n-border-color);
    border-radius: 4px;
    margin-bottom: 10px;
}

.file-info {
    display: flex;
    align-items: center;
}

.file-actions {
    display: flex;
    gap: 10px;
}

.excel-preview {
    margin-top: 15px;
    width: 100%;
}

.excel-preview h4 {
    margin-bottom: 10px;
    font-weight: 500;
    display: flex;
    align-items: center;
    font-size: 14px;
}

.excel-preview .record-count {
    font-size: 12px;
    color: var(--n-text-color-3);
    margin-left: 5px;
    font-weight: normal;
}

.scroll-wrapper {
    width: 100%;
    overflow: hidden;
    position: relative;
}

.table-container {
    border: 1px solid var(--n-border-color);
    border-radius: 4px;
    overflow-x: auto;
    overflow-y: auto;
    max-height: 180px;
    margin-bottom: 8px;
    position: relative;
    background-image: 
        linear-gradient(to right, var(--n-card-color), var(--n-card-color)),
        linear-gradient(to right, var(--n-card-color), var(--n-card-color)),
        linear-gradient(to right, rgba(0,0,0,.1), rgba(255,255,255,0)),
        linear-gradient(to left, rgba(0,0,0,.1), rgba(255,255,255,0));
    background-position: left center, right center, left center, right center;
    background-repeat: no-repeat;
    background-color: var(--n-card-color);
    background-size: 20px 100%, 20px 100%, 10px 100%, 10px 100%;
    background-attachment: local, local, scroll, scroll;
}

.table-container table {
    width: auto;
    white-space: nowrap;
    border-collapse: collapse;
}

.table-header {
    background-color: var(--n-color-primary-fade-1);
    padding: 8px 12px;
    text-align: left;
    position: sticky;
    top: 0;
    font-weight: 500;
    font-size: 12px;
    z-index: 2;
    border-bottom: 1px solid var(--n-border-color);
    white-space: nowrap;
}

.table-cell {
    padding: 8px 12px;
    border-bottom: 1px solid var(--n-border-color);
    font-size: 12px;
    position: relative;
    white-space: nowrap;
}

.table-container::-webkit-scrollbar {
    height: 6px;
    width: 6px;
}

.table-container::-webkit-scrollbar-thumb {
    background-color: var(--n-scrollbar-color);
    border-radius: 6px;
}

.table-container::-webkit-scrollbar-track {
    background-color: var(--n-scrollbar-track-color);
    border-radius: 6px;
}

.scroll-hint {
    text-align: center;
    color: var(--n-text-color-3);
    font-size: 12px;
    margin-bottom: 15px;
    opacity: 0.8;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
}

.scroll-hint::before,
.scroll-hint::after {
    content: '';
    width: 16px;
    height: 1px;
    background-color: var(--n-text-color-3);
    display: inline-block;
}

.tooltip, .header-tooltip {
    display: none;
    position: absolute;
    background-color: var(--n-popover-color);
    padding: 8px 12px;
    border-radius: 4px;
    z-index: 100;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    font-size: 12px;
    top: -5px;
    left: 50%;
    transform: translateX(-50%) translateY(-100%);
    max-width: 300px;
    word-break: break-all;
    word-wrap: break-word;
    white-space: normal;
    line-height: 1.4;
    transition: opacity 0.2s ease;
    color: var(--n-text-color);
}

.header-tooltip::after, .cell-tooltip::after {
    content: '';
    position: absolute;
    bottom: -5px;
    left: 50%;
    transform: translateX(-50%);
    border-width: 5px 5px 0;
    border-style: solid;
    border-color: var(--n-popover-color) transparent transparent;
}

.table-header:hover .header-tooltip {
    display: block;
}

.table-cell:hover .cell-tooltip {
    display: block;
}

.more-data-hint {
    text-align: center;
    padding: 5px;
    color: var(--n-text-color-3);
    font-size: 12px;
    background-color: var(--n-color-primary-fade-1);
}

/* 批量结果列表 */
.batch-results-list {
    list-style-type: none;
    padding: 0;
    margin: 0;
    max-height: 300px;
    overflow-y: auto;
}

.batch-result-item {
    display: flex;
    align-items: center;
    padding: 12px;
    border-bottom: 1px solid var(--n-border-color);
    cursor: pointer;
    transition: all 0.3s;
}

.batch-result-item:hover {
    background-color: var(--n-color-primary-fade-1);
}

.batch-result-item:last-child {
    border-bottom: none;
}

/* 右侧分析结果区域 */
.result-section {
    flex: 1 1 450px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.analysis-card {
    background-color: var(--n-card-color);
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-bottom: 20px;
    height: auto;
    width: 100%;
}

.analysis-card.inactive {
    opacity: 0.7;
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

.record-count {
    font-size: 13px;
    color: var(--n-text-color-3);
    margin-left: 5px;
}

.card-content {
    color: var(--n-text-color);
    font-size: 14px;
    line-height: 1.6;
    margin-top: 5px;
}

.placeholder-text {
    color: var(--n-text-color-3);
    font-style: italic;
}

.error-text {
    color: var(--n-error-color);
    font-style: italic;
}

.result-container {
    margin-top: 10px;
}

.result-item {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
}

.result-label {
    width: 120px;
    color: var(--n-text-color-3);
}

.result-value {
    flex: 1;
    color: var(--n-text-color);
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
    margin-top: 5px;
    width: 100%;
}

.batch-results-summary {
    display: flex;
    gap: 12px;
    margin: 10px 0;
    flex-wrap: wrap;
}

.summary-item {
    background-color: var(--n-color-primary-fade-1);
    padding: 8px 12px;
    border-radius: 6px;
    min-width: 80px;
    flex: 1;
}

.summary-label {
    font-size: 12px;
    color: var(--n-text-color-3);
    margin-bottom: 3px;
}

.summary-value {
    font-size: 18px;
    font-weight: bold;
    color: var(--n-primary-color);
}

.batch-instruction {
    color: var(--n-text-color-3);
    font-size: 12px;
    font-style: italic;
    margin-top: 8px;
}

/* 诊断结果列表样式 */
.diagnosis-result-list {
    list-style-type: none;
    padding: 0;
    margin: 0;
    max-height: 450px;
    overflow-y: auto;
}

.diagnosis-result-item {
    border: 1px solid var(--n-border-color);
    border-radius: 6px;
    margin-bottom: 10px;
    overflow: hidden;
}

.result-header {
    background-color: var(--n-color-primary-fade-1);
    padding: 8px 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    position: relative; /* 确保在滚动时不会重叠 */
    z-index: 1;
}

.result-index {
    background-color: var(--n-primary-color);
    color: white;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: bold;
}

.result-id, .result-fund {
    font-size: 12px;
    color: var(--n-text-color-3);
}

.result-body {
    padding: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

.result-diagnosis {
    flex: 1;
    font-size: 13px;
    position: relative;
    padding-right: 10px;
}

.truncated-text {
    display: inline-block;
    max-width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.result-tooltip {
    display: none;
    position: absolute;
    background-color: var(--n-popover-color);
    padding: 8px 12px;
    border-radius: 4px;
    z-index: 100;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    font-size: 12px;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    margin-bottom: 5px;
    max-width: 300px;
    word-break: break-all;
    word-wrap: break-word;
    white-space: normal;
    line-height: 1.4;
    color: var(--n-text-color);
}

.result-tooltip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border-width: 5px 5px 0;
    border-style: solid;
    border-color: var(--n-popover-color) transparent transparent;
}

.result-diagnosis:hover .result-tooltip {
    display: block;
}

.view-detail-btn {
    white-space: nowrap;
}

.no-results {
    text-align: center;
    padding: 20px;
    color: var(--n-text-color-3);
}

.summary-title {
    font-size: 14px;
    margin-bottom: 10px;
    font-weight: 500;
}

/* 响应式设计 */
@media screen and (max-width: 1200px) {
    .diagnosis-container {
        flex-direction: column;
    }

    .upload-section, 
    .result-section {
        width: 100%;
        max-width: 100%;
    }
    
    .table-container {
        max-height: 150px;
    }
    
    .diagnosis-result-list {
        max-height: 300px;
    }
}
</style>