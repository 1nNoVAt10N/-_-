<script setup lang="ts">
import { ref, onMounted } from 'vue';
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
} from '@vicons/ionicons5';
import { NIcon, NCard, NButton, NSpin, NUpload, NImage, NTag, NProgress } from 'naive-ui';

const router = useRouter();
const message = useMessage();
const dialog = useDialog();
const backendAddr = import.meta.env.VITE_API_URL;

const leftFileInput = ref<HTMLInputElement | null>(null);
const rightFileInput = ref<HTMLInputElement | null>(null);
const leftPreviewImage = ref('');
const rightPreviewImage = ref('');
const leftPreviewVisible = ref(false);
const rightPreviewVisible = ref(false);
const analyzeBtn = ref(false);
const leftFile = ref<any>(false);
const rightFile = ref<any>(false);
const isAnalyzing = ref(false);
const analysisCompleted = ref(false);

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
// 处理拖拽上传
const handleDragOver = (e: any, side: any) => {
    e.preventDefault();
    const uploadArea = document.querySelector(`.upload-area.${side}`);
    if (uploadArea) {
        uploadArea.classList.add('drag-over');
    }
};
const handleDragLeave = (side: any) => {
    const uploadArea = document.querySelector(`.upload-area.${side}`);
    if (uploadArea) {
        uploadArea.classList.remove('drag-over');
    }
};

const handleDrop = (e: any, side: any) => {
    e.preventDefault();
    const uploadArea = document.querySelector(`.upload-area.${side}`);
    if (uploadArea) {
        uploadArea.classList.remove('drag-over');
    }

    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0], side);
    }
};

// 文件处理
const handleFile = (file: any, side: any) => {
    if (!file.type.match('image.*')) {
        alert('请上传图片文件！');
        return;
    }

    if (side === 'left') {
        leftFile.value = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            leftPreviewImage.value = e.target?.result?.toString() || '';
            leftPreviewVisible.value = true;
            checkAnalyzeButton();
        };
        reader.readAsDataURL(file);
    } else {
        rightFile.value = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            rightPreviewImage.value = e.target?.result?.toString() || '';
            rightPreviewVisible.value = true;
            checkAnalyzeButton();
        };
        reader.readAsDataURL(file);
    }
};

// 文件上传处理
const handleLeftFileUpload = (event: any) => {
    const file = event.target.files[0];
    if (file) {
        handleFile(file, 'left');
    }
};
const handleRightFileUpload = (event: any) => {
    const file = event.target.files[0];
    if (file) {
        handleFile(file, 'right');
    }
};
// 检查是否可以开始分析
const checkAnalyzeButton = () => {
    analyzeBtn.value = leftFile.value && rightFile.value;
};
const merged = ref('');
const left_eye_x1 = ref('');
const left_eye_x2 = ref('');
const right_eye_y1 = ref('');
const right_eye_y2 = ref('');

// 医生输入
const left_eye_text = ref('');
const right_eye_text = ref('');
// 不允许text为中文
watch([left_eye_text, right_eye_text], ([left, right]) => {
    if (left.match(/[\u4e00-\u9fa5]/) || right.match(/[\u4e00-\u9fa5]/)) {
        message.error('不允许输入中文');
        left_eye_text.value = '';
        right_eye_text.value = '';
    }
});
const doctorNotes = ref('');
const patientId = ref('');
const patientName = ref('张三');
const patientGender = ref('');
const patientAge = ref(0);
// 分析处理
const startAnalysis = async () => {
    // 检查是否可以开始分析
    if (!analyzeBtn.value) {
        message.error('请先上传左右眼图像');
        return;
    }
    if (patientId.value === '') {
        message.error('请输入患者ID');
        return;
    }
    if (patientGender.value === '') {
        message.error('请选择患者性别');
        return;
    }
    if (patientAge.value === 0) {
        message.error('请输入患者年龄');
        return;
    }
    if (left_eye_text.value === '' || right_eye_text.value === '') {
        message.error('请输入左右眼疾病预诊断关键词');
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
        formData.append('left_eye', leftFile.value instanceof Blob ? leftFile.value : '');
        formData.append('right_eye', rightFile.value instanceof Blob ? rightFile.value : '');
        formData.append('left_eye_text', left_eye_text.value);
        formData.append('right_eye_text', right_eye_text.value);
        formData.append('patientId', patientId.value);
        formData.append('patientName', patientName.value);
        formData.append('patientGender', patientGender.value);
        formData.append('patientAge', String(patientAge.value));
        // 打印 FormData 内容
        for (let [key, value] of formData.entries()) {
            console.log(`${key}: ${value}`);
        }
        const { json } = await lyla.post('http://localhost:5000/predict', {
            body: formData,
            onUploadProgress: ({ percent }) => {
                console.log('上传进度:', Math.ceil(percent));
            },
        });


        // 更新检测卡片状态
        detectionCard.value.status = 'completed';
        detectionCard.value.statusText = '分析完成';

        // 处理返回结果
        const results: any[] = [];
        // for (const [key, value] of Object.entries(json)) {
        //     if (Array.isArray(value)) {
        //         console.log('检测结果:', key, value);
        //         results.push({
        //             name: value[0],
        //             details: value[1],
        //             // isPositive: value[2] === 'positive',
        //             isPositive: true,
        //             confidence: 80,
        //         });
        //     }
        // }
        console.log('检测结果:', json);

        const patientIll = ref('');
        const patientNameX = ref('');
        const patientAgeX = ref('');
        const patientGenderX = ref('');
        const patientIllness = ref('');
        const patientRecommendMedicine = ref('');
        const patientRecommendCheck = ref('');
        const patientAttention = ref('');

        patientIll.value = json['result'][0];
        patientNameX.value = json['result'][1]["患者名称"];
        patientAgeX.value = json['result'][1]["患者年龄"];
        patientGenderX.value = json['result'][1]["患者性别"];
        patientIllness.value = json['result'][1]["病情"];
        patientRecommendMedicine.value = json['result'][1]["建议用药"];
        patientRecommendCheck.value = json['result'][1]["建议检查项目"];
        patientAttention.value = json['result'][1]["注意事项"];
        results.push({
            name: json['result'][0],
        });
        detectionCard.value.results = results;
        merged.value = 'data:image/jpeg;base64,' + json['merged_base64'];
        left_eye_x1.value = 'data:image/jpeg;base64,' + json['left_eye_x1'];
        left_eye_x2.value = 'data:image/jpeg;base64,' + json['left_eye_x2'];
        right_eye_y1.value = 'data:image/jpeg;base64,' + json['right_eye_y1'];
        right_eye_y2.value = 'data:image/jpeg;base64,' + json['right_eye_y2'];
        // console.log('检测结果:', json);

        // 更新诊断卡片状态
        diagnosisCard.value.status = 'completed';
        diagnosisCard.value.statusText = '分析完成';

        // 根据检测结果生成问题和建议
        const problems = [];
        const recommendations = [];

        // 处理每个检测结果
        results.forEach((result) => {
            if (result.isPositive) {
                problems.push(`检测到${result.name}建议及时就医检查`);
                recommendations.push(`针对${result.name}进行专业治疗`);
            }
        });

        problems.push(patientIll.value);
        recommendations.push(`患者姓名：${patientNameX.value}`);
        recommendations.push(`患者年龄：${patientAgeX.value}`);
        recommendations.push(`患者性别：${patientGenderX.value}`);
        recommendations.push(`病情：${patientIllness.value}`);
        recommendations.push(`建议用药：${patientRecommendMedicine.value}`);
        recommendations.push(`建议检查项目：${patientRecommendCheck.value}`);
        recommendations.push(`注意事项：${patientAttention.value}`);

        diagnosisCard.value.content = {
            problems,
            recommendations,
        };

        // 更新状态
        analysisCompleted.value = true;

        // 重新加载历史记录
        // loadHistoryRecords()
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
// 查看详细报告
const viewDetailReport = () => {

    router.push(`/diagnosis/${patientId.value}?type=new`);
};
const goBack = () => {
    router.go(-1);
};
</script>

<template>
    <div class="diagnosis-page">
        <!-- 面包屑导航 -->
        <NPageHeader title="眼底影像智能诊断" @back="goBack">
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
                    上传眼底图像
                </h2>

                <!-- 左眼预览区域 -->
                <NCard class="analysis-card">
                    <div class="eye-sections">
                        <!-- 左眼上传区域 -->
                        <div class="eye-section">
                            <h3 class="eye-title">左眼图像</h3>
                            <div v-if="leftPreviewImage == ''" class="upload-area left" @click="leftFileInput?.click()"
                                @dragover="(e) => handleDragOver(e, 'left')" @dragleave="() => handleDragLeave('left')"
                                @drop="(e) => handleDrop(e, 'left')">
                                <div class="upload-icon">
                                    <NIcon size="48">
                                        <CloudUploadOutline />
                                    </NIcon>
                                </div>
                                <div class="upload-text">点击或拖拽文件到此区域上传</div>
                                <div class="upload-hint">支持 JPG、PNG、TIFF 格式，单个文件不超过20MB</div>
                                <NButton type="primary">选择文件</NButton>
                                <input type="file" ref="leftFileInput" style="display: none" accept="image/*"
                                    @change="handleLeftFileUpload" />
                            </div>
                            <div class="preview-area">
                                <div v-if="!leftPreviewVisible" class="preview-placeholder"></div>
                                <NImage v-if="leftPreviewVisible" :src="leftPreviewImage" class="preview-image show"
                                    alt="左眼眼底图像预览" preview-disabled />
                            </div>
                            <div v-if="detectionCard.status !== 'completed'" class="card-title">
                                <NIcon size="20" class="mr-2">
                                    <CheckmarkCircleOutline />
                                </NIcon>
                                左眼眼部疾病预诊断关键词
                            </div>
                            <div v-if="detectionCard.status !== 'completed'">
                                <NInput v-model:value="left_eye_text" type="textarea" placeholder="左眼眼部疾病预诊断关键词"
                                    rows="4" />
                            </div>
                        </div>
                        <!-- 右眼上传区域 -->
                        <div class="eye-section">
                            <h3 class="eye-title">右眼图像</h3>
                            <div v-if="rightPreviewImage == ''" class="upload-area right"
                                @click="rightFileInput?.click()" @dragover="(e) => handleDragOver(e, 'right')"
                                @dragleave="() => handleDragLeave('right')" @drop="(e) => handleDrop(e, 'right')">
                                <div class="upload-icon">
                                    <NIcon size="48">
                                        <CloudUploadOutline />
                                    </NIcon>
                                </div>
                                <div class="upload-text">点击或拖拽文件到此区域上传</div>
                                <div class="upload-hint">支持 JPG、PNG、TIFF 格式，单个文件不超过20MB</div>
                                <NButton type="primary">选择文件</NButton>
                                <input type="file" ref="rightFileInput" style="display: none" accept="image/*"
                                    @change="handleRightFileUpload" />
                            </div>
                            <div class="preview-area">
                                <div v-if="!rightPreviewVisible" class="preview-placeholder"></div>
                                <NImage v-if="rightPreviewVisible" :src="rightPreviewImage" class="preview-image show"
                                    alt="右眼眼底图像预览" preview-disabled />
                            </div>
                            <div v-if="detectionCard.status !== 'completed'" class="card-title">
                                <NIcon size="20" class="mr-2">
                                    <CheckmarkCircleOutline />
                                </NIcon>
                                右眼眼部疾病预诊断关键词
                            </div>
                            <div v-if="detectionCard.status !== 'completed'">
                                <NInput v-model:value="right_eye_text" type="textarea" placeholder="右眼眼部疾病预诊断关键词"
                                    rows="4" />
                            </div>
                        </div>
                    </div>
                </NCard>
                <NCard v-if="detectionCard.status === 'completed'" class="analysis-card">
                    <h2 class="section-title">
                        <NIcon size="20" class="mr-2">
                            <ImageOutline />
                        </NIcon>
                        血管分割图像
                    </h2>
                    <div class="eye-sections">
                        <div class="preview-area">
                            <div v-if="!leftPreviewVisible" class="preview-placeholder">左眼眼底预处理图像预览区域1</div>
                            <NImage v-if="leftPreviewVisible" :src="left_eye_x1" class="preview-image show"
                                alt="左眼眼底预处理图像预览1" preview-disabled />
                        </div>
                        <div class="preview-area">
                            <div v-if="!rightPreviewVisible" class="preview-placeholder">右眼眼底预处理图像预览区域1</div>
                            <NImage v-if="rightPreviewVisible" :src="right_eye_y1" class="preview-image show"
                                alt="右眼眼底预处理图像预览1" preview-disabled />
                        </div>
                    </div>
                </NCard>
                <NCard v-if="detectionCard.status === 'completed'" class="analysis-card">
                    <h2 class="section-title">
                        <NIcon size="20" class="mr-2">
                            <ImageOutline />
                        </NIcon>
                        血管映射图像
                    </h2>
                    <div class="eye-sections">
                        <div class="preview-area">
                            <div v-if="!leftPreviewVisible" class="preview-placeholder">左眼眼底预处理图像预览区域2</div>
                            <NImage v-if="leftPreviewVisible" :src="left_eye_x2" class="preview-image show"
                                alt="左眼眼底预处理图像预览2" preview-disabled />
                        </div>
                        <div class="preview-area">
                            <div v-if="!rightPreviewVisible" class="preview-placeholder">右眼眼底预处理图像预览区域2</div>
                            <NImage v-if="rightPreviewVisible" :src="right_eye_y2" class="preview-image show"
                                alt="右眼眼底预处理图像预览2" preview-disabled />
                        </div>
                    </div>
                </NCard>
            </div>
            <!-- 右侧分析结果区域 -->
            <div class="result-section">
                <NForm v-if="detectionCard.status !== 'completed'">
                    <NFormItem label="患者ID">
                        <NInput v-model:value="patientId" placeholder="请输入患者ID" />
                    </NFormItem>
                    <NFormItem label="患者姓名">
                        <NInput v-model:value="patientName" placeholder="张三" disabled />
                    </NFormItem>
                    <NFormItem label="患者性别">
                        <NRadioGroup v-model:value="patientGender">
                            <NRadio value="Male">男</NRadio>
                            <NRadio value="Female">女</NRadio>
                        </NRadioGroup>
                    </NFormItem>
                    <NFormItem label="患者年龄">
                        <NInputNumber v-model:value="patientAge" placeholder="请输入患者年龄" />
                    </NFormItem>
                    <NFormItem label="备注内容">
                        <NInput v-model:value="doctorNotes" type="textarea" placeholder="请输入医生备注..." rows="4" />
                    </NFormItem>
                </NForm>
                <!-- 分析结果卡片 -->
                <NCard :class="{ inactive: !detectionCard.isActive }" class="analysis-card">
                    <template #header>
                        <div class="card-header">
                            <div class="card-title">
                                <NIcon size="20" class="mr-2">
                                    <SearchOutline />
                                </NIcon>
                                眼底病变检测
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
                                    ? '请先上传眼底图像并点击"开始分析"按钮'
                                    : '正在分析中，请稍候...'
                            }}
                        </p>
                        <div v-if="detectionCard.status === 'completed'" class="analysis-card">
                            <h2 class="section-title">
                                <NIcon size="20" class="mr-2">
                                    <ImageOutline />
                                </NIcon>
                                预处理图像
                            </h2>
                            <div class="eye-sections">
                                <div class="preview-area">
                                    <div v-if="!leftPreviewVisible" class="preview-placeholder">左眼眼底预处理图像预览区域</div>
                                    <NImage v-if="leftPreviewVisible" :src="merged" class="preview-image1 show"
                                        alt="合并预处理图像预览" preview-disabled />
                                </div>
                            </div>
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
                                    <MedicalOutline />
                                </NIcon>
                                诊断建议
                            </div>
                            <NTag :type="diagnosisCard.status === 'completed' ? 'success' :
                                diagnosisCard.status === 'analyzing' ? 'warning' :
                                    diagnosisCard.status === 'error' ? 'error' : 'info'">
                                {{ diagnosisCard.statusText }}
                            </NTag>
                        </div>
                    </template>
                    <div class="card-content">
                        <p v-if="diagnosisCard.status === 'waiting' || diagnosisCard.status === 'analyzing'"
                            class="placeholder-text">
                            {{ diagnosisCard.status === 'waiting' ? '分析完成后将显示诊断建议' : '正在生成诊断建议，请稍候...' }}
                        </p>
                        <div v-if="diagnosisCard.status === 'completed'" class="diagnosis-content">
                            <p>基于AI分析，发现以下问题：</p>
                            <ol style="margin-left: 20px; margin-top: 10px;">
                                <li v-for="(problem, index) in diagnosisCard.content.problems" :key="index">
                                    <span v-html="problem"></span>
                                </li>
                            </ol>
                            <p style="margin-top: 15px;"><strong>建议：</strong></p>
                            <ul style="margin-left: 20px; margin-top: 5px;">
                                <li v-for="(rec, index) in diagnosisCard.content.recommendations" :key="index">
                                    {{ rec }}
                                </li>
                            </ul>
                        </div>
                        <p v-if="diagnosisCard.status === 'error'" class="error-text">
                            诊断建议生成失败，请重试
                        </p>
                    </div>
                </NCard>
                <!-- 分析按钮 -->
                <NButton type="primary" size="large" block :disabled="!analyzeBtn || isAnalyzing"
                    @click="analysisCompleted ? viewDetailReport() : startAnalysis()" :loading="isAnalyzing">
                    {{ isAnalyzing ? '分析中...' : analysisCompleted ? '查看详细报告' : '开始分析' }}
                </NButton>
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
}

/* 左侧上传和预览区域 */
.upload-section {
    flex: 1;
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

.eye-sections {
    display: flex;
    gap: 20px;
    justify-content: space-between;
}

.eye-section {
    flex: 1;
}

.eye-title {
    font-size: 16px;
    color: var(--n-text-color);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
}

.upload-area.left,
.upload-area.right {
    border: 2px dashed var(--n-border-color);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    margin-bottom: 10px;
    cursor: pointer;
    transition: all 0.3s;
    background-color: var(--n-card-color);
}

.upload-area.left:hover,
.upload-area.left.drag-over,
.upload-area.right:hover,
.upload-area.right.drag-over {
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

.preview-area {
    width: auto;
    height: auto;
    overflow: hidden;
    border-radius: 8px;
    position: relative;
    background-color: var(--n-fill-color);
    display: flex;
    align-items: center;
    justify-content: center;
}
.preview-area1 {
    width: 200%;
    height: 200%;
    overflow: hidden;
    border-radius: 8px;
    position: relative;
    background-color: var(--n-fill-color);
    display: flex;
    align-items: center;
    justify-content: center;
}
.preview-placeholder {
    color: var(--n-text-color-3);
    font-size: 14px;
}

.preview-image {
    max-width: 300px;
    /* 设置图片的最大宽度 */
    max-height: 300px;
    /* 设置图片的最大高度 */
    display: none;
    object-fit: contain;
    /* 确保图片等比例缩小 */
}
.preview-image1 {
    max-width: 600px;
    /* 设置图片的最大宽度 */
    max-height: 600px;
    /* 设置图片的最大高度 */
    display: none;
    object-fit: contain;
    /* 确保图片等比例缩小 */
}

.preview-image.show {
    display: block;
}
.preview-image1.show {
    display: block;
}
/* 右侧分析结果区域 */
.result-section {
    flex: 1;
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
    /* 设置卡片的高度为自动 */
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

.card-content {
    color: var(--n-text-color);
    font-size: 14px;
    line-height: 1.6;
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
    margin-top: 15px;
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

/* 历史记录区域 */
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
    width: 60px;
    height: 60px;
    border-radius: 4px;
    overflow: hidden;
    margin-right: 15px;
    background-color: var(--n-fill-color);
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
@media screen and (max-width: 992px) {
    .diagnosis-container {
        flex-direction: column;
    }

    .eye-sections {
        flex-direction: column;
    }
}

/* 医生备注 */
.notes-content {
    padding: 10px 0;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    background-color: var(--n-card-color);
}

.notes-textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid var(--n-border-color);
    border-radius: 4px;
    background-color: var(--n-input-color);
    color: var(--n-text-color);
    resize: vertical;

    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.notes-textarea:focus {
    outline: none;
    border-color: var(--n-primary-color);
}
</style>
