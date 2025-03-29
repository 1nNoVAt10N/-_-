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
    SaveOutline,
    DocumentTextOutline,
    CloseCircleOutline
    } from '@vicons/ionicons5'
    import { NIcon, NCard, NButton, NSpin, NImage, NTag, NProgress, NForm, NFormItem, NInput, NSelect, NRadioGroup, NRadio, NSpace, NModal, NDivider, NSkeleton, NGrid, NGridItem, NDatePicker, NPageHeader } from 'naive-ui'
    import axios from 'axios'

    const route = useRoute()
    const router = useRouter()
    const message = useMessage()
    
    const loading = ref(true)
    const submitting = ref(false)
    const isPdfReady = ref(false)
    const pdfUrl = ref('')
    const showPdfModal = ref(false)
    
    class Fund {
        diagnosis: string
        doctor_name: string
        examinations: string
        left_eye_image: string
        left_eye_keywords: string
        medication: string
        patient_age: string
        patient_gender: string
        patient_id: string
        patient_name: string
        precautions: string
        right_eye_image: string
        right_eye_keywords: string
        constructor(diagnosis: string,
        doctor_name: string,
        examinations: string,
        left_eye_image: string,
        left_eye_keywords: string,
        medication: string,
        patient_age: string,
        patient_gender: string,
        patient_id: string,
        patient_name: string,
        precautions: string,
        right_eye_image: string,
        right_eye_keywords: string){
            this.diagnosis = diagnosis
            this.doctor_name = doctor_name
            this.examinations = examinations
            this.left_eye_image = left_eye_image
            this.left_eye_keywords = left_eye_keywords
            this.medication = medication
            this.patient_age = patient_age
            this.patient_gender = patient_gender
            this.patient_id = patient_id
            this.patient_name = patient_name
            this.precautions = precautions
            this.right_eye_image = right_eye_image
            this.right_eye_keywords = right_eye_keywords
        }
    }
    const doctor_name = ref('张医生')
    const patient_fund = ref<Fund>(new Fund('', '', '', '', '', '','', '', '', '', '', '', ''))
    onMounted(() => { 
        const FundId = route.params.id
        console.log('FundId:', FundId);
        loading.value = true
        axios.post('http://127.0.0.1:5000/get_fund_infoX', { fund_id: FundId }).then(res => {
            console.log('res:', res);
            patient_fund.value = res.data
            loading.value = false
        }).catch(err => {
            console.log('err:', err);
            message.error('获取病例信息失败')
            loading.value = false
        })
    })

    const submitForm = () => {
        message.success('病例信息保存成功')
    }

    const summonPDF = () => {
        submitting.value = true
        patient_fund.value.patient_age = patient_fund.value.patient_age.toString()
        patient_fund.value.patient_id = patient_fund.value.patient_id.toString()
        axios.post('http://127.0.0.1:5000/create_pdf', {
            data: patient_fund.value
        }, {
            responseType: 'blob' // 重要：设置响应类型为blob
        }).then(res => {
            // 创建一个Blob对象
            const blob = new Blob([res.data], { type: 'application/pdf' })
            // 创建URL
            const url = window.URL.createObjectURL(blob)
            pdfUrl.value = url
            isPdfReady.value = true
            showPdfModal.value = true
            submitting.value = false
            message.success('PDF生成成功')
        }).catch(err => {
            console.log('err:', err)
            message.error('PDF生成失败')
            submitting.value = false
        })
    }

    const downloadPdf = () => {
        if (isPdfReady.value) {
            const link = document.createElement('a')
            link.href = pdfUrl.value
            link.download = `${patient_fund.value.patient_id}_病例报告.pdf`
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
        }
    }

    const goBack = () => {
        router.go(-1)
    }
</script>

<template>
    <div class="fund-page">
        <NPageHeader title="病例详情" @back="goBack">
            <template #avatar>
                <NIcon>
                    <MedicalOutline />
                </NIcon>
            </template>
            <template #extra>
                <NSpace>
                    <NButton type="primary" @click="submitForm" :loading="submitting">
                        <template #icon>
                            <NIcon>
                                <SaveOutline />
                            </NIcon>
                        </template>
                        保存修改
                    </NButton>
                    <NButton type="success" @click="summonPDF" :loading="submitting" :disabled="loading">
                        <template #icon>
                            <NIcon>
                                <DocumentTextOutline />
                            </NIcon>
                        </template>
                        生成PDF
                    </NButton>
                </NSpace>
            </template>
        </NPageHeader>

        <NSpin :show="loading">
            <NSkeleton v-if="loading" text :repeat="10" />
            <div v-else class="form-container">
                <NCard title="患者基本信息" size="large">
                    <NGrid :cols="24" :x-gap="24">
                        <NGridItem :span="8">
                            <NFormItem label="患者ID">
                                <NInput v-model:value="patient_fund.patient_id" placeholder="请输入患者ID" />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="8">
                            <NFormItem label="患者姓名">
                                <NInput v-model:value="patient_fund.patient_name" placeholder="请输入患者姓名" />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="4">
                            <NFormItem label="患者年龄">
                                <NInput v-model:value="patient_fund.patient_age" placeholder="年龄" />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="4">
                            <NFormItem label="患者性别">
                                <NRadioGroup v-model:value="patient_fund.patient_gender">
                                    <NRadio value="Male">男</NRadio>
                                    <NRadio value="Female">女</NRadio>
                                </NRadioGroup>
                            </NFormItem>
                        </NGridItem>
                    </NGrid>
                </NCard>

                <NDivider />

                <NCard title="眼底检查结果" size="large">
                    <NGrid :cols="24" :x-gap="24">
                        <NGridItem :span="12">
                            <NCard title="左眼" embedded>
                                <div class="eye-image-container">
                                    <NImage 
                                        v-if="patient_fund.left_eye_image" 
                                        :src="'data:image/jpeg;base64,'+patient_fund.left_eye_image" 
                                        object-fit="contain"
                                        :preview-disabled="false"
                                        width="100%"
                                    />
                                    <div v-else class="no-image">暂无左眼图像</div>
                                </div>
                                <NFormItem label="左眼诊断关键词">
                                    <NInput
                                        v-model:value="patient_fund.left_eye_keywords" 
                                        placeholder="请输入左眼诊断关键词"
                                        type="textarea"
                                        :autosize="{ minRows: 2, maxRows: 5 }"
                                    />
                                </NFormItem>
                            </NCard>
                        </NGridItem>
                        <NGridItem :span="12">
                            <NCard title="右眼" embedded>
                                <div class="eye-image-container">
                                    <NImage 
                                        v-if="patient_fund.right_eye_image" 
                                        :src="'data:image/jpeg;base64,'+patient_fund.right_eye_image" 
                                        object-fit="contain"
                                        :preview-disabled="false"
                                        width="100%"
                                    />
                                    <div v-else class="no-image">暂无右眼图像</div>
                                </div>
                                <NFormItem label="右眼诊断关键词">
                                    <NInput 
                                        v-model:value="patient_fund.right_eye_keywords" 
                                        placeholder="请输入右眼诊断关键词"
                                        type="textarea"
                                        :autosize="{ minRows: 2, maxRows: 5 }"
                                    />
                                </NFormItem>
                            </NCard>
                        </NGridItem>
                    </NGrid>
                </NCard>

                <NDivider />

                <NCard title="诊断结果与建议" size="large">
                    <NGrid :cols="24" :x-gap="24">
                        <NGridItem :span="24">
                            <NFormItem label="医生姓名">
                                <NInput 
                                    v-model:value="doctor_name" 
                                    type="text"
                                    placeholder="请输入医生姓名"
                                />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="24">
                            <NFormItem label="诊断结果">
                                <NInput 
                                    v-model:value="patient_fund.diagnosis" 
                                    placeholder="请输入诊断结果"
                                    type="textarea"
                                    :autosize="{ minRows: 3, maxRows: 6 }"
                                />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="24">
                            <NFormItem label="检查建议">
                                <NInput 
                                    v-model:value="patient_fund.examinations" 
                                    placeholder="请输入检查建议"
                                    type="textarea"
                                    :autosize="{ minRows: 3, maxRows: 6 }"
                                />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="24">
                            <NFormItem label="用药建议">
                                <NInput 
                                    v-model:value="patient_fund.medication"
                                    placeholder="请输入用药建议"
                                    type="textarea"
                                    :autosize="{ minRows: 3, maxRows: 6 }"
                                />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="24">
                            <NFormItem label="预防措施">
                                <NInput 
                                    v-model:value="patient_fund.precautions"
                                    placeholder="请输入预防措施"
                                    type="textarea"
                                    :autosize="{ minRows: 3, maxRows: 6 }"
                                />
                            </NFormItem>
                        </NGridItem>
                    </NGrid>
                </NCard>
            </div>
        </NSpin>

        <NModal v-model:show="showPdfModal" preset="card" title="PDF预览与下载" style="width: 600px">
            <template #header-extra>
                <NButton quaternary circle @click="showPdfModal = false">
                    <NIcon>
                        <CloseCircleOutline />
                    </NIcon>
                </NButton>
            </template>
            <div class="pdf-container">
                <p>PDF已生成，您可以直接下载或者预览。</p>
                <div class="pdf-actions">
                    <NButton type="primary" @click="downloadPdf" :disabled="!isPdfReady">
                        <template #icon>
                            <NIcon>
                                <PrintOutline />
                            </NIcon>
                        </template>
                        下载PDF
                    </NButton>
                    <NButton type="info" tag="a" :href="pdfUrl" target="_blank" :disabled="!isPdfReady">
                        <template #icon>
                            <NIcon>
                                <EyeOutline />
                            </NIcon>
                        </template>
                        预览PDF
                    </NButton>
                </div>
            </div>
        </NModal>
    </div>
</template>

<style scoped>
.fund-page {
    padding: 20px;
}

.form-container {
    margin-top: 20px;
}

.eye-image-container {
    height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px dashed #ccc;
    border-radius: 8px;
    margin-bottom: 16px;
}

.no-image {
    color: #999;
    font-size: 14px;
}

.pdf-container {
    padding: 20px;
    text-align: center;
}

.pdf-actions {
    margin-top: 20px;
    display: flex;
    justify-content: center;
    gap: 16px;
}
</style>
