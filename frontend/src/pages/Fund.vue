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
    
    class fund {
        fund_id: string
        left_fund: string
        left_fund_keyword: string
        right_fund: string
        right_fund_keyword: string
        patient_id: string
        constructor(fund_id: string, left_fund: string, left_fund_keyword: string, right_fund: string, right_fund_keyword: string, patient_id: string) {
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
        patient_age: string
        patient_gender: string
        constructor(patient_id: string,patient_name: string,    patient_age:string , patient_gender: string){
            this.patient_id = patient_id
            this.patient_name = patient_name
            this.patient_age = patient_age
            this.patient_gender = patient_gender
        }
    }

    class record {
        diagnosis_date: string
        fund_id: string
        patient_id: string
        record_id: string
        result: string
        suggestion: string
        user_id : string
        constructor(diagnosis_date: string, fund_id: string,  patient_id: string, record_id: string, result: string, suggestion: string, user_id: string) {
            this.diagnosis_date = diagnosis_date
            this.fund_id = fund_id
            this.patient_id = patient_id
            this.record_id = record_id
            this.result = result
            this.suggestion = suggestion
            this.user_id = user_id
        }
    }
    class Fund {
        fund: fund
        patient: patient
        record: record
        constructor(fund: fund, patient: patient, record: record) {
            this.fund = fund
            this.patient = patient
            this.record = record
        }
    }
    const doctor_name = ref('张医生')
    const patient_fund = ref<Fund>(new Fund(new fund('', '', '', '', '', ''), new patient('', '', '', ''), new record('', '', '', '', '', '', '')))
    onMounted(() => { 
        const FundId = route.params.id
        console.log('FundId:', FundId);
        loading.value = true
        axios.post('http://127.0.0.1:5000/get_fund_infoX', { fund_id: FundId }).then(res => {
            console.log('res:', res);
            patient_fund.value.fund = res.data.fund
            patient_fund.value.patient = res.data.patient
            patient_fund.value.record = res.data.records[0]
            loading.value = false
            patient_fund.value.fund.left_fund = 'data:image/jpeg;base64,'+patient_fund.value.fund.left_fund
            patient_fund.value.fund.right_fund = 'data:image/jpeg;base64,'+patient_fund.value.fund.right_fund
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
        axios.post('http://127.0.0.1:5000/create_pdf', {
            data:{
                doctor_name: doctor_name.value,
                patient_id: patient_fund.value.patient.patient_id,
                patient_name: patient_fund.value.patient.patient_name,
                patient_age: patient_fund.value.patient.patient_age,
                patient_gender: patient_fund.value.patient.patient_gender,
                left_eye_keywords: patient_fund.value.fund.left_fund_keyword,
                right_eye_keywords: patient_fund.value.fund.right_fund_keyword,
                left_eye_image: patient_fund.value.fund.left_fund,
                right_eye_image: patient_fund.value.fund.right_fund,
                diagnosis: patient_fund.value.record.result,
                medication: patient_fund.value.record.suggestion,
            }
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
            link.download = `${patient_fund.value.patient.patient_name}_病例报告.pdf`
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
                                <NInput v-model:value="patient_fund.patient.patient_id" placeholder="请输入患者ID" />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="8">
                            <NFormItem label="患者姓名">
                                <NInput v-model:value="patient_fund.patient.patient_name" placeholder="请输入患者姓名" />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="4">
                            <NFormItem label="患者年龄">
                                <NInput v-model:value="patient_fund.patient.patient_age" placeholder="年龄" />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="4">
                            <NFormItem label="患者性别">
                                <NRadioGroup v-model:value="patient_fund.patient.patient_gender">
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
                                        v-if="patient_fund.fund.left_fund" 
                                        :src="patient_fund.fund.left_fund" 
                                        object-fit="contain"
                                        :preview-disabled="false"
                                        width="100%"
                                    />
                                    <div v-else class="no-image">暂无左眼图像</div>
                                </div>
                                <NFormItem label="左眼诊断关键词">
                                    <NInput
                                        v-model:value="patient_fund.fund.left_fund_keyword" 
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
                                        v-if="patient_fund.fund.right_fund" 
                                        :src="patient_fund.fund.right_fund" 
                                        object-fit="contain"
                                        :preview-disabled="false"
                                        width="100%"
                                    />
                                    <div v-else class="no-image">暂无右眼图像</div>
                                </div>
                                <NFormItem label="右眼诊断关键词">
                                    <NInput 
                                        v-model:value="patient_fund.fund.right_fund_keyword" 
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
                            <NFormItem label="诊断日期">
                                <NInput 
                                    v-model:value="patient_fund.record.diagnosis_date" 
                                    type="text"
                                    placeholder="请输入诊断日期（YYYY-MM-DD）"
                                />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="24">
                            <NFormItem label="诊断结果">
                                <NInput 
                                    v-model:value="patient_fund.record.result" 
                                    placeholder="请输入诊断结果"
                                    type="textarea"
                                    :autosize="{ minRows: 3, maxRows: 6 }"
                                />
                            </NFormItem>
                        </NGridItem>
                        <NGridItem :span="24">
                            <NFormItem label="治疗建议">
                                <NInput 
                                    v-model:value="patient_fund.record.suggestion" 
                                    placeholder="请输入治疗建议、用药建议等"
                                    type="textarea"
                                    :autosize="{ minRows: 10, maxRows: 20 }"
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
