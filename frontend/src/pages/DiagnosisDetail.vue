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
    PrintOutline
    } from '@vicons/ionicons5'
    import { NIcon, NCard, NButton, NSpin, NImage, NTag, NProgress } from 'naive-ui'
import axios from 'axios'

    const route = useRoute()
    const router = useRouter()
    const message = useMessage()

    const diagnosisId = route.params.id
    console.log('diagnosisId:', diagnosisId);
    axios.post('http://127.0.0.1:5000/get_patient_record', { id: diagnosisId }).then(res => {
        console.log('res:', res);
    }).catch(err => {
        console.log('err:', err);
    })
</script>

<template>
    <div>
        <h1>诊断详情</h1>
    </div>
</template>
