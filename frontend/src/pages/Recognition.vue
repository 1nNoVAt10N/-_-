<script setup lang="ts">
import { ref } from 'vue';
import { Photo, HandRock, Refresh, FaceIdError } from '@vicons/tabler';
import type { UploadCustomRequestOptions, UploadFileInfo } from 'naive-ui';
import { lyla } from '@lylajs/web';
import { useMessage } from 'naive-ui';
const message = useMessage();
const current = ref(1);
const backendAddr = import.meta.env.VITE_API_URL;

interface PredictResult {
  label: string;
  confidence: number;
}

const result = ref<PredictResult[] | null>(null);
const leftFile = ref<File | null>(null);
const rightFile = ref<File | null>(null);

const uploadRequest = ({
  file,
  data,
  headers,
  withCredentials,
  action,
  onFinish,
  onError,
  onProgress,
}: UploadCustomRequestOptions) => {
  const formData = new FormData();
  if (data) {
    Object.keys(data).forEach((key) => {
      formData.append(key, data[key as keyof UploadCustomRequestOptions['data']]);
    });
  }
  if (leftFile.value && rightFile.value) {
    formData.append('left_eye', leftFile.value);
    formData.append('right_eye', rightFile.value);
    current.value = 2;
    lyla
      .post(action as string, {
        withCredentials,
        headers: headers as Record<string, string>,
        body: formData,
        onUploadProgress: ({ percent }) => {
          onProgress({ percent: Math.ceil(percent) });
        },
      })
      .then(({ json }) => {
        result.value = json;
        current.value = 3;
        onFinish();
      })
      .catch((error) => {
        message.error(error.message);
        message.error('哦不，好像有点问题，请刷新重试！');
        onError();
      });
  } else {
    message.error('请上传两张图片');
  }
};

const handleLeftFileChange = (info: { file: UploadFileInfo }) => {
  leftFile.value = info.file.file as File;
};

const handleRightFileChange = (info: { file: UploadFileInfo }) => {
  rightFile.value = info.file.file as File;
};

const restart = () => {
  current.value = 1;
  result.value = null;
  leftFile.value = null;
  rightFile.value = null;
};
</script>

<template>
  <div class="flex flex-col w-full">
    <div class="mx-10 flex items-start justify-center mt-10 w-full">
      <n-steps :current="current" class="w-full" status="process">
        <n-step title="上传照片" description="你得告诉我什么样" />
        <n-step title="等等" description="等等我们的AI" />
        <n-step title="所以你得了什么病" description="让我告诉你" />
      </n-steps>
    </div>
    <div class="mt-5 w-full flex items-center">
      <div v-if="current === 1" class="flex items-center w-full mx-20 mt-10 space-x-5">
        <n-upload :action="backendAddr" :custom-request="uploadRequest" @change="handleLeftFileChange">
          <n-upload-dragger>
            <div style="margin-bottom: 12px">
              <n-icon size="48" :depth="3">
                <Photo />
              </n-icon>
            </div>
            <n-text style="font-size: 16px"> 点击或者拖动照片1到该区域来上传 </n-text>
            <n-p depth="3" style="margin: 8px 0 0 0"> 左眼球 </n-p>
          </n-upload-dragger>
        </n-upload>
        <n-upload :action="backendAddr" :custom-request="uploadRequest" @change="handleRightFileChange">
          <n-upload-dragger>
            <div style="margin-bottom: 12px">
              <n-icon size="48" :depth="3">
                <Photo />
              </n-icon>
            </div>
            <n-text style="font-size: 16px"> 点击或者拖动照片2到该区域来上传 </n-text>
            <n-p depth="3" style="margin: 8px 0 0 0"> 右眼球 </n-p>
          </n-upload-dragger>
        </n-upload>
      </div>
      <div
        v-else-if="current === 2"
        class="flex items-center justify-center w-full h-[70vh] flex-col"
      >
        <n-text class="text-2xl mb-3">在想了在想了</n-text>
        <n-spin size="large" />
      </div>
      <div v-else class="flex items-center justify-center w-full h-[70vh] flex-col">
        <div class="flex flex-row items-center justify-center mb-5">
          <n-icon class="text-6xl">
            <FaceIdError v-if="result![0].confidence < 0.6" />
            <HandRock v-else />
          </n-icon>
          <n-text class="ml-5 text-3xl" v-if="result![0].confidence < 0.6"> 我好像没懂 </n-text>
          <n-text class="ml-5 text-3xl" v-else> 嚯嚯！我懂了！ </n-text>
        </div>
        <n-text class="text-xl" v-if="result![0].confidence < 0.6">我不太确定这是什么奥...</n-text>
        <n-text class="text-xl" v-else>这应该是: {{ result![0].label }}</n-text>
        <n-text class="self-start mx-20 text-xl mt-10">详细识别结果：</n-text>
        <div class="flex items-start justify-start flex-col w-full px-20 mt-5">
          <div
            v-for="item in result"
            :key="item.label"
            class="flex items-center justify-between flex-row"
          >
            <n-text class="w-[10vw] mr-[5vw]">
              {{ item.label }}
            </n-text>
            <div class="w-[70vw]">
              <n-progress
                type="line"
                :height="20"
                :indicator-placement="'inside'"
                processing
                :fill-border-radius="0"
                :border-radius="4"
                :percentage="item.confidence * 100"
              />
            </div>
          </div>
        </div>
        <div class="flex mt-5">
          <n-button type="primary" @click="restart">
            <n-icon>
              <Refresh />
            </n-icon>
            再来一次
          </n-button>
        </div>
      </div>
    </div>
  </div>
</template>