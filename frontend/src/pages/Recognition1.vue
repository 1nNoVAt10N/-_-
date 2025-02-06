<script setup lang="ts">
import { ref } from 'vue';
import { Photo, HandRock, Refresh, EyeCheck, MoodSuprised, MoodSad, MoodSmile, FileZip } from '@vicons/tabler';
import type { UploadCustomRequestOptions, UploadFileInfo } from 'naive-ui';
import { lyla } from '@lylajs/web';
import { useMessage } from 'naive-ui';
const message = useMessage();
const current = ref(1);
const backendAddr = import.meta.env.VITE_API_URL;

const result = ref<Record<string, string[]> | null>(null);
const zipFile = ref<File | null>(null);

const dialog = useDialog();
import { useRouter } from 'vue-router';
const router = useRouter();
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
  if (zipFile.value) {
    formData.append('zip_file', zipFile.value);
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
        console.log(json);
        
        result.value = json;
        current.value = 3;
        onFinish();
      })
      .catch((error) => {
        message.error(error.message);
        message.error('哦不，好像有点问题，请刷新重试！');
        dialog.error({
          title: '错误',
          content: '你传的东西有问题',
          positiveText: '再来一次',
          negativeText: '回到首页',
          onPositiveClick: () => {
            current.value = 1;
          },
          onNegativeClick: () => {
            router.push('/');
          }
        })
        onError();
      });
  } else {
    message.error('请上传压缩包');
  }
};

const handleZipFileChange = (info: { file: UploadFileInfo }) => {
  zipFile.value = info.file.file as File;
};

const restart = () => {
  current.value = 1;
  result.value = null;
  zipFile.value = null;
};


</script>

<template>
  <div class="flex flex-col w-full">
    <div class="m-10 flex items-start justify-center mt-10 w-full">
      <n-steps :current="current" class="w-full" status="process">
        <n-step title="上传压缩包" description="你得告诉我什么样" />
        <n-step title="等等" description="等等我们的AI" />
        <n-step title="所以你得了什么病" description="让我告诉你" />
      </n-steps>
    </div>
    <div class="mt-5 w-full flex items-center">
      <div v-if="current === 1" class="flex items-center w-full mx-20 mt-10 space-x-5">
        <n-upload :action="backendAddr" :custom-request="uploadRequest" @change="handleZipFileChange">
          <n-upload-dragger>
            <div style="margin-bottom: 12px">
              <n-icon size="48" :depth="3">
                <FileZip />
              </n-icon>
            </div>
            <n-text style="font-size: 16px"> 点击或者拖动压缩包到该区域来上传 </n-text>
            <n-p depth="3" style="margin: 8px 0 0 0"> 压缩包 </n-p>
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
      <div v-else class="mt-10 flex items-center justify-center w-full h-[70vh] flex-col">
        <div 
            v-for="(diseases, key) in result"
            :key="key" class="flex items-center justify-center w-full h-[70vh] flex-col">
            <div class="flex flex-row items-center justify-center mb-5">
            <n-icon class="text-6xl">
                <MoodSmile v-if="diseases![0]=='正常'" />
                <MoodSad v-else />
            </n-icon>
            <n-text class="ml-5 text-3xl" v-if="diseases![0]=='正常'"> 好耶！{{key}}的眼睛好像是正常的 </n-text>
            <n-text class="ml-5 text-3xl" v-else> 完蛋！{{key}}的眼睛好像得病了 </n-text>
            </div>
            <n-text class="text-xl" v-if="diseases![0]=='正常'">可能的病有这些:{{ diseases }}</n-text>
            <n-text class="text-xl" v-else>可能的病有这些:{{ diseases }}</n-text>
        </div>
        <div class="self-start mx-20 text-xl mt-10">详细识别结果：</div>
        <div class="flex items-start justify-start flex-col w-full px-20 mt-5">
          <div
            v-for="(diseases, key) in result"
            :key="key"
            class="flex items-start justify-start flex-col mb-5"
          >
            <n-text class="text-lg font-bold">{{ key }}</n-text>
            <div class="flex items-start justify-start flex-col w-full">
              <div
                v-for="disease in diseases"
                :key="disease"
                class="flex items-center justify-between flex-row"
              >
                <n-text class="w-[10vw] mr-[5vw]">
                  {{ disease }}
                </n-text>
                <div class="w-[70vw]">
                  <n-progress
                    type="line"
                    :height="20"
                    :indicator-placement="'inside'"
                    processing
                    :fill-border-radius="0"
                    :border-radius="4"
                    :percentage="1 * 10"
                  />
                </div>
              </div>
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