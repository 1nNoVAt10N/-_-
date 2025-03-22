<template>
  <div class="login-container">
    <n-card class="login-card" title="用户登录" :bordered="false">
      <n-form
        ref="formRef"
        :model="formData"
        :rules="rules"
        @submit.prevent="handleSubmit"
      >
        <n-form-item path="username" label="用户名">
          <n-input
            v-model:value="formData.username"
            placeholder="请输入用户名"
            @keydown.enter.prevent
          />
        </n-form-item>

        <n-form-item path="password" label="密码">
          <n-input
            v-model:value="formData.password"
            type="password"
            placeholder="请输入密码"
            show-password-on="click"
            @keydown.enter.prevent
          />
        </n-form-item>

        <div class="login-options">
          <n-checkbox v-model:checked="rememberMe">记住我</n-checkbox>
          <n-button text type="primary" @click="handleForgetPassword">
            忘记密码？
          </n-button>
        </div>

        <n-button
          attr-type="submit"
          type="primary"
          block
          :loading="submitting"
        >
          登录
        </n-button>

        <div class="register-link">
          没有账号？
          <n-button text type="primary" @click="handleRegister">
            立即注册
          </n-button>
        </div>
      </n-form>
    </n-card>
  </div>
</template>

<script setup>
import { ref, defineEmits } from 'vue';
import { useMessage } from 'naive-ui';

const formRef = ref(null);
const message = useMessage();
const emit = defineEmits(['loginSuccess']);

const formData = ref({
  username: '',
  password: ''
});

const rememberMe = ref(false);
const submitting = ref(false);

const rules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 4, max: 16, message: '用户名长度需在4-16位之间', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 20, message: '密码长度需在6-20位之间', trigger: 'blur' }
  ]
};

const handleSubmit = (e) => {
  e.preventDefault();
  formRef.value?.validate(async (errors) => {
    if (!errors) {
      submitting.value = true;
      try {
        // 这里替换为实际的登录API调用
        await new Promise(resolve => setTimeout(resolve, 1500));
        message.success('登录成功');
        // 登录成功后的跳转逻辑
        emit('loginSuccess');
      } catch (error) {
        message.error('登录失败，请检查用户名和密码');
      } finally {
        submitting.value = false;
      }
    } else {
      message.error('请正确填写表单');
    }
  });
};

const handleForgetPassword = () => {
  message.info('请联系管理员重置密码');
};

const handleRegister = () => {
  // 这里添加注册跳转逻辑
  message.info('跳转至注册页面');
};
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background-color: rgba(0, 0, 0, 0.5); /* 半透明背景 */
  z-index: 1000;
}

.login-card {
  width: 400px;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.login-options {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 16px 0;
}

.register-link {
  margin-top: 24px;
  text-align: center;
  color: #606266;
}
</style>