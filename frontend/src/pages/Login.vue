<template>
  <!-- 整个组件外部，用于居中 -->
  <div class="flex items-center justify-center min-h-screen bg-gray-100">
    <!-- 登录主容器 -->
    <div
      class="flex w-[900px] h-[600px] bg-white/90 rounded-[15px]
             backdrop-blur-[8px] transition-transform duration-300
             relative z-[1]
          "
    >
      <!-- 左侧：登录表单区域 -->
      <div class="flex flex-col flex-1 p-10">
        <!-- Logo -->
        <div class="text-center mb-10">
          <!-- 眼科图标 + pulse 动画 -->
          <div class="text-4xl text-[#2196a3] mb-4 animate-pulse">
            <i class="fas fa-eye"></i>
          </div>
        </div>

        <!-- 表单 -->
        <div class="flex-1">
          <form id="loginForm" class="space-y-6" @submit.prevent="handleSubmit">
            <!-- 用户名 -->
            <div>
              <label for="username" class="block text-sm text-gray-700 mb-2">
                用户名
              </label>
              <div class="relative">
                <i
                  class="fas fa-user absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
                ></i>
                <input
                  v-model="username"
                  type="text"
                  id="username"
                  name="username"
                  placeholder="请输入您的用户名"
                  required
                  class="w-full pl-9 pr-3 py-2 rounded-md border border-gray-300
                         focus:border-[#2196a3] focus:outline-none
                         focus:ring-2 focus:ring-[#2196a3]/15
                         transition-colors"
                />
              </div>
            </div>

            <!-- 密码 -->
            <div>
              <label for="password" class="block text-sm text-gray-700 mb-2">
                密码
              </label>
              <div class="relative">
                <i
                  class="fas fa-lock absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
                ></i>
                <input
                  v-model="password"
                  type="password"
                  id="password"
                  name="password"
                  placeholder="请输入您的密码"
                  required
                  class="w-full pl-9 pr-10 py-2 rounded-md border border-gray-300
                         focus:border-[#2196a3] focus:outline-none
                         focus:ring-2 focus:ring-[#2196a3]/15
                         transition-colors"
                />
                <i
                  class="fas fa-eye absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 cursor-pointer"
                  @click="togglePassword"
                ></i>
              </div>
            </div>

            <!-- 记住我 + 忘记密码 -->
            <div class="flex items-center justify-between">
              <div class="flex items-center">
                <input
                  v-model="rememberMe"
                  type="checkbox"
                  id="remember"
                  class="mr-2"
                />
                <label for="remember" class="text-sm text-gray-700"
                  >记住我</label
                >
              </div>
              <a
                href="#"
                class="text-sm text-[#2196a3] hover:text-[#0a7680] hover:underline"
                >忘记密码?</a
              >
            </div>

            <!-- 登录按钮 -->
            <button
              type="submit"
              class="w-full py-2 text-white font-semibold rounded-md
                     bg-[#36ad6a] hover:bg-green-600 shadow-md
                     transition-colors text-base"
            >
              登录系统
            </button>
          </form>
        </div>
      </div>

      <!-- 右侧：简介信息（中大屏可见，小屏隐藏） -->
      <div
        class="hidden md:flex flex-1 bg-[#36ad6a] text-white p-10
               flex-col justify-center relative"
      >
        <h1 class="text-3xl mb-3 font-semibold">眼底影像AI诊断</h1>
        <p class="mb-6 opacity-90">
          基于深度学习的眼科疾病智能筛查与诊断平台
        </p>
        <ul class="space-y-3">
          <li class="flex items-center text-sm hover:translate-x-1 transition">
            <i class="fas fa-eye text-lg mr-2 text-white/85"></i>
            眼底图像智能分析
          </li>
          <li class="flex items-center text-sm hover:translate-x-1 transition">
            <i class="fas fa-chart-line text-lg mr-2 text-white/85"></i>
            疾病风险预测
          </li>
          <li class="flex items-center text-sm hover:translate-x-1 transition">
            <i class="fas fa-brain text-lg mr-2 text-white/85"></i>
            AI辅助临床决策
          </li>
          <li class="flex items-center text-sm hover:translate-x-1 transition">
            <i class="fas fa-database text-lg mr-2 text-white/85"></i>
            眼科影像数据库
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const username = ref('')
const password = ref('')
const rememberMe = ref(false)

// 切换密码显示/隐藏
const togglePassword = () => {
  const passwordField = document.getElementById('password') as HTMLInputElement
  if (passwordField.type === 'password') {
    passwordField.type = 'text'
  } else {
    passwordField.type = 'password'
  }
}

const handleSubmit = () => {
  console.log('用户名:', username.value)
  console.log('密码:', password.value)
  console.log('记住我:', rememberMe.value)
  router.push('/dashboard')
}
</script>

<style scoped>
</style>
