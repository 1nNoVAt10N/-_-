import { createApp } from 'vue';
import { createRouter, createWebHistory } from 'vue-router';
import App from './App.vue';
import '@/styles/tailwind.css';
import Homepage from './pages/Homepage.vue';
import NotFound from './pages/NotFound.vue';
import Recognition from './pages/Recognition.vue';
import Recognition1 from './pages/Recognition1.vue';
import Diagnosis from './pages/Diagnosis.vue';
import DiagnosisDetail from './pages/DiagnosisDetail.vue';
import Dashboard from './pages/Dashboard.vue';
import Login from './pages/Login.vue';

// Set up the routes
const routes = [
  { path: '/', component: Homepage },
  { path: '/login', component: Login },
  { path: '/dashboard', component: Dashboard },
  { path: '/singleRecognition', component: Recognition },
  { path: '/multiRecognition', component: Recognition1 },
  { path: '/diagnosis', component: Diagnosis },
  { path: '/diagnosis/:id', component: DiagnosisDetail },
  { path: '/:pathMatch(.*)*', component: NotFound },
];

// Create the router
const router = createRouter({
  history: createWebHistory(),
  routes,
});

const meta = document.createElement('meta');
meta.name = 'naive-ui-style';
document.head.appendChild(meta);

// Use the router
createApp(App).use(router).mount('#app');
