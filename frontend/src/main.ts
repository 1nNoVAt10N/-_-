import { createApp } from 'vue';
import { createRouter, createWebHistory } from 'vue-router';
import App from './App.vue';
import '@/styles/tailwind.css';
import Homepage from './pages/Homepage.vue';
import NotFound from './pages/NotFound.vue';
import Recognition from './pages/Recognition.vue';
import Recognition1 from './pages/Recognition1.vue';
import Diagnosis from './pages/Diagnosis.vue';
import DiagnosisList from './pages/DiagnosisList.vue';
import MedicalRecords from './pages/MedicalRecords.vue';
import Fund from './pages/Fund.vue';
import Dashboard from './pages/Dashboard.vue';
import Login from './pages/Login.vue';
import MultiModalDiagnosis from './pages/MultiModalDiagnosis.vue';
import Billing from './pages/Billing.vue';
import Chart from './pages/Chart.vue';
import Network from './pages/Network.vue';
import Services from './pages/Services.vue';
import System from './pages/System.vue';

// Set up the routes
const routes = [
  { path: '/', component: Homepage },
  { path: '/login', component: Login },
  { path: '/dashboard', component: Dashboard },
  { path: '/singleRecognition', component: Recognition },
  { path: '/multiRecognition', component: MultiModalDiagnosis },
  { path: '/diagnosis', component: Diagnosis },
  { path: '/diagnosis/:id', component: DiagnosisList },
  {path: '/medical-records', component: MedicalRecords},
  {path: '/fund/:id', component: Fund},
  { path: '/:pathMatch(.*)*', component: NotFound },
  { path: '/Billing', component: Billing},
  { path: '/charts', component: Chart},
  { path: '/services', component: Services},
  { path: '/network', component: Network},
  { path: '/system', component: System}
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
