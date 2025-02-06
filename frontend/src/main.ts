import { createApp } from 'vue';
import { createRouter, createWebHistory } from 'vue-router';
import App from './App.vue';
import '@/styles/tailwind.css';
import Homepage from './pages/Homepage.vue';
import NotFound from './pages/NotFound.vue';
import Recognition from './pages/Recognition.vue';

// Set up the routes
const routes = [
  { path: '/', component: Homepage },
  { path: '/recognition', component: Recognition },
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
