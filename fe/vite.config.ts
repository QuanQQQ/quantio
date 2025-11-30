import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/api')
      }
    },
    fs: {
      allow: [
        // 允许访问项目根目录及数据目录，用于静态加载 CSV/DB（仅本地开发）
        '..',
        '.'
      ]
    }
  }
})
