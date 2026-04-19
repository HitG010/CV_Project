import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/art-cloak':        'http://localhost:8080',
      '/face-cloak':       'http://localhost:8080',
      '/compare-attacks':  'http://localhost:8080',
      '/agent':            'http://localhost:8080',
    }
  }
})
