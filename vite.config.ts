/// <reference types="vitest" />

import legacy from '@vitejs/plugin-legacy'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
  ],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
  },
	esbuild: {
    target: "esnext", // Change from es2020 to esnext
  },
  build: {
		target: ['esnext'],
		commonjsOptions: {
      transformMixedEsModules: true,
    },
  },
	optimizeDeps: {
		esbuildOptions: {
			target: "esnext",
		}
	},
})
