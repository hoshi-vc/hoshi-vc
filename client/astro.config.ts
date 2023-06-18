// Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { defineConfig } from 'astro/config'

import react from '@astrojs/react'
import unocss from 'unocss/astro'

// https://astro.build/config
export default defineConfig({
  integrations: [
    react(),
    unocss({
      injectReset: true,
    }),
  ],

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  // prevent vite from obscuring rust errors
  clearScreen: false,
  // tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
  },
  // to make use of `TAURI_DEBUG` and other env variables
  // https://tauri.studio/v1/api/config#buildconfig.beforedevcommand
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    // Tauri supports es2021
    target: process.env['TAURI_PLATFORM'] == 'windows' ? 'chrome105' : 'safari13',
    // don't minify for debug builds
    minify: !process.env['TAURI_DEBUG'] ? 'esbuild' : false,
    // produce sourcemaps for debug builds
    sourcemap: !!process.env['TAURI_DEBUG'],
  },

  // Vite options not supported by astro natively. (values are same as above)
  vite: {
    server: {
      strictPort: true,
    },
    build: {
      target: process.env['TAURI_PLATFORM'] == 'windows' ? 'chrome105' : 'safari13',
    },
  },
})
