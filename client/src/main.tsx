// Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import '@unocss/reset/tailwind.css'

import '@/global.css'

import 'virtual:uno.css'

import { App } from '@/components/app'
import React from 'react'
import ReactDOM from 'react-dom/client'

ReactDOM.createRoot(document.getElementById('app') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
