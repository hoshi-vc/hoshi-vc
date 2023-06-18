// Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { defineConfig, presetUno } from 'unocss'

export default defineConfig({
  presets: [presetUno()],
  theme: {
    // ref: https://ui.shadcn.com/docs/installation
    container: {
      center: true,
      padding: '2rem',
    },
    breakpoints: {
      '2xl': '1400px',
    },
    colors: {
      border: 'hsl(var(--border))',
      input: 'hsl(var(--input))',
      ring: 'hsl(var(--ring))',
      background: 'hsl(var(--background))',
      foreground: 'hsl(var(--foreground))',
      primary: {
        DEFAULT: 'hsl(var(--primary))',
        foreground: 'hsl(var(--primary-foreground))',
      },
      secondary: {
        DEFAULT: 'hsl(var(--secondary))',
        foreground: 'hsl(var(--secondary-foreground))',
      },
      destructive: {
        DEFAULT: 'hsl(var(--destructive))',
        foreground: 'hsl(var(--destructive-foreground))',
      },
      muted: {
        DEFAULT: 'hsl(var(--muted))',
        foreground: 'hsl(var(--muted-foreground))',
      },
      accent: {
        DEFAULT: 'hsl(var(--accent))',
        foreground: 'hsl(var(--accent-foreground))',
      },
      popover: {
        DEFAULT: 'hsl(var(--popover))',
        foreground: 'hsl(var(--popover-foreground))',
      },
      card: {
        DEFAULT: 'hsl(var(--card))',
        foreground: 'hsl(var(--card-foreground))',
      },
    },
    borderRadius: {
      // ref: https://github.com/unocss/unocss/blob/698a7e68d6dd2be0cc9e4fa7c335fd940776012c/packages/preset-mini/src/_theme/misc.ts
      lg: 'var(--radius)',
      md: 'calc(var(--radius) - 2px)',
      sm: 'calc(var(--radius) - 4px)',
    },
    animation: {
      // ref: https://github.com/unocss/unocss/blob/0e8dcbb3755d772a23b480aa2ea50d9ee406227c/packages/preset-wind/src/theme.ts
      keyframes: {
        'accordion-down': '{0% {height:0} 100% {height:var(--radix-accordion-content-height)}}',
        'accordion-up': '{0% {height:var(--radix-accordion-content-height)} 100% {height:0}}',
      },
      durations: {
        'accordion-down': '0.2s',
        'accordion-up': '0.2s',
      },
      timingFns: {
        'accordion-down': 'ease-out',
        'accordion-up': 'ease-out',
      },
    },
  },
}) as unknown
