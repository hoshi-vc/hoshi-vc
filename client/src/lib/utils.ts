// Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const error = (o: string) => {
  throw new Error(o)
}
export const bug = (o?: string) => error(o ? `BUG: ${o}` : 'BUG')
export const never = (_: never) => error('BUG (NEVER)')
export const todo = () => error('NOT IMPLEMENTED')
export const assert = (o: boolean, msg?: string): asserts o => void (o ? 0 : error(msg ?? 'ASSERT'))
export const assertWarn = (o: boolean, msg?: string) => void (o ? 0 : console.warn(msg ?? 'ASSERT'))

export const uuid = () => crypto.randomUUID()

export type Expand<T> = T extends infer O ? { [K in keyof O]: O[K] } : never
export type ExpandRecursively<T> = T extends object
  ? T extends infer O
    ? { [K in keyof O]: ExpandRecursively<O[K]> }
    : never
  : T
