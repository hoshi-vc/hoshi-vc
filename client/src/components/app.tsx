// Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { invoke } from '@tauri-apps/api/tauri'
import { PartyPopperIcon } from 'lucide-react'
import { useState } from 'react'

export const App = () => {
  const [greetMsg, setGreetMsg] = useState('')
  const [name, setName] = useState('')

  async function greet() {
    // Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
    setGreetMsg(await invoke('greet', { name }))
  }

  return (
    <>
      <h1 className='flex justify-center items-center'>
        ... and React!
        <PartyPopperIcon size='1em' className='ms-1' />
      </h1>

      <form
        className='flex w-full max-w-sm mx-auto items-center space-x-2'
        onSubmit={(e) => {
          e.preventDefault()
          greet()
        }}>
        <Input onChange={(e) => setName(e.currentTarget.value)} placeholder='Enter a name...' />
        <Button type='submit'>Greet</Button>
      </form>

      <p>{greetMsg}</p>
    </>
  )
}
