// Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
        onSubmit={(e) => {
          e.preventDefault()
          greet()
        }}>
        <input onChange={(e) => setName(e.currentTarget.value)} placeholder='Enter a name...' />
        <button type='submit'>Greet</button>
      </form>

      <p>{greetMsg}</p>
    </>
  )
}
