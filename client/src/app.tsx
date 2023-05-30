import { invoke } from '@tauri-apps/api/tauri'
import { useState } from 'preact/hooks'

function App() {
  const [greetMsg, setGreetMsg] = useState('')
  const [name, setName] = useState('')

  async function greet() {
    // Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
    setGreetMsg(await invoke('greet', { name }))
  }

  return (
    <>
      <h1>Welcome to Tauri!</h1>

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

export default App
