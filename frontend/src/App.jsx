import { useMemo, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)

  const sortedProbabilities = useMemo(() => {
    if (!result?.probabilities) return []
    return Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
  }, [result])

  const onFileChange = (selected) => {
    if (!selected) return
    setFile(selected)
    setPreview(URL.createObjectURL(selected))
    setResult(null)
  }

  const submit = async () => {
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)

    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setResult(data)
      setHistory((prev) => [{ name: file.name, ...data }, ...prev].slice(0, 8))
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="container">
      <section className="card">
        <h1>Fake Image Detection</h1>
        <p>Drag, drop, and classify images with ensemble AI detection.</p>

        <label
          className="dropzone"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault()
            onFileChange(event.dataTransfer.files[0])
          }}
        >
          <input type="file" accept="image/*" onChange={(e) => onFileChange(e.target.files?.[0])} />
          <span>{file ? file.name : 'Drop image here or click to upload'}</span>
        </label>

        {preview && <img className="preview" src={preview} alt="Preview" />}

        <button disabled={!file || loading} onClick={submit}>
          {loading ? 'Analyzing...' : 'Predict'}
        </button>

        {result && (
          <article className="result">
            <h2>Prediction</h2>
            <p>
              <strong>{result.label}</strong> ({(result.confidence * 100).toFixed(2)}%)
            </p>
            <ul>
              {sortedProbabilities.map(([label, confidence]) => (
                <li key={label}>
                  {label}: {(confidence * 100).toFixed(2)}%
                </li>
              ))}
            </ul>
          </article>
        )}
      </section>

      <section className="card history">
        <h2>History</h2>
        {history.length === 0 ? (
          <p>No predictions yet.</p>
        ) : (
          <ul>
            {history.map((entry, index) => (
              <li key={`${entry.name}-${index}`}>
                <span>{entry.name}</span>
                <strong>
                  {entry.label} ({(entry.confidence * 100).toFixed(2)}%)
                </strong>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  )
}

export default App
