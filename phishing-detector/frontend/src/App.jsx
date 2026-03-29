import { useMemo, useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [emailText, setEmailText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const suspiciousSet = useMemo(() => new Set(result?.suspicious_words ?? []), [result])

  const highlightedPreview = useMemo(() => {
    if (!emailText.trim()) return [{ text: '', highlighted: false }]
    const words = emailText.split(/(\s+)/)
    return words.map((piece, index) => {
      const normalized = piece.toLowerCase().replace(/[^a-z0-9 ]/g, '').trim()
      const highlighted = suspiciousSet.has(normalized)
      return { key: `${piece}-${index}`, text: piece, highlighted }
    })
  }, [emailText, suspiciousSet])

  const onAnalyze = async () => {
    setError('')
    setResult(null)
    if (!emailText.trim()) {
      setError('Please paste email content before analysis.')
      return
    }
    try {
      setLoading(true)
      const response = await axios.post('http://127.0.0.1:5000/predict', {
        email_text: emailText,
      })
      setResult(response.data)
    } catch (err) {
      setError(err?.response?.data?.error || 'Backend not reachable. Start Flask server.')
    } finally {
      setLoading(false)
    }
  }

  const probabilityPct = Math.round((result?.probability || 0) * 100)

  return (
    <main className="page">
      <header>
        <h1>Phishing Email Detection Dashboard</h1>
        <p className="subtitle">
          Paste an email and compare ML-based phishing risk indicators.
        </p>
      </header>

      <section className="card">
        <label htmlFor="email-input">Email Content</label>
        <textarea
          id="email-input"
          value={emailText}
          onChange={(e) => setEmailText(e.target.value)}
          placeholder="Paste raw email text here..."
          rows={10}
        />
        <button className="analyze-btn" onClick={onAnalyze} disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Email'}
        </button>
        {error ? <p className="error">{error}</p> : null}
      </section>

      {result ? (
        <section className="results">
          <div className="card">
            <h2>Prediction</h2>
            <p
              className={`prediction ${
                result.prediction === 'phishing' ? 'danger' : 'safe'
              }`}
            >
              {result.prediction}
            </p>
            <div className="prob-wrap">
              <div className="prob-head">
                <span>Phishing Probability</span>
                <span>{probabilityPct}%</span>
              </div>
              <div className="progress">
                <div className="bar" style={{ width: `${probabilityPct}%` }} />
              </div>
            </div>
            <p className="explanation">{result.explanation}</p>
          </div>

          <div className="card">
            <h2>Suspicious Indicators</h2>
            <p>
              <strong>Suspicious words:</strong>{' '}
              {result.suspicious_words?.length
                ? result.suspicious_words.join(', ')
                : 'None detected'}
            </p>
            <p>
              <strong>Links detected:</strong>{' '}
              {result.links_detected?.length ? result.links_detected.join(', ') : 'None'}
            </p>
            <div className="feature-grid">
              {Object.entries(result.extracted_features || {}).map(([key, value]) => (
                <div key={key} className="feature-item">
                  <span>{key}</span>
                  <strong>{String(value)}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h2>Highlighted Suspicious Words</h2>
            <div className="preview">
              {highlightedPreview.map((token) =>
                token.highlighted ? (
                  <mark key={token.key}>{token.text}</mark>
                ) : (
                  <span key={token.key}>{token.text}</span>
                ),
              )}
            </div>
          </div>
        </section>
      ) : null}
    </main>
  )
}

export default App
