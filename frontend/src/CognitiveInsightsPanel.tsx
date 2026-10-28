import { useMemo, useState, type CSSProperties, type FormEvent } from 'react'
import { Activity, BrainCircuit, Download, Send } from 'lucide-react'
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { analyzeEegCsv, answerInsight, type CognitiveRow } from './cognitiveInsights'

type Props = {
  rows: CognitiveRow[]
  columns: string[]
  datasetName: string
}

const CHANNEL_COLORS = ['#38bdf8', '#34d399', '#a78bfa', '#fbbf24', '#fb7185', '#2dd4bf', '#818cf8', '#f97316']
const BRAIN_SITES = [
  { x: 118, y: 92 },
  { x: 167, y: 58 },
  { x: 213, y: 58 },
  { x: 262, y: 92 },
  { x: 105, y: 145 },
  { x: 164, y: 170 },
  { x: 216, y: 170 },
  { x: 275, y: 145 },
]

function activityColor(activity: number): string {
  if (activity < 0.34) return '#38bdf8'
  if (activity < 0.67) return '#34d399'
  return '#fb7185'
}

function downloadReport(report: string, datasetName: string) {
  const blob = new Blob([report], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${datasetName.replace(/\.[^/.]+$/, '') || 'neurobridge'}-neuro-summary.txt`
  link.click()
  URL.revokeObjectURL(url)
}

export default function CognitiveInsightsPanel({ rows, columns, datasetName }: Props) {
  const analysis = useMemo(() => analyzeEegCsv(rows, columns), [columns, rows])
  const [chartType, setChartType] = useState<'bar' | 'pie'>('bar')
  const [question, setQuestion] = useState('')
  const [chat, setChat] = useState<Array<{ question: string; answer: string }>>([])

  if (!analysis) {
    return (
      <div className="page-block">
        <section className="panel card">
          <div className="cognitive-heading">
            <BrainCircuit aria-hidden="true" />
            <div>
              <span className="eyebrow">Experimental EEG analytics</span>
              <h2>Cognitive Insights</h2>
            </div>
          </div>
          <div className="empty-state">
            Upload a CSV with numeric EEG channel columns (for example EEG_CH1 through EEG_CH8) and at least two samples.
            Generic datasets are not classified as EEG and will not produce cognitive indicators.
          </div>
        </section>
      </div>
    )
  }

  const insights = analysis
  const eventByMean = [...insights.eventActivity].sort((a, b) => b.meanAmplitude - a.meanAmplitude)[0]
  const peakEvent = analysis.eventActivity[0]
  const loadColor = analysis.loadLevel === 'High' ? '#fb7185' : analysis.loadLevel === 'Medium' ? '#fbbf24' : '#34d399'
  const channelData = analysis.channels.map((channel) => ({
    name: channel.channel,
    activity: Number(channel.activity.toFixed(3)),
    rms: Number(channel.rms.toFixed(4)),
  }))
  const mappedChannels = columns
    .filter((column) => analysis.channels.some((channel) => channel.channel === column))
    .slice(0, 8)
  const report = [
    `NeuroBridge exploratory EEG summary`,
    `Dataset: ${datasetName}`,
    `Brain Signature ID: ${analysis.fingerprint}`,
    `Dominant Channel: ${analysis.channels[0]?.channel ?? 'Unavailable'}`,
    `Highest Activity Event: ${peakEvent?.event ?? 'Unavailable'}`,
    `Cognitive Load: ${analysis.loadLevel} (${analysis.cognitiveLoad} / 100 experimental variability index)`,
    `Focus Index: ${analysis.focusIndex}% (unvalidated proxy)`,
    `Stress Index: ${analysis.stressIndex}% (unvalidated variability proxy)`,
    `Signal Stability Score: ${analysis.stabilityScore}% (amplitude consistency estimate)`,
    `Sleep Quality Index: Unavailable; sleep staging and recording context are required.`,
    `Amplitude outliers: ${analysis.abnormalWindows} window(s) flagged by a three-standard-deviation screen; not a clinical abnormality finding.`,
    analysis.attentionShift
      ? `Possible activity shift: ${analysis.attentionShift.channel}, window ${analysis.attentionShift.window}, ${analysis.attentionShift.changePercent}% RMS change.`
      : `Possible activity shift: none exceeded the 50% RMS-change screen.`,
    '',
    'These exploratory signal summaries are not clinically validated and do not assess brain health, attention, stress, sleep, or disease.',
  ].join('\n')

  function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const submitted = question.trim()
    if (!submitted) return
    setChat((previous) => [...previous, { question: submitted, answer: answerInsight(submitted, insights) }])
    setQuestion('')
  }

  return (
    <div className="page-block cognitive-page">
      <section className="panel card cognitive-intro">
        <div className="cognitive-heading">
          <BrainCircuit aria-hidden="true" />
          <div>
            <span className="eyebrow">Experimental EEG analytics</span>
            <h2>Cognitive Insights</h2>
          </div>
          <span className="experimental-tag">Research prototype · not clinical</span>
        </div>
        <p className="note">
          These values are transparent amplitude-based heuristics, not an AI model or validated cognitive/health measurements.
          Cognitive load and stress proxy use capped coefficient of variation of sequential channel-RMS windows × 100;
          focus and stability are its inverse. Interpret only as exploratory summaries of this CSV.
        </p>
      </section>

      <section className="cognitive-score-grid" aria-label="Exploratory EEG indices">
        <article className="cognitive-card load-card">
          <div className="cognitive-card-title"><Activity size={17} /> Cognitive Load Index</div>
          <div
            className="load-gauge"
            style={{ '--gauge-value': `${analysis.cognitiveLoad * 3.6}deg`, '--gauge-color': loadColor } as CSSProperties}
            role="img"
            aria-label={`Experimental cognitive activity variability index ${analysis.cognitiveLoad} percent, ${analysis.loadLevel}`}
          >
            <div className="gauge-center">
              <strong>{analysis.cognitiveLoad}</strong>
              <span>/ 100</span>
            </div>
          </div>
          <strong className="load-level" style={{ color: loadColor }}>{analysis.loadLevel}</strong>
          <small>Based on between-window amplitude variability; thresholds: Low &lt;34, Medium 34–66, High ≥67.</small>
          {analysis.loadLevel === 'High' && <p className="load-warning">Cognitive Load Increased</p>}
        </article>

        <article className="cognitive-card">
          <div className="cognitive-card-title">Sleep Quality Index</div>
          <strong className="score-unavailable">Unavailable</strong>
          <p>EEG CSV alone does not provide reliable sleep staging or sleep-quality context.</p>
        </article>
        <article className="cognitive-card">
          <div className="cognitive-card-title">Stress Index <span className="proxy-label">proxy</span></div>
          <strong>{analysis.stressIndex}%</strong>
          <p>Amplitude variability proxy only; not a measurement of psychological stress.</p>
        </article>
        <article className="cognitive-card">
          <div className="cognitive-card-title">Focus Index <span className="proxy-label">proxy</span></div>
          <strong>{analysis.focusIndex}%</strong>
          <p>Inverse amplitude-variability estimate; not a direct attention measurement.</p>
        </article>
        <article className="cognitive-card">
          <div className="cognitive-card-title">Signal Stability</div>
          <strong>{analysis.stabilityScore}%</strong>
          <p>Amplitude consistency across {analysis.windowCount} sequential windows.</p>
        </article>
      </section>

      <section className="panel-grid cognitive-panels">
        <article className="panel card">
          <div className="panel-heading-row">
            <div>
              <span className="eyebrow">Relative channel RMS</span>
              <h3>Channel Activity</h3>
            </div>
            <select
              aria-label="Channel activity chart type"
              value={chartType}
              onChange={(event) => setChartType(event.target.value as 'bar' | 'pie')}
            >
              <option value="bar">Bar chart</option>
              <option value="pie">Pie chart</option>
            </select>
          </div>
          <div className="cognitive-chart">
            <ResponsiveContainer width="100%" height={280}>
              {chartType === 'bar' ? (
                <BarChart data={channelData}>
                  <XAxis dataKey="name" tick={{ fill: '#9AAEC3', fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#9AAEC3' }} />
                  <Tooltip formatter={(value) => [`${(Number(value) * 100).toFixed(1)}% of strongest channel`, 'Relative RMS']} />
                  <Bar dataKey="activity" name="Relative RMS" radius={[5, 5, 0, 0]}>
                    {channelData.map((channel, index) => (
                      <Cell key={channel.name} fill={CHANNEL_COLORS[index % CHANNEL_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              ) : (
                <PieChart>
                  <Pie data={channelData} dataKey="rms" nameKey="name" outerRadius={100} label>
                    {channelData.map((channel, index) => (
                      <Cell key={channel.name} fill={CHANNEL_COLORS[index % CHANNEL_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [Number(value).toFixed(4), 'RMS amplitude']} />
                </PieChart>
              )}
            </ResponsiveContainer>
          </div>
          <small className="note">RMS values are relative comparisons across selected channel columns; data units are inherited from the CSV.</small>
        </article>

        <article className="panel card">
          <div>
            <span className="eyebrow">Schematic visualization</span>
            <h3>3D-style EEG Activity Map</h3>
          </div>
          <div className="brain-map-wrap">
            <svg className="brain-map" viewBox="0 0 380 240" role="img" aria-label="Schematic brain activity map using channel relative RMS">
              <defs>
                <radialGradient id="brain-surface" cx="48%" cy="35%" r="70%">
                  <stop offset="0%" stopColor="#284966" />
                  <stop offset="75%" stopColor="#122b43" />
                  <stop offset="100%" stopColor="#091727" />
                </radialGradient>
                <filter id="brain-shadow" x="-30%" y="-30%" width="160%" height="170%">
                  <feDropShadow dx="0" dy="12" stdDeviation="9" floodColor="#020617" floodOpacity="0.75" />
                </filter>
              </defs>
              <path
                d="M190 27 C143 12 96 34 80 70 C53 72 43 107 60 128 C43 154 65 183 91 187 C108 222 151 229 190 210 C229 229 272 222 289 187 C315 183 337 154 320 128 C337 107 327 72 300 70 C284 34 237 12 190 27Z"
                fill="url(#brain-surface)"
                stroke="#5d829e"
                strokeWidth="2"
                filter="url(#brain-shadow)"
              />
              <path d="M190 34 C180 78 200 113 190 207" fill="none" stroke="#88a9bf" strokeOpacity="0.4" strokeWidth="2" />
              {mappedChannels.map((channelName, index) => {
                const channel = analysis.channels.find((entry) => entry.channel === channelName)
                if (!channel) return null
                const site = BRAIN_SITES[index]
                const x = site?.x ?? 140 + (index % 4) * 34
                const y = site?.y ?? 85 + Math.floor(index / 4) * 48
                const color = activityColor(channel.activity)
                return (
                  <g key={channel.channel}>
                    <circle cx={x} cy={y} r={13 + channel.activity * 5} fill={color} opacity="0.22" />
                    <circle cx={x} cy={y} r="7" fill={color} stroke="#e2e8f0" strokeOpacity="0.7" strokeWidth="1" />
                    <text x={x} y={y + 22} textAnchor="middle" fill="#dbeafe" fontSize="10">{channel.channel}</text>
                  </g>
                )
              })}
            </svg>
            <div className="heat-legend">
              <span><i className="heat-low" /> Low activity</span>
              <span><i className="heat-mid" /> Moderate</span>
              <span><i className="heat-high" /> High activity</span>
            </div>
            <small className="note">Channel positions are illustrative, not electrode coordinates or anatomical localization.</small>
          </div>
        </article>
      </section>

      <section className="panel-grid cognitive-panels">
        <article className="panel card">
          <span className="eyebrow">Change screening</span>
          <h3>Attention Shift Detector</h3>
          {analysis.attentionShift ? (
            <div className="insight-alert">
              <strong>Attention Shift Detected in {analysis.attentionShift.channel}</strong>
              <span>{analysis.attentionShift.changePercent}% RMS amplitude change near window {analysis.attentionShift.window}.</span>
            </div>
          ) : (
            <p className="note">No adjacent-window amplitude change crossed the 50% screening threshold. This does not measure attention directly.</p>
          )}
          <p className="note">A rapid amplitude shift is only a signal-level observation; it cannot confirm an attention change.</p>
          <div className="fingerprint-card">
            <span>Dataset fingerprint</span>
            <strong>{analysis.fingerprint}</strong>
            <small>Derived from the uploaded channel values; not a biological identity or secure identifier.</small>
          </div>
        </article>

        <article className="panel card">
          <div className="panel-heading-row">
            <div>
              <span className="eyebrow">Rule-based answers from uploaded data</span>
              <h3>AI Insight Engine</h3>
            </div>
          </div>
          <div className="prompt-chips">
            {['What is the strongest channel?', 'Which event has higher amplitude?', 'Show abnormalities.', 'Which channel is most active?', 'Which event generated peak activity?'].map((prompt) => (
              <button key={prompt} type="button" onClick={() => setChat((previous) => [...previous, { question: prompt, answer: answerInsight(prompt, analysis) }])}>
                {prompt}
              </button>
            ))}
          </div>
          <div className="insight-chat" aria-live="polite">
            {!chat.length && <p className="note">Ask about this dataset to see a transparent, rule-based summary.</p>}
            {chat.map((item, index) => (
              <div className="chat-pair" key={`${item.question}-${index}`}>
                <p className="chat-question">{item.question}</p>
                <p className="chat-answer">{item.answer}</p>
              </div>
            ))}
          </div>
          <form className="insight-input" onSubmit={submitQuestion}>
            <input
              aria-label="Ask about this EEG dataset"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about this EEG data"
            />
            <button type="submit" className="primary-button" disabled={!question.trim()} aria-label="Send question">
              <Send size={16} />
            </button>
          </form>
        </article>
      </section>

      <section className="panel card">
        <div className="panel-heading-row">
          <div>
            <span className="eyebrow">Generated from the current upload</span>
            <h3>Neuro Summary Report</h3>
          </div>
          <button type="button" className="secondary-button" onClick={() => downloadReport(report, datasetName)}>
            <Download size={16} /> Download summary
          </button>
        </div>
        <ul className="neuro-summary-list">
          <li><strong>Dominant Channel:</strong> {analysis.channels[0]?.channel ?? 'Unavailable'}</li>
          <li><strong>Highest Activity Event:</strong> {peakEvent?.event ?? 'No event/label column detected'}</li>
          <li><strong>Mean-Amplitude Event:</strong> {eventByMean?.event ?? 'Unavailable'}</li>
          <li><strong>Cognitive Load:</strong> {analysis.loadLevel} ({analysis.cognitiveLoad}/100 experimental variability index)</li>
          <li><strong>Focus Index:</strong> {analysis.focusIndex}% experimental proxy</li>
          <li><strong>Stress Index:</strong> {analysis.stressIndex}% experimental proxy</li>
          <li><strong>Signal Stability:</strong> {analysis.stabilityScore}% amplitude consistency estimate</li>
          <li><strong>Amplitude screening:</strong> {analysis.abnormalWindows ? `${analysis.abnormalWindows} potential outlier window(s) flagged` : 'No three-standard-deviation amplitude outliers flagged'}; this is not a clinical assessment.</li>
        </ul>
        <p className="clinical-disclaimer">
          Sleep quality cannot be calculated from channel amplitude alone. Cognitive load, focus, stress, and signal stability values are experimental heuristics and must not be used for diagnosis or health decisions.
        </p>
      </section>
    </div>
  )
}
