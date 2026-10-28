import { useMemo, useState, type ChangeEvent } from 'react'
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  Download,
  FileText,
  Gauge,
  TableProperties,
  Upload,
} from 'lucide-react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import CognitiveInsightsPanel from './CognitiveInsightsPanel'
import './App.css'

type PageKey = 'Dashboard' | 'Data Upload' | 'Data Table' | 'Chart Studio' | 'EEG Explorer' | 'Cognitive Insights'
type ChartType = 'line' | 'bar' | 'pie' | 'scatter' | 'histogram' | 'area'
type Aggregation = 'count' | 'sum' | 'mean'
type DataRow = Record<string, string | number | null>
type ColumnMeta = {
  name: string
  type: 'numeric' | 'category' | 'mixed'
  missing: number
  total: number
}

type ChartBandDatum = { label: string; value: number }
type ChartSeriesDatum = { name: string; [key: string]: string | number | null }
type ScatterDatum = { x: number; y: number }
type ScatterSeries = { column: string; data: ScatterDatum[] }
type ChartDatum = ChartSeriesDatum | ScatterDatum

const SERIES_COLORS = ['#48D9F5', '#19C6B4', '#7ae7ff', '#facc15', '#f472b6', '#c084fc']
const MAX_CHART_POINTS = 800
const EEG_BANDS = [
  { label: 'Delta', key: 'delta', range: [0.5, 4] },
  { label: 'Theta', key: 'theta', range: [4, 8] },
  { label: 'Alpha', key: 'alpha', range: [8, 13] },
  { label: 'Beta', key: 'beta', range: [13, 30] },
  { label: 'Gamma', key: 'gamma', range: [30, 45] },
] as const

function toNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null
  const text = String(value).trim()
  if (!text) return null
  const parsed = Number(text)
  return Number.isFinite(parsed) ? parsed : null
}

function isNumericLike(value: unknown): boolean {
  return toNumber(value) !== null
}

function parseCSVText(text: string): { columns: string[]; rows: DataRow[] } {
  const lines = text.replace(/\r\n/g, '\n').split('\n').filter((line) => line.trim().length > 0)
  if (!lines.length) return { columns: [], rows: [] }

  const headers = lines[0].split(',').map((header) => header.trim() || 'untitled')
  const rows: DataRow[] = []

  for (let rowIndex = 1; rowIndex < lines.length; rowIndex += 1) {
    const cells = lines[rowIndex].split(',')
    const row: DataRow = {}
    headers.forEach((header, index) => {
      const rawValue = cells[index]?.trim() ?? ''
      row[header] = rawValue === '' ? null : rawValue
    })
    rows.push(row)
  }

  return { columns: headers, rows }
}

function summarizeColumns(columns: string[], rows: DataRow[]): ColumnMeta[] {
  return columns.map((name) => {
    const values = rows.map((row) => row[name])
    const missing = values.filter((entry) => entry === null || entry === '').length
    const numericCount = values.filter((entry) => isNumericLike(entry)).length
    const hasText = values.some((entry) => entry !== null && entry !== '' && !isNumericLike(entry))
    const type = numericCount === values.length && !hasText ? 'numeric' : hasText && numericCount > 0 ? 'mixed' : 'category'

    return { name, type, missing, total: rows.length }
  })
}

function buildHistogram(values: number[], bins = 10) {
  if (!values.length) return []
  const min = Math.min(...values)
  const max = Math.max(...values)
  const spread = max - min || 1
  const step = spread / bins

  return Array.from({ length: bins }, (_, index) => {
    const lower = min + index * step
    const upper = index === bins - 1 ? max : lower + step
    const count = values.filter((value) => value >= lower && value < upper).length
    return {
      name: `${lower.toFixed(1)}-${upper.toFixed(1)}`,
      count,
    }
  })
}

function sampleChartPoints<T>(values: T[]): T[] {
  if (values.length <= MAX_CHART_POINTS) return values
  return Array.from({ length: MAX_CHART_POINTS }, (_, index) => {
    const sourceIndex = Math.round((index * (values.length - 1)) / (MAX_CHART_POINTS - 1))
    return values[sourceIndex]
  })
}

function computeBandPower(rows: DataRow[]) {
  if (!rows.length) return []
  const candidateColumns = Object.keys(rows[0]).filter((column) => rows.every((row) => isNumericLike(row[column])))
  if (!candidateColumns.length) return []

  const channelValues = rows.map((row) => toNumber(row[candidateColumns[0]]) ?? 0)
  const n = channelValues.length
  const sampleRate = n > 1 ? 256 : 1
  const frequencies = Array.from({ length: n }, (_, index) => (index * sampleRate) / n)
  const spectrum = Array.from({ length: n }, (_, index) => {
    let real = 0
    let imag = 0
    for (let timeIndex = 0; timeIndex < n; timeIndex += 1) {
      const angle = (-2 * Math.PI * index * timeIndex) / n
      real += channelValues[timeIndex] * Math.cos(angle)
      imag += channelValues[timeIndex] * Math.sin(angle)
    }
    return Math.sqrt(real * real + imag * imag) / n
  })

  const totals = EEG_BANDS.map(({ label, range }) => {
    const [low, high] = range
    const power = frequencies.reduce((sum, freq, idx) => {
      if (freq >= low && freq <= high) {
        return sum + spectrum[idx] * spectrum[idx]
      }
      return sum
    }, 0)
    return { label, value: power }
  })

  const totalPower = totals.reduce((sum, band) => sum + band.value, 0)
  if (totalPower <= 0) {
    return totals.map((item) => ({ label: item.label, value: 0 }))
  }

  return totals.map((item) => ({
    label: item.label,
    value: (item.value / totalPower) * 100,
  }))
}

function App() {
  const [activePage, setActivePage] = useState<PageKey>('Dashboard')
  const [datasetName, setDatasetName] = useState('No dataset loaded')
  const [rows, setRows] = useState<DataRow[]>([])
  const [columns, setColumns] = useState<string[]>([])
  const [columnMeta, setColumnMeta] = useState<ColumnMeta[]>([])
  const [chartType, setChartType] = useState<ChartType>('line')
  const [xColumn, setXColumn] = useState('')
  const [yColumns, setYColumns] = useState<string[]>([])
  const [chartTitle, setChartTitle] = useState('Interactive chart')
  const [xLabel, setXLabel] = useState('')
  const [yLabel, setYLabel] = useState('')
  const [legendVisible, setLegendVisible] = useState(true)
  const [aggregation, setAggregation] = useState<Aggregation>('mean')
  const [filterText, setFilterText] = useState('')
  const [sortKey, setSortKey] = useState('')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')
  const [message, setMessage] = useState('Upload a CSV to begin creating charts, examining data, and checking EEG-like patterns.')
  const [error, setError] = useState('')

  const numericColumns = useMemo(
    () => columnMeta.filter((column) => column.type === 'numeric').map((column) => column.name),
    [columnMeta],
  )
  const yColumn = yColumns[0] ?? ''

  const filteredRows = useMemo(() => {
    if (!filterText.trim()) return rows
    const keyword = filterText.toLowerCase()
    return rows.filter((row) => Object.values(row).some((value) => String(value ?? '').toLowerCase().includes(keyword)))
  }, [filterText, rows])

  const sortedRows = useMemo(() => {
    if (!sortKey) return filteredRows
    const output = [...filteredRows]
    output.sort((left, right) => {
      const leftValue = left[sortKey]
      const rightValue = right[sortKey]
      const leftNumber = toNumber(leftValue)
      const rightNumber = toNumber(rightValue)
      if (leftNumber !== null && rightNumber !== null) {
        return sortDirection === 'asc' ? leftNumber - rightNumber : rightNumber - leftNumber
      }
      const leftText = String(leftValue ?? '')
      const rightText = String(rightValue ?? '')
      return sortDirection === 'asc' ? leftText.localeCompare(rightText) : rightText.localeCompare(leftText)
    })
    return output
  }, [filteredRows, sortDirection, sortKey])

  const eegSummary = useMemo(() => {
    if (!rows.length) {
      return {
        present: false,
        message: 'No CSV loaded yet.',
        bandData: [] as ChartBandDatum[],
        timeColumn: '',
        signalColumns: [] as string[],
      }
    }

    const timeColumn = columns.find((column) => /time|sample|index|frame|ms/i.test(column)) ?? columns[0] ?? ''
    const validNumeric = columnMeta.filter((column) => column.type === 'numeric').map((column) => column.name)
    const signalColumns = validNumeric.filter((name) => name !== timeColumn)

    if (!signalColumns.length) {
      return {
        present: false,
        message: 'This CSV does not include a time axis and at least one numerical signal channel, so it is not yet reliable enough for EEG analysis.',
        bandData: [] as ChartBandDatum[],
        timeColumn,
        signalColumns: [] as string[],
      }
    }

    return {
      present: true,
      timeColumn,
      signalColumns,
      bandData: computeBandPower(rows.slice(0, 200)),
      message: 'EEG-like pattern detected. Relative band-power percentages are computed from the included bands only.',
    }
  }, [columnMeta, columns, rows])

  const chartData = useMemo<ChartDatum[]>(() => {
    if (!rows.length || !xColumn) return []

    if (chartType === 'pie') {
      if (!yColumn) return []
      const grouped = new Map<string, number>()
      filteredRows.forEach((row) => {
        const bucket = String(row[xColumn] ?? 'Unknown')
        const value = toNumber(row[yColumn])
        if (value === null) return
        grouped.set(bucket, (grouped.get(bucket) ?? 0) + value)
      })
      return Array.from(grouped.entries()).map(([name, value]) => ({ name, value }))
    }

    if (chartType === 'scatter') {
      return []
    }

    if (chartType === 'histogram') {
      const sourceColumn = yColumn || xColumn
      const source = filteredRows
        .map((row) => toNumber(row[sourceColumn]))
        .filter((value): value is number => value !== null && Number.isFinite(value))
      return buildHistogram(source)
    }

    if (!yColumns.length) return []

    const grouped = new Map<string, Map<string, number[]>>()
    filteredRows.forEach((row) => {
      const key = String(row[xColumn] ?? 'Unknown')
      const seriesValues = grouped.get(key) ?? new Map<string, number[]>()
      yColumns.forEach((column) => {
        const rawValue = row[column]
        const value = aggregation === 'count' && rawValue !== null && rawValue !== ''
          ? 1
          : toNumber(rawValue)
        if (value === null) return
        const values = seriesValues.get(column) ?? []
        values.push(value)
        seriesValues.set(column, values)
      })
      grouped.set(key, seriesValues)
    })

    const groupedSeries = Array.from(grouped.entries()).map(([name, seriesValues]) => {
      const point: ChartSeriesDatum = { name }
      yColumns.forEach((column) => {
        const values = seriesValues.get(column) ?? []
        if (!values.length) {
          point[column] = null
          return
        }
        const total = values.reduce((sum, value) => sum + value, 0)
        point[column] = aggregation === 'mean' ? total / values.length : total
      })
      return point
    })
    return sampleChartPoints(groupedSeries)
  }, [aggregation, chartType, filteredRows, rows.length, xColumn, yColumns, yColumn])

  const scatterSeries = useMemo<ScatterSeries[]>(
    () => yColumns.map((column) => ({
      column,
      data: sampleChartPoints(filteredRows
        .map((row) => ({ x: toNumber(row[xColumn]), y: toNumber(row[column]) }))
        .filter((point): point is ScatterDatum => point.x !== null && point.y !== null)),
    })),
    [filteredRows, xColumn, yColumns],
  )
  const chartDownsampled = chartType === 'scatter'
    ? scatterSeries.some((series) => series.data.length === MAX_CHART_POINTS && filteredRows.length > MAX_CHART_POINTS)
    : (chartType === 'line' || chartType === 'bar' || chartType === 'area') && filteredRows.length > MAX_CHART_POINTS

  function handleCSVUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = () => {
      try {
        const text = String(reader.result ?? '')
        if (!text.trim()) throw new Error('The uploaded file is empty.')

        const { columns: parsedColumns, rows: parsedRows } = parseCSVText(text)
        if (!parsedColumns.length || !parsedRows.length) {
          throw new Error('The file does not contain a valid header row and data rows.')
        }

        const summary = summarizeColumns(parsedColumns, parsedRows)
        setRows(parsedRows)
        setColumns(parsedColumns)
        setColumnMeta(summary)
        setDatasetName(file.name)
        setError('')
        setMessage(`Loaded ${file.name}. Choose columns and switch chart types without uploading the file again.`)

        const numericCandidates = summary.filter((column) => column.type === 'numeric').map((column) => column.name)
        const autoX = parsedColumns.find((column) => /time|sample|index|frame|ms/i.test(column)) ?? numericCandidates[0] ?? parsedColumns[0]
        const autoY = numericCandidates.filter((column) => column !== autoX)
        const initialYColumns = autoY.length ? autoY : [numericCandidates[0] ?? parsedColumns[0]]

        setXColumn(autoX)
        setYColumns(initialYColumns)
        setChartTitle(`${file.name} overview`)
        setXLabel(autoX)
        setYLabel(initialYColumns.join(', '))
      } catch (uploadError) {
        setRows([])
        setColumns([])
        setColumnMeta([])
        setDatasetName('No dataset loaded')
        setError(uploadError instanceof Error ? uploadError.message : 'The file could not be parsed.')
        setMessage('Upload failed. Ensure the file is a valid CSV with a header row and numeric content.')
      }
    }
    reader.readAsText(file)
  }

  function exportCurrentDataset() {
    if (!rows.length) return
    const csvHeader = columns.join(',')
    const csvRows = rows.map((row) => columns.map((column) => JSON.stringify(String(row[column] ?? ''))).join(','))
    const csv = [csvHeader, ...csvRows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${datasetName.replace(/\.[^/.]+$/, '') || 'dataset'}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  const validChartSelection = Boolean(xColumn && yColumns.length)

  function renderChart() {
    if (!validChartSelection) {
      return <div className="empty-state">Choose an X-axis and Y-axis to render a chart.</div>
    }

    if (chartType === 'pie') {
      const pieData = chartData.filter((entry): entry is { name: string; value: number } => typeof entry === 'object' && 'name' in entry && 'value' in entry && typeof entry.value === 'number')
      if (!pieData.length) {
        return <div className="empty-state">Pie charts need non-empty positive values. Try another column combination.</div>
      }
      if (pieData.some((entry) => entry.value < 0)) {
        return <div className="empty-state">Pie charts require non-negative values. Negative totals cannot be represented as a meaningful pie chart.</div>
      }
      if (pieData.reduce((total, entry) => total + entry.value, 0) <= 0) {
        return <div className="empty-state">Pie charts need a positive total in the selected numeric column.</div>
      }
      if (pieData.length > 50) {
        return <div className="empty-state">This X-axis has more than 50 categories, which would make a pie chart hard to read. Choose a categorical column with fewer distinct values.</div>
      }
      return (
        <ResponsiveContainer width="100%" height={360}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={60} outerRadius={120} paddingAngle={3} isAnimationActive={false}>
              {pieData.map((entry, index) => (
                <Cell key={`${entry.name}-${index}`} fill={SERIES_COLORS[index % SERIES_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value: number | string | readonly (number | string)[] | undefined) => {
              const numericValue = Array.isArray(value) ? Number(value[0] ?? 0) : Number(value ?? 0)
              return [numericValue.toFixed(2), yLabel || yColumn]
            }} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      )
    }

    if (chartType === 'scatter') {
      if (!scatterSeries.some((series) => series.data.length)) {
        return <div className="empty-state">Scatter plots need numeric values in the selected X column and at least one selected Y series.</div>
      }
      return (
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart>
            <CartesianGrid stroke="#1f3d5a" />
            <XAxis type="number" dataKey="x" name={xLabel || xColumn} tick={{ fill: '#9AAEC3' }} />
            <YAxis type="number" dataKey="y" name={yLabel || yColumn} tick={{ fill: '#9AAEC3' }} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(value: number | string | readonly (number | string)[] | undefined) => {
              const numericValue = Array.isArray(value) ? Number(value[0] ?? 0) : Number(value ?? 0)
              return numericValue.toFixed(2)
            }} />
            {scatterSeries.map((series, index) => (
              <Scatter
                key={series.column}
                name={series.column}
                data={series.data}
                fill={SERIES_COLORS[index % SERIES_COLORS.length]}
              />
            ))}
            {legendVisible && <Legend />}
          </ScatterChart>
        </ResponsiveContainer>
      )
    }

    if (chartType === 'histogram') {
      if (!chartData.length) {
        return <div className="empty-state">Histograms need numeric values in the first selected Y series.</div>
      }
      return (
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={chartData}>
            <CartesianGrid stroke="#1f3d5a" />
            <XAxis dataKey="name" tick={{ fill: '#9AAEC3' }} />
            <YAxis tick={{ fill: '#9AAEC3' }} />
            <Tooltip />
            <Bar dataKey="count" fill="#48D9F5" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )
    }

    if (chartType === 'bar') {
      return (
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={chartData}>
            <CartesianGrid stroke="#1f3d5a" />
            <XAxis dataKey="name" tick={{ fill: '#9AAEC3' }} />
            <YAxis tick={{ fill: '#9AAEC3' }} />
            <Tooltip />
            {yColumns.map((column, index) => (
              <Bar
                key={column}
                dataKey={column}
                fill={SERIES_COLORS[index % SERIES_COLORS.length]}
                radius={[6, 6, 0, 0]}
                isAnimationActive={false}
              />
            ))}
            {legendVisible && <Legend />}
          </BarChart>
        </ResponsiveContainer>
      )
    }

    if (chartType === 'area') {
      return (
        <ResponsiveContainer width="100%" height={360}>
          <AreaChart data={chartData}>
            <CartesianGrid stroke="#1f3d5a" />
            <XAxis dataKey="name" tick={{ fill: '#9AAEC3' }} />
            <YAxis tick={{ fill: '#9AAEC3' }} />
            <Tooltip />
            {yColumns.map((column, index) => (
              <Area
                key={column}
                type="monotone"
                dataKey={column}
                stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                fill={SERIES_COLORS[index % SERIES_COLORS.length]}
                fillOpacity={0.18}
                isAnimationActive={false}
              />
            ))}
            {legendVisible && <Legend />}
          </AreaChart>
        </ResponsiveContainer>
      )
    }

    return (
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={chartData}>
          <CartesianGrid stroke="#1f3d5a" />
          <XAxis dataKey="name" tick={{ fill: '#9AAEC3' }} />
          <YAxis tick={{ fill: '#9AAEC3' }} />
          <Tooltip />
          {yColumns.map((column, index) => (
            <Line
              key={column}
              type="monotone"
              dataKey={column}
              stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
              dot={false}
              isAnimationActive={false}
            />
          ))}
          {legendVisible && <Legend />}
        </LineChart>
      </ResponsiveContainer>
    )
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark"><BrainCircuit size={22} /></div>
          <div>
            <strong>NeuroBridge</strong>
            <small>Interactive EEG & data visualization</small>
          </div>
        </div>

        <nav className="nav-list">
          {(['Dashboard', 'Data Upload', 'Data Table', 'Chart Studio', 'EEG Explorer', 'Cognitive Insights'] as PageKey[]).map((page) => (
            <button
              key={page}
              type="button"
              className={`nav-item ${activePage === page ? 'active' : ''}`}
              onClick={() => setActivePage(page)}
            >
              {page}
            </button>
          ))}
        </nav>

        <div className="disclaimer-box">
          <Activity size={16} />
          <span>Educational and research prototype. Not clinical diagnostic software.</span>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <span className="eyebrow">Project status</span>
            <h1>{datasetName}</h1>
          </div>
          <div className="status-pill">{rows.length ? 'dataset ready' : 'waiting for CSV'}</div>
        </header>

        <div className="alert-bar">
          <FileText size={16} />
          <span>{message}</span>
        </div>

        {error && <div className="error-bar">{error}</div>}

        {activePage === 'Dashboard' && (
          <div className="page-block">
            <div className="cards-grid">
              <div className="metric-card">
                <div className="metric-icon"><Database size={18} /></div>
                <div>
                  <span>Rows</span>
                  <strong>{rows.length}</strong>
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-icon"><TableProperties size={18} /></div>
                <div>
                  <span>Columns</span>
                  <strong>{columns.length}</strong>
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-icon"><Gauge size={18} /></div>
                <div>
                  <span>Numeric</span>
                  <strong>{numericColumns.length}</strong>
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-icon"><BarChart3 size={18} /></div>
                <div>
                  <span>Chart type</span>
                  <strong>{chartType}</strong>
                </div>
              </div>
            </div>

            <div className="panel-grid">
              <div className="panel card">
                <h3>Quick start</h3>
                <div className="cta-row">
                  <label className="upload-button" htmlFor="csv-upload-dashboard">
                    <Upload size={16} /> Upload CSV
                    <input id="csv-upload-dashboard" type="file" accept=".csv" onChange={handleCSVUpload} hidden />
                  </label>
                  <button type="button" className="secondary-button" onClick={() => setActivePage('Chart Studio')}>
                    Open chart builder
                  </button>
                </div>
              </div>

              <div className="panel card">
                <h3>Supported workflow</h3>
                <ul className="bullet-list">
                  <li>CSV upload and metadata inspection</li>
                  <li>Interactive table with sorting and filtering</li>
                  <li>Line, bar, pie, scatter, histogram, and area charts</li>
                  <li>EEG band-power explorer with documented frequency boundaries</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activePage === 'Data Upload' && (
          <div className="page-block">
            <div className="panel card">
              <h3>Upload CSV</h3>
              <label className="upload-button file-upload-large" htmlFor="csv-upload-page">
                <Upload size={16} /> Select CSV file
                <input id="csv-upload-page" type="file" accept=".csv" onChange={handleCSVUpload} hidden />
              </label>
              <p className="note">The data stays in the browser so you can change chart settings without re-uploading.</p>
            </div>
          </div>
        )}

        {activePage === 'Data Table' && (
          <div className="page-block">
            <div className="panel card">
              <h3>Column summary</h3>
              <div className="column-summary-grid">
                {columnMeta.map((column) => (
                  <div key={column.name} className="column-pill">
                    <strong>{column.name}</strong>
                    <span>{column.type}</span>
                    <small>{column.missing} missing</small>
                  </div>
                ))}
              </div>
            </div>

            <div className="panel card">
              <div className="table-toolbar">
                <label>
                  Filter rows
                  <input value={filterText} onChange={(event) => setFilterText(event.target.value)} placeholder="Search values" />
                </label>
              </div>

              {sortedRows.length ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        {columns.map((column) => (
                          <th key={column}>
                            <button
                              type="button"
                              className="table-sort"
                              onClick={() => {
                                const nextDirection = sortKey === column && sortDirection === 'asc' ? 'desc' : 'asc'
                                setSortKey(column)
                                setSortDirection(nextDirection)
                              }}
                            >
                              {column}
                            </button>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {sortedRows.slice(0, 80).map((row, index) => (
                        <tr key={`${row[columns[0] ?? 'row']}-${index}`}>
                          {columns.map((column) => (
                            <td key={`${column}-${index}`}>{row[column] ?? '—'}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-state">No rows to display yet. Upload a CSV to begin.</div>
              )}
            </div>
          </div>
        )}

        {activePage === 'Chart Studio' && (
          <div className="page-block">
            <div className="panel card chart-builder-grid">
              <div className="controls-panel">
                <h3>Chart builder</h3>

                <label>
                  Chart type
                  <select value={chartType} onChange={(event) => setChartType(event.target.value as ChartType)}>
                    <option value="line">Line chart</option>
                    <option value="bar">Bar chart</option>
                    <option value="pie">Pie chart</option>
                    <option value="scatter">Scatter plot</option>
                    <option value="histogram">Histogram</option>
                    <option value="area">Area chart</option>
                  </select>
                </label>

                <label>
                  X-axis
                  <select value={xColumn} onChange={(event) => setXColumn(event.target.value)}>
                    <option value="">Select a column</option>
                    {columns.map((column) => (
                      <option key={column} value={column}>{column}</option>
                    ))}
                  </select>
                </label>

                <fieldset className="series-selector">
                  <legend>Y-axis series</legend>
                  {columns.length ? (
                    <div className="series-options">
                      {columns.map((column) => (
                        <label key={column} className="series-option">
                          <input
                            type="checkbox"
                            checked={yColumns.includes(column)}
                            onChange={(event) => {
                              const { checked } = event.currentTarget
                              setYColumns((selected) => checked
                                ? [...selected, column]
                                : selected.filter((item) => item !== column))
                            }}
                          />
                          <span>{column}</span>
                        </label>
                      ))}
                    </div>
                  ) : (
                    <p className="note">Upload a CSV to choose data series.</p>
                  )}
                  <small>Select multiple columns for line, bar, area, and scatter charts. Pie and histogram use the first selected series.</small>
                </fieldset>

                <label>
                  Aggregation
                  <select value={aggregation} onChange={(event) => setAggregation(event.target.value as Aggregation)}>
                    <option value="count">Count</option>
                    <option value="sum">Sum</option>
                    <option value="mean">Mean</option>
                  </select>
                </label>

                <label>
                  Chart title
                  <input value={chartTitle} onChange={(event) => setChartTitle(event.target.value)} />
                </label>

                <label>
                  X label
                  <input value={xLabel} onChange={(event) => setXLabel(event.target.value)} />
                </label>

                <label>
                  Y label
                  <input value={yLabel} onChange={(event) => setYLabel(event.target.value)} />
                </label>

                <label className="checkbox-row">
                  <input type="checkbox" checked={legendVisible} onChange={() => setLegendVisible((value) => !value)} />
                  Show legend
                </label>

                <button type="button" className="secondary-button" onClick={exportCurrentDataset} disabled={!rows.length}>
                  <Download size={16} /> Download CSV
                </button>
              </div>

              <div className="chart-panel">
                <div className="chart-header">
                  <div>
                    <span className="eyebrow">Chart preview</span>
                    <h3>{chartTitle || 'Interactive chart'}</h3>
                  </div>
                </div>
                {rows.length ? renderChart() : <div className="empty-state">Upload a CSV to generate a chart preview.</div>}
                {rows.length > 0 && chartDownsampled && (
                  <p className="note">For responsiveness, this chart shows up to {MAX_CHART_POINTS} evenly spaced points. All rows remain available in the table and CSV download.</p>
                )}
              </div>
            </div>
          </div>
        )}

        {activePage === 'EEG Explorer' && (
          <div className="page-block">
            <div className="panel card">
              <h3>EEG signal explorer</h3>
              {eegSummary.present ? (
                <>
                  <p className="note">{eegSummary.message}</p>
                  <div className="info-grid">
                    <div><strong>Time column:</strong> {eegSummary.timeColumn}</div>
                    <div><strong>Signal channels:</strong> {eegSummary.signalColumns.length ? eegSummary.signalColumns.join(', ') : 'No numeric signal channels detected'}</div>
                  </div>

                  <div className="chart-wrap compact-chart">
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={eegSummary.bandData}>
                        <CartesianGrid stroke="#1f3d5a" />
                        <XAxis dataKey="label" tick={{ fill: '#9AAEC3' }} />
                        <YAxis tick={{ fill: '#9AAEC3' }} />
                        <Tooltip formatter={(value: number | string | readonly (number | string)[] | undefined) => {
                          const numericValue = Array.isArray(value) ? Number(value[0] ?? 0) : Number(value ?? 0)
                          return [`${numericValue.toFixed(2)}%`, 'Relative power']
                        }} />
                        <Bar dataKey="value" fill="#48D9F5" radius={[6, 6, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <p className="note">
                    Frequency boundaries used here are delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, and gamma 30–45 Hz. Percentage = (selected band power / total included-band power) × 100.
                  </p>
                </>
              ) : (
                <div className="empty-state">{eegSummary.message}</div>
              )}
            </div>
          </div>
        )}

        {activePage === 'Cognitive Insights' && (
          <CognitiveInsightsPanel rows={rows} columns={columns} datasetName={datasetName} />
        )}
      </main>
    </div>
  )
}

export default App
