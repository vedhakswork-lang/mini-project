export type CognitiveRow = Record<string, string | number | null>

export type ChannelActivity = {
  channel: string
  rms: number
  activity: number
}

export type AttentionShift = {
  channel: string
  window: number
  changePercent: number
} | null

export type EventActivity = {
  event: string
  meanAmplitude: number
  peakAmplitude: number
}

export type CognitiveInsights = {
  channels: ChannelActivity[]
  cognitiveLoad: number
  loadLevel: 'Low' | 'Medium' | 'High'
  stressIndex: number
  focusIndex: number
  stabilityScore: number
  attentionShift: AttentionShift
  eventActivity: EventActivity[]
  abnormalWindows: number
  fingerprint: string
  windowCount: number
}

function numericValue(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function mean(values: number[]): number {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0
}

function rms(values: number[]): number {
  if (!values.length) return 0
  const center = mean(values)
  return Math.sqrt(mean(values.map((value) => (value - center) ** 2)))
}

function isEegChannel(column: string): boolean {
  return /^eeg(?:[_ -]?ch)?[_ -]?\d+$/i.test(column)
    || /^(?:fp|af|f|ft|fc|c|t|tp|cp|p|po|o)(?:z|[0-9])/i.test(column)
}

function makeFingerprint(rows: CognitiveRow[], channels: string[]): string {
  let first = 0x811c9dc5
  let second = 0x9e3779b9
  const input = rows.map((row) => channels.map((channel) => String(row[channel] ?? '')).join(',')).join(';')
  for (let index = 0; index < input.length; index += 1) {
    const code = input.charCodeAt(index)
    first = Math.imul(first ^ code, 0x01000193)
    second = Math.imul(second ^ code, 0x85ebca6b)
  }
  const fingerprint = `${(first >>> 0).toString(16).padStart(8, '0')}${(second >>> 0).toString(16).padStart(8, '0')}`
  return `BRN-${fingerprint.slice(0, 8).toUpperCase()}`
}

export function analyzeEegCsv(rows: CognitiveRow[], columns: string[]): CognitiveInsights | null {
  if (rows.length < 2) return null

  const signalColumns = columns.filter((column) => isEegChannel(column))
  if (!signalColumns.length) return null

  const seriesByChannel = new Map<string, Array<number | null>>()
  for (const column of signalColumns) {
    const values = rows.map((row) => numericValue(row[column]))
    if (values.filter((value) => value !== null).length >= 2) seriesByChannel.set(column, values)
  }
  if (!seriesByChannel.size) return null

  const channels = Array.from(seriesByChannel, ([channel, values]) => ({
    channel,
    rms: rms(values.filter((value): value is number => value !== null)),
    activity: 0,
  })).sort((left, right) => right.rms - left.rms)
  const maximumRms = channels[0]?.rms ?? 0
  channels.forEach((channel) => {
    channel.activity = maximumRms > 0 ? channel.rms / maximumRms : 0
  })

  const windowSize = Math.max(2, Math.ceil(rows.length / 10))
  const overallWindowRms: number[] = []
  const windowCount = Math.ceil(rows.length / windowSize)
  for (let window = 0; window < windowCount; window += 1) {
    const windowValues: number[] = []
    for (const values of seriesByChannel.values()) {
      const section = values.slice(window * windowSize, (window + 1) * windowSize)
        .filter((value): value is number => value !== null)
      if (section.length > 1) windowValues.push(rms(section))
    }
    if (windowValues.length) overallWindowRms.push(mean(windowValues))
  }

  const windowMean = mean(overallWindowRms)
  const coefficientOfVariation = windowMean > 0
    ? Math.sqrt(mean(overallWindowRms.map((value) => (value - windowMean) ** 2))) / windowMean
    : 0
  const variability = Math.max(0, Math.min(100, coefficientOfVariation * 100))
  const cognitiveLoad = Math.round(variability)
  const loadLevel = cognitiveLoad < 34 ? 'Low' : cognitiveLoad < 67 ? 'Medium' : 'High'

  let attentionShift: AttentionShift = null
  for (const channel of signalColumns) {
    const values = seriesByChannel.get(channel)
    if (!values) continue
    const windowRms: number[] = []
    for (let start = 0; start < rows.length; start += windowSize) {
      const section = values.slice(start, start + windowSize)
        .filter((value): value is number => value !== null)
      if (section.length > 1) windowRms.push(rms(section))
    }
    for (let index = 1; index < windowRms.length; index += 1) {
      const baseline = Math.max(windowRms[index - 1], Number.EPSILON)
      const changePercent = (Math.abs(windowRms[index] - windowRms[index - 1]) / baseline) * 100
      if (changePercent >= 50 && (!attentionShift || changePercent > attentionShift.changePercent)) {
        attentionShift = { channel, window: index + 1, changePercent: Math.round(changePercent) }
      }
    }
  }

  const windowMeanActivity = mean(overallWindowRms)
  const windowStdActivity = Math.sqrt(mean(overallWindowRms.map((window) => (window - windowMeanActivity) ** 2)))
  const abnormalThreshold = windowMeanActivity + 3 * windowStdActivity
  const abnormalWindows = windowMeanActivity > 0
    ? overallWindowRms.filter((window) => window > abnormalThreshold).length
    : 0

  const eventColumn = columns.find((column) => /event|label|condition|task|class/i.test(column))
  const eventMap = new Map<string, number[]>()
  if (eventColumn) {
    rows.forEach((row) => {
      const event = String(row[eventColumn] ?? '').trim()
      if (!event) return
      const amplitude = signalColumns
        .map((channel) => numericValue(row[channel]))
        .filter((value): value is number => value !== null)
        .reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0)
      const amplitudes = eventMap.get(event) ?? []
      amplitudes.push(amplitude)
      eventMap.set(event, amplitudes)
    })
  }
  const eventActivity = Array.from(eventMap, ([event, amplitudes]) => ({
    event,
    meanAmplitude: mean(amplitudes),
    peakAmplitude: amplitudes.reduce((maximum, value) => Math.max(maximum, value), 0),
  })).sort((left, right) => right.peakAmplitude - left.peakAmplitude)

  return {
    channels,
    cognitiveLoad,
    loadLevel,
    stressIndex: Math.round(variability),
    focusIndex: Math.round(100 - variability),
    stabilityScore: Math.round(100 - variability),
    attentionShift,
    eventActivity,
    abnormalWindows,
    fingerprint: makeFingerprint(rows, Array.from(seriesByChannel.keys())),
    windowCount,
  }
}

export function answerInsight(question: string, insights: CognitiveInsights): string {
  const query = question.toLowerCase()
  const strongest = insights.channels[0]
  const peakEvent = insights.eventActivity[0]
  const highestMeanEvent = [...insights.eventActivity].sort((a, b) => b.meanAmplitude - a.meanAmplitude)[0]

  if (/strongest|most active|active channel|channel.*active/.test(query)) {
    return strongest
      ? `${strongest.channel} has the highest RMS amplitude (${strongest.rms.toFixed(3)}) among the detected EEG channels in this file.`
      : 'No numeric EEG channel data is available to compare.'
  }
  if (/event|peak|amplitude/.test(query) && insights.eventActivity.length) {
    const event = /higher|mean|average/.test(query) ? highestMeanEvent : peakEvent
    return event
      ? `${event.event} has ${event === peakEvent ? 'the largest observed' : 'the highest mean'} absolute amplitude (${(event === peakEvent ? event.peakAmplitude : event.meanAmplitude).toFixed(3)}) across available EEG channels.`
      : 'No event activity could be calculated.'
  }
  if (/abnormal|outlier|anomal/.test(query)) {
    return insights.abnormalWindows
      ? `${insights.abnormalWindows} channel-window amplitude outlier(s) exceeded the dataset's mean plus three standard deviations. This is a statistical flag, not a medical abnormality diagnosis.`
      : 'No amplitude outliers exceeded the dataset-based three-standard-deviation threshold. This does not rule out clinical abnormalities.'
  }
  if (/attention|shift/.test(query)) {
    return insights.attentionShift
      ? `Attention Shift Detected in ${insights.attentionShift.channel}: a ${insights.attentionShift.changePercent}% RMS change was found around window ${insights.attentionShift.window}. This indicates an amplitude shift only; attention cannot be confirmed from this heuristic.`
      : 'No adjacent-window amplitude change exceeded the 50% screening threshold. This does not measure attention directly.'
  }
  return `The strongest detected channel is ${strongest?.channel ?? 'unavailable'}. Cognitive activity variability is ${insights.cognitiveLoad}% (${insights.loadLevel}); these dataset-derived indicators are experimental and not clinically validated.`
}
