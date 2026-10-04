import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const AXIS_TICK = { fill: '#798984', fontSize: 10 }
const CALIBRATION_BINS = 10
const GAINS_STEP = 0.02
const IMPORTANCE_ROWS = 8

const percent = (value, digits = 0) => (Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—')

function ExplainTooltip({ active, payload, label, formatLabel, rows }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <strong>{formatLabel ? formatLabel(label, payload[0].payload) : label}</strong>
      {rows(payload[0].payload).map(([name, value]) => <span key={name}>{name}: {value}</span>)}
    </div>
  )
}

function EmptyState({ children }) {
  return <div className="explain-empty">{children}</div>
}

export function DriverImportancePanel({ importance, labels }) {
  const data = importance.slice(0, IMPORTANCE_ROWS).map((item) => ({ ...item, label: labels[item.feature] }))
  return (
    <article className="chart-panel">
      <div className="panel-heading"><div><p className="eyebrow">MODEL EXPLAINABILITY</p><h2>What drives churn risk</h2></div><span className="panel-context">mean |SHAP| · whole partition</span></div>
      <div className="chart-area explain-chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 14, left: 4, bottom: 0 }}>
            <CartesianGrid stroke="#e8eeeb" horizontal={false} />
            <XAxis type="number" tickLine={false} axisLine={false} tick={AXIS_TICK} tickFormatter={(value) => value.toFixed(1)} />
            <YAxis type="category" dataKey="label" width={132} interval={0} tickLine={false} axisLine={false} tick={{ ...AXIS_TICK, fill: '#586a64' }} />
            <Tooltip content={<ExplainTooltip rows={(item) => [['Mean impact', `${item.meanAbs.toFixed(2)} log-odds`], ['Raises risk for', `${percent(item.raisesShare)} of customers`]]} />} />
            <Bar dataKey="meanAbs" radius={[0, 3, 3, 0]} isAnimationActive={false}>
              {data.map((item, index) => <Cell key={item.feature} fill={index < 4 ? '#087f70' : '#9fbdb3'} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="explain-note">Average size of each feature&apos;s push on the XGBoost score (log-odds). Click a customer below to see their own drivers.</p>
    </article>
  )
}

function calibrationBins(rows) {
  const bins = Array.from({ length: CALIBRATION_BINS }, () => ({ count: 0, predicted: 0, churned: 0 }))
  let brier = 0
  for (const row of rows) {
    const bin = bins[Math.min(CALIBRATION_BINS - 1, Math.floor(row.churnScore * CALIBRATION_BINS))]
    const outcome = Number(row.churned)
    bin.count += 1
    bin.predicted += row.churnScore
    bin.churned += outcome
    brier += (row.churnScore - outcome) ** 2
  }
  return {
    brier: rows.length ? brier / rows.length : null,
    points: bins
      .map((bin, index) => ({
        range: `${index * 10}–${index * 10 + 10}%`,
        count: bin.count,
        predicted: bin.count ? bin.predicted / bin.count : null,
        observed: bin.count ? bin.churned / bin.count : null,
      }))
      .filter((bin) => bin.count),
  }
}

export function CalibrationPanel({ rows }) {
  const labeled = rows.length > 0 && rows[0].churned !== null
  const { brier, points } = useMemo(() => (labeled ? calibrationBins(rows) : { brier: null, points: [] }), [rows, labeled])
  return (
    <article className="chart-panel">
      <div className="panel-heading"><div><p className="eyebrow">MODEL EXPLAINABILITY</p><h2>Calibration</h2></div><span className="panel-context">{labeled ? `Brier score ${brier?.toFixed(4) ?? '—'}` : 'needs churn labels'}</span></div>
      {labeled && points.length ? (
        <div className="chart-area explain-chart">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={points} margin={{ top: 8, right: 12, left: -18, bottom: 2 }}>
              <CartesianGrid stroke="#e8eeeb" />
              <XAxis type="number" dataKey="predicted" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickLine={false} axisLine={false} tick={AXIS_TICK} tickFormatter={(value) => percent(value)} />
              <YAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickLine={false} axisLine={false} tick={AXIS_TICK} tickFormatter={(value) => percent(value)} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#b9c9c3" strokeDasharray="4 4" ifOverflow="extendDomain" />
              <Tooltip content={<ExplainTooltip formatLabel={(_, item) => `Score bin ${item.range}`} rows={(item) => [['Avg. predicted', percent(item.predicted, 1)], ['Observed churn', percent(item.observed, 1)], ['Customers', item.count.toLocaleString('en-US')]]} />} />
              <Line type="linear" dataKey="observed" stroke="#d4a12e" strokeWidth={2.2} dot={{ r: 3, fill: '#d4a12e' }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <EmptyState>{labeled ? 'No customers match these filters.' : 'New customers have no observed churn outcome yet, so calibration cannot be checked.'}</EmptyState>
      )}
      <p className="explain-note">Observed churn rate per 10-point score bin. Points on the dashed diagonal mean the score can be read as a probability.</p>
    </article>
  )
}

function gainsCurve(rows) {
  const labeled = rows[0]?.churned !== null
  const totalChurners = labeled ? rows.reduce((sum, row) => sum + Number(row.churned), 0) : 0
  const totalValue = rows.reduce((sum, row) => sum + row.expectedLoss, 0)
  const points = [{ share: 0, churners: 0, value: 0, random: 0 }]
  let churners = 0
  let value = 0
  let next = GAINS_STEP
  rows.forEach((row, index) => {
    churners += Number(row.churned === true)
    value += row.expectedLoss
    const share = (index + 1) / rows.length
    if (share + 1e-9 >= next || index === rows.length - 1) {
      points.push({
        share,
        churners: totalChurners ? churners / totalChurners : null,
        value: totalValue ? value / totalValue : null,
        random: share,
      })
      while (next <= share + 1e-9) next += GAINS_STEP
    }
  })
  const at = (share) => points.find((point) => point.share + 1e-9 >= share)
  return { labeled, points, at10: at(0.1), at20: at(0.2) }
}

export function GainsPanel({ rows }) {
  const { labeled, points, at10, at20 } = useMemo(() => gainsCurve(rows), [rows])
  return (
    <article className="chart-panel">
      <div className="panel-heading"><div><p className="eyebrow">MODEL EXPLAINABILITY</p><h2>Cumulative gains</h2></div><span className="panel-context">contacting customers by score, highest first</span></div>
      {rows.length ? (
        <>
          <div className="chart-area explain-chart">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={points} margin={{ top: 8, right: 12, left: -18, bottom: 2 }}>
                <CartesianGrid stroke="#e8eeeb" />
                <XAxis type="number" dataKey="share" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tickLine={false} axisLine={false} tick={AXIS_TICK} tickFormatter={(value) => percent(value)} />
                <YAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickLine={false} axisLine={false} tick={AXIS_TICK} tickFormatter={(value) => percent(value)} />
                <Tooltip content={<ExplainTooltip formatLabel={(share) => `Top ${percent(share)} contacted`} rows={(item) => [
                  ...(labeled ? [['Churners reached', percent(item.churners, 1)]] : []),
                  ['Expected loss covered', percent(item.value, 1)],
                ]} />} />
                <Line dataKey="random" stroke="#b9c9c3" strokeDasharray="4 4" dot={false} isAnimationActive={false} />
                {labeled && <Line dataKey="churners" stroke="#087f70" strokeWidth={2.2} dot={false} isAnimationActive={false} />}
                <Line dataKey="value" stroke="#d4a12e" strokeWidth={2.2} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="explain-stats">
            {labeled && <div><span>CHURNERS IN TOP 20%</span><b>{percent(at20?.churners, 1)}</b></div>}
            {labeled && <div><span>LIFT @ TOP 10%</span><b>{at10?.churners == null ? '—' : `${(at10.churners / at10.share).toFixed(1)}×`}</b></div>}
            <div><span>LOSS COVERED BY TOP 20%</span><b>{percent(at20?.value, 1)}</b></div>
          </div>
        </>
      ) : (
        <EmptyState>No customers match these filters.</EmptyState>
      )}
      <div className="chart-legend">{labeled && <span><i className="legend-line teal-line" /> Churners reached</span>}<span><i className="legend-line gold-line" /> Expected loss covered</span><span><i className="legend-line dashed-line" /> Random targeting</span></div>
    </article>
  )
}

export function CustomerDrivers({ drivers, error }) {
  if (error) return <p className="driver-status">Could not load drivers: {error}</p>
  if (!drivers) return <p className="driver-status">Loading drivers…</p>
  const scale = Math.max(...drivers.map((driver) => Math.abs(driver.contribution)), 1e-6)
  return (
    <ul className="driver-list" aria-label="Top drivers for this customer">
      {drivers.map((driver) => {
        const raises = driver.contribution > 0
        return (
          <li key={driver.name}>
            <span className="driver-label">{driver.label}<small>{formatDriverValue(driver)}</small></span>
            <span className="driver-bar"><i className={raises ? 'raises' : 'lowers'} style={{ width: `${(Math.abs(driver.contribution) / scale) * 100}%` }} /></span>
            <span className={`driver-effect ${raises ? 'raises' : 'lowers'}`}>{raises ? '+' : '−'}{Math.abs(driver.contribution).toFixed(2)} {raises ? 'raises' : 'lowers'} risk</span>
          </li>
        )
      })}
    </ul>
  )
}

function formatDriverValue({ name, value, binary }) {
  if (binary) return value ? 'yes' : 'no'
  if (['Balance', 'Income', 'Outstanding Loans'].includes(name)) {
    return value.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
  }
  return value.toLocaleString('en-US', { maximumFractionDigits: 0 })
}
