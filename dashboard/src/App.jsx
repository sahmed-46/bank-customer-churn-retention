import { useDeferredValue, useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  ArrowDownToLine,
  ArrowRight,
  BadgeCheck,
  Banknote,
  ChartNoAxesCombined,
  ChevronDown,
  CircleHelp,
  Crosshair,
  Filter,
  RotateCcw,
  ShieldAlert,
  UsersRound,
} from 'lucide-react'
import { driversForRow, loadDashboardData, loadDrivers } from './data'
import { CalibrationPanel, CustomerDrivers, DriverImportancePanel, GainsPanel } from './Explainability'
import './App.css'

const PARTITIONS = [
  { id: 'train', label: 'Training', note: 'In-sample' },
  { id: 'validation', label: 'Validation', note: 'Model selection' },
  { id: 'test', label: 'Test', note: 'Held out' },
  { id: 'inference', label: 'New customers', note: 'Unlabeled batch' },
]

const RISK_BANDS = [
  { id: 'Low', min: 0, max: 0.05, color: '#b9c9c3' },
  { id: 'Watch', min: 0.05, max: 0.5, color: '#7399aa' },
  { id: 'High', min: 0.5, max: 0.9, color: '#e6a544' },
  { id: 'Critical', min: 0.9, max: 1.01, color: '#c84e37' },
]

const ACTION_COLORS = {
  no_action: '#a9bbb5',
  general_retention_email: '#429989',
  cross_sell_bundle_offer: '#d9a33c',
  onboarding_support_and_welcome_offer: '#627ea4',
}

const SEGMENT_COLORS = {
  mass_market: '#087f70',
  affluent_multi_product: '#d4a12e',
  single_product_low_engagement: '#6b7e9b',
  new_customer: '#d56a4f',
  high_complaint_risk: '#ad4f68',
  Corporate: '#087f70',
  Retail: '#d4a12e',
  SME: '#6b7e9b',
}

const FILTER_DEFAULTS = {
  segment: 'All segments',
  action: 'All actions',
  products: 'All products',
  riskBand: 'All risk bands',
  tenure: [1, 30],
  balance: [0, 250000],
}

const BENCHMARKS = {
  Logistic: {
    validation: { auc: 0.9963011625798448, prAuc: 0.9748806729015573, recall5: 0.41, recall10: 0.792, recall20: 1 },
    test: { auc: 0.9965788086785828, prAuc: 0.9778780213216426 },
  },
  XGBoost: {
    validation: { auc: 0.9996861698654125, prAuc: 0.997831176653169, recall5: 0.41, recall10: 0.82, recall20: 1 },
    test: { auc: 0.9998131505715718, prAuc: 0.9986731283975931 },
  },
}

const numberFormat = new Intl.NumberFormat('en-US')
const currencyFormat = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
})

function getRiskBand(score) {
  return RISK_BANDS.find((band) => score >= band.min && score < band.max)?.id ?? 'Critical'
}

function formatPercent(value, digits = 1) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—'
}

// Metric helpers expect rows sorted by churnScore, highest first.
function calculateAuc(sorted) {
  const positiveCount = sorted.reduce((sum, row) => sum + Number(row.churned), 0)
  const negativeCount = sorted.length - positiveCount
  if (!positiveCount || !negativeCount) return null

  let negativesAbove = 0
  let orderedPairs = 0
  for (let start = 0; start < sorted.length;) {
    let end = start + 1
    while (end < sorted.length && sorted[end].churnScore === sorted[start].churnScore) end += 1
    let positives = 0
    for (let index = start; index < end; index += 1) positives += Number(sorted[index].churned)
    const negatives = end - start - positives
    orderedPairs += positives * (negativeCount - negativesAbove - negatives) + 0.5 * positives * negatives
    negativesAbove += negatives
    start = end
  }

  return orderedPairs / (positiveCount * negativeCount)
}

function calculateAveragePrecision(sorted) {
  const positiveCount = sorted.reduce((sum, row) => sum + Number(row.churned), 0)
  if (!positiveCount) return null

  let truePositive = 0
  let falsePositive = 0
  let previousRecall = 0
  let averagePrecision = 0
  for (let start = 0; start < sorted.length;) {
    let end = start + 1
    while (end < sorted.length && sorted[end].churnScore === sorted[start].churnScore) end += 1
    for (let index = start; index < end; index += 1) {
      if (sorted[index].churned) truePositive += 1
      else falsePositive += 1
    }
    const recall = truePositive / positiveCount
    const precision = truePositive / (truePositive + falsePositive)
    averagePrecision += (recall - previousRecall) * precision
    previousRecall = recall
    start = end
  }
  return averagePrecision
}

function calculateRecallAt(sorted, fraction) {
  const positiveCount = sorted.reduce((sum, row) => sum + Number(row.churned), 0)
  if (!positiveCount) return null
  const cutoff = Math.max(1, Math.ceil(sorted.length * fraction))
  return sorted.slice(0, cutoff).reduce((sum, row) => sum + Number(row.churned), 0) / positiveCount
}

function calculateMetrics(rows) {
  const labeled = rows.filter((row) => row.churned !== null)
  if (!labeled.length) return null

  const truePositive = labeled.filter((row) => row.churned && row.churnScore >= 0.5).length
  const falsePositive = labeled.filter((row) => !row.churned && row.churnScore >= 0.5).length
  const falseNegative = labeled.filter((row) => row.churned && row.churnScore < 0.5).length
  const precision = truePositive + falsePositive ? truePositive / (truePositive + falsePositive) : 0
  const recall = truePositive + falseNegative ? truePositive / (truePositive + falseNegative) : 0
  return {
    auc: calculateAuc(labeled),
    prAuc: calculateAveragePrecision(labeled),
    recall5: calculateRecallAt(labeled, 0.05),
    recall10: calculateRecallAt(labeled, 0.1),
    recall20: calculateRecallAt(labeled, 0.2),
    accuracy: labeled.length ? (labeled.length - falsePositive - falseNegative) / labeled.length : null,
    precision,
    recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    prevalence: labeled.reduce((sum, row) => sum + Number(row.churned), 0) / labeled.length,
  }
}

function matchesCohort(row, filters, search, skipFilter) {
  return (skipFilter === 'segment' || filters.segment === FILTER_DEFAULTS.segment || row.customerSegment === filters.segment)
    && (skipFilter === 'action' || filters.action === FILTER_DEFAULTS.action || row.action === filters.action)
    && (skipFilter === 'products' || filters.products === FILTER_DEFAULTS.products || row.products === Number(filters.products))
    && (skipFilter === 'riskBand' || filters.riskBand === FILTER_DEFAULTS.riskBand || getRiskBand(row.churnScore) === filters.riskBand)
    && row.tenure >= filters.tenure[0]
    && row.tenure <= filters.tenure[1]
    && row.balance >= filters.balance[0]
    && row.balance <= filters.balance[1]
    && (!search || `${row.customerSegment} ${row.segment} ${row.action} ${row.sourceIndex + 1}`.toLowerCase().includes(search))
}

function App() {
  const [data, setData] = useState(null)
  const [partition, setPartition] = useState('test')
  const [filters, setFilters] = useState(FILTER_DEFAULTS)
  const [search, setSearch] = useState('')
  const [selectedRow, setSelectedRow] = useState(null)
  const [showBenchmarks, setShowBenchmarks] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [selectedDrivers, setSelectedDrivers] = useState({ key: null, drivers: null, error: '' })

  useEffect(() => {
    let cancelled = false
    loadDashboardData()
      .then((loaded) => {
        if (cancelled) return
        setData(loaded)
        setIsLoading(false)
      })
      .catch((error) => {
        if (!cancelled) {
          setLoadError(error.message)
          setIsLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [])

  const datasetName = partition === 'inference' ? 'inference' : 'labeled'
  const activeRows = useMemo(() => {
    if (!data) return []
    const rows = partition === 'inference'
      ? data.inference
      : data.labeled.slice(data.manifest.partitions[partition].start, data.manifest.partitions[partition].end)
    return [...rows].sort((left, right) => right.churnScore - left.churnScore)
  }, [data, partition])

  const selectedKey = selectedRow ? `${datasetName}-${selectedRow.sourceIndex}` : null
  useEffect(() => {
    if (!selectedRow || !data) return
    let cancelled = false
    loadDrivers(data.manifest, datasetName)
      .then((drivers) => {
        if (!cancelled) setSelectedDrivers({ key: selectedKey, drivers: driversForRow(drivers, data.manifest, datasetName, selectedRow.sourceIndex), error: '' })
      })
      .catch((error) => {
        if (!cancelled) setSelectedDrivers({ key: selectedKey, drivers: null, error: error.message })
      })
    return () => { cancelled = true }
  }, [data, datasetName, selectedRow, selectedKey])
  const currentDrivers = selectedDrivers.key === selectedKey ? selectedDrivers : { drivers: null, error: '' }
  const deferredSearch = useDeferredValue(search.trim().toLowerCase())
  const maxTenure = useMemo(() => activeRows.reduce((max, row) => Math.max(max, row.tenure), 0), [activeRows])
  const maxBalance = useMemo(() => activeRows.reduce((max, row) => Math.max(max, row.balance), 0), [activeRows])

  const filterOptions = useMemo(() => ({
    segments: [...new Set(activeRows.map((row) => row.customerSegment))].sort(),
    actions: [...new Set(activeRows.map((row) => row.action))].sort(),
    products: [...new Set(activeRows.map((row) => row.products).filter(Boolean))].sort((a, b) => a - b),
  }), [activeRows])

  const filteredRows = useMemo(() => activeRows.filter((row) => matchesCohort(row, filters, deferredSearch)), [activeRows, filters, deferredSearch])

  const metrics = useMemo(() => calculateMetrics(filteredRows), [filteredRows])
  const segmentBaseRows = useMemo(() => activeRows.filter((row) => matchesCohort(row, filters, deferredSearch, 'segment')), [activeRows, filters, deferredSearch])
  const actionBaseRows = useMemo(() => activeRows.filter((row) => matchesCohort(row, filters, deferredSearch, 'action')), [activeRows, filters, deferredSearch])
  const riskBandBaseCount = useMemo(() => activeRows.reduce((count, row) => count + Number(matchesCohort(row, filters, deferredSearch, 'riskBand')), 0), [activeRows, filters, deferredSearch])

  const segmentChartData = useMemo(() => {
    const counts = new Map()
    for (const row of segmentBaseRows) counts.set(row.customerSegment, (counts.get(row.customerSegment) || 0) + 1)
    return [...counts.entries()].map(([name, count]) => ({ name, count, fill: SEGMENT_COLORS[name] || '#758981' }))
  }, [segmentBaseRows])

  const actionChartData = useMemo(() => {
    const counts = new Map()
    for (const row of actionBaseRows) counts.set(row.action, (counts.get(row.action) || 0) + 1)
    return [...counts.entries()]
      .map(([action, count]) => ({ action, count, fill: ACTION_COLORS[action] || '#758981' }))
      .sort((left, right) => right.count - left.count)
  }, [actionBaseRows])

  const scoreTrend = useMemo(() => {
    const bins = RISK_BANDS.map((band) => ({
      range: band.id,
      min: band.min,
      max: band.max,
      count: 0,
      churned: 0,
    }))
    for (const row of filteredRows) {
      const index = RISK_BANDS.findIndex((band) => row.churnScore >= band.min && row.churnScore < band.max)
      if (index < 0) continue
      bins[index].count += 1
      bins[index].churned += Number(row.churned === true)
    }
    return bins.map((bin) => ({ ...bin, actualRate: bin.count && partition !== 'inference' ? bin.churned / bin.count : null }))
  }, [filteredRows, partition])

  const rankedRows = useMemo(() => [...filteredRows]
    .sort((left, right) => right.expectedLoss - left.expectedLoss)
    .slice(0, 12), [filteredRows])

  const currentPartition = PARTITIONS.find((item) => item.id === partition)
  const totalExpectedLoss = filteredRows.reduce((sum, row) => sum + row.expectedLoss, 0)
  const recommendedRows = filteredRows.filter((row) => row.action !== 'no_action').length
  const targetedValue = filteredRows.filter((row) => row.action !== 'no_action')
    .reduce((sum, row) => sum + row.expectedLoss, 0)

  const setFilter = (key, value) => setFilters((previous) => ({ ...previous, [key]: value }))
  const toggleFilter = (key, value, defaultValue) => {
    setFilter(key, filters[key] === value ? defaultValue : value)
  }
  const resetFilters = () => {
    setFilters(FILTER_DEFAULTS)
    setSearch('')
    setSelectedRow(null)
  }

  const exportFilteredRows = () => {
    const columns = ['sourceIndex', 'customerSegment', 'segment', 'action', 'churnScore', 'clv', 'expectedLoss', 'churned', 'tenure', 'balance', 'products']
    const csv = [columns.join(','), ...filteredRows.map((row) => columns.map((column) => row[column] ?? '').join(','))].join('\n')
    const link = document.createElement('a')
    link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }))
    link.download = `churn-${partition}-filtered.csv`
    link.click()
    URL.revokeObjectURL(link.href)
  }

  const numberTick = (value) => numberFormat.format(value)
  const chartTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div className="chart-tooltip">
        <strong>{label ?? payload[0].name}</strong>
        {payload.map((item) => (
          <span key={item.dataKey || item.name}>
            {item.name || item.dataKey}: {typeof item.value === 'number' && item.value < 1 ? formatPercent(item.value) : numberTick(item.value)}
          </span>
        ))}
      </div>
    )
  }

  if (isLoading) {
    return <div className="loading-screen"><span className="loading-mark"><ChartNoAxesCombined size={20} /></span><p>Reading scored customer cohorts…</p></div>
  }
  if (loadError) {
    return <div className="loading-screen"><div className="load-error"><ShieldAlert size={24} /><h1>Could not load churn results</h1><p>{loadError}</p><button onClick={() => location.reload()}>Retry</button></div></div>
  }

  return (
    <div className="app-shell">
      <aside className="side-rail">
        <div className="brand-mark" aria-label="Bank retention analytics"><span>B</span></div>
        <div className="rail-divider" />
        <button className="rail-button active" title="Portfolio overview" aria-label="Portfolio overview"><ChartNoAxesCombined size={19} /></button>
        <button className="rail-button" title="Customer prioritization" aria-label="Customer prioritization" onClick={() => document.getElementById('customer-list')?.scrollIntoView({ behavior: 'smooth' })}><UsersRound size={19} /></button>
        <div className="rail-bottom"><span className="online-dot" /><span>MODEL<br />READY</span></div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="breadcrumb"><span>PORTFOLIO INTELLIGENCE</span><ArrowRight size={13} /><strong>CHURN / RETENTION</strong></div>
          <div className="topbar-right"><span className="dataset-status"><i /> Saved model outputs</span><span className="topbar-divider" /><span className="model-version">XGBOOST · 300 TREES</span></div>
        </header>

        <section className="page-heading">
          <div>
            <p className="eyebrow">BANK CUSTOMER RETENTION / 2024 SYNTHETIC DATA</p>
            <h1>Retention signals<span>.</span></h1>
            <p className="intro-copy">A scored portfolio view for directing retention effort toward customers with the highest expected loss.</p>
          </div>
          <div className="heading-actions">
            <div className="partition-size"><span>ACTIVE COHORT</span><strong>{numberTick(filteredRows.length)} <small>records</small></strong></div>
            <button className="export-button" onClick={exportFilteredRows}><ArrowDownToLine size={15} /> Export view</button>
          </div>
        </section>

        <section className="partition-strip" aria-label="Select data partition">
          <div className="partition-tabs">
            {PARTITIONS.map((item) => (
              <button key={item.id} className={`partition-tab ${partition === item.id ? 'selected' : ''}`} onClick={() => { setPartition(item.id); resetFilters() }}>
                <span className="partition-label">{item.label}</span><span className="partition-note">{item.note}</span>
              </button>
            ))}
          </div>
          <p className="partition-source">{partition === 'inference' ? `${numberTick(activeRows.length)} ROWS · NEW-CUSTOMER SCORING EXPORT` : `${numberTick(activeRows.length)} ROWS · ${partition.toUpperCase()} PARTITION`}</p>
        </section>

        <section className="filter-bar" aria-label="Customer filters">
          <div className="filter-title"><Filter size={15} /><span>FILTER COHORT</span></div>
          <label className="filter-control"><span>Customer segment</span><select value={filters.segment} onChange={(event) => setFilter('segment', event.target.value)}><option>{FILTER_DEFAULTS.segment}</option>{filterOptions.segments.map((segment) => <option key={segment}>{segment}</option>)}</select><ChevronDown size={13} /></label>
          <label className="filter-control action-filter"><span>Retention action</span><select value={filters.action} onChange={(event) => setFilter('action', event.target.value)}><option>{FILTER_DEFAULTS.action}</option>{filterOptions.actions.map((action) => <option key={action} value={action}>{action.replaceAll('_', ' ')}</option>)}</select><ChevronDown size={13} /></label>
          <label className="filter-control compact-filter"><span>Products</span><select value={filters.products} onChange={(event) => setFilter('products', event.target.value)}><option>{FILTER_DEFAULTS.products}</option>{filterOptions.products.map((product) => <option key={product} value={product}>{product} products</option>)}</select><ChevronDown size={13} /></label>
          <label className="filter-control compact-filter"><span>Risk band</span><select value={filters.riskBand} onChange={(event) => setFilter('riskBand', event.target.value)}><option>{FILTER_DEFAULTS.riskBand}</option>{RISK_BANDS.map((band) => <option key={band.id}>{band.id}</option>)}</select><ChevronDown size={13} /></label>
          <div className="range-control"><div><span>Tenure</span><b>{filters.tenure[0]}–{filters.tenure[1]} yr</b></div><div className="dual-range"><input aria-label="Minimum tenure" type="range" min="1" max={Math.max(1, maxTenure)} value={filters.tenure[0]} onChange={(event) => setFilter('tenure', [Math.min(Number(event.target.value), filters.tenure[1]), filters.tenure[1]])} /><input aria-label="Maximum tenure" type="range" min="1" max={Math.max(1, maxTenure)} value={filters.tenure[1]} onChange={(event) => setFilter('tenure', [filters.tenure[0], Math.max(Number(event.target.value), filters.tenure[0])])} /></div></div>
          <label className="range-control balance-range"><div><span>Balance ≤</span><b>{currencyFormat.format(filters.balance[1])}</b></div><input aria-label="Maximum balance" type="range" min="0" max={Math.max(1, maxBalance)} value={Math.min(filters.balance[1], Math.max(1, maxBalance))} onChange={(event) => setFilter('balance', [filters.balance[0], Number(event.target.value)])} /></label>
          <button className="reset-button" onClick={resetFilters} title="Reset filters" aria-label="Reset filters"><RotateCcw size={15} /></button>
        </section>

        <section className="metric-grid" aria-label={`${currentPartition?.label} metrics`}>
          {partition === 'inference' ? (
            <>
              <MetricCard icon={<UsersRound size={16} />} label="CUSTOMERS SCORED" value={numberTick(filteredRows.length)} note="New-customer inference batch" />
              <MetricCard icon={<ShieldAlert size={16} />} label="HIGH-RISK CUSTOMERS" value={numberTick(filteredRows.filter((row) => row.churnScore >= 0.9).length)} note={formatPercent(filteredRows.length ? filteredRows.filter((row) => row.churnScore >= 0.9).length / filteredRows.length : 0) + ' of selected batch'} accent="coral" />
              <MetricCard icon={<Banknote size={16} />} label="EXPECTED LOST VALUE" value={currencyFormat.format(totalExpectedLoss)} note="Risk-weighted estimated customer value" accent="gold" />
              <MetricCard icon={<BadgeCheck size={16} />} label="OUTREACH ASSIGNED" value={numberTick(recommendedRows)} note={`${formatPercent(filteredRows.length ? recommendedRows / filteredRows.length : 0)} of selected batch`} accent="blue" />
            </>
          ) : (
            <>
              <MetricCard icon={<ChartNoAxesCombined size={16} />} label="ROC AUC" value={formatPercent(metrics?.auc, 2)} note={metrics?.auc == null ? 'Requires churners and non-churners' : partition === 'train' ? 'In-sample XGBoost score' : 'XGBoost scored output'} accent="teal" />
              <MetricCard icon={<Crosshair size={16} />} label="PR AUC" value={formatPercent(metrics?.prAuc, 2)} note={`Churn prevalence ${formatPercent(metrics?.prevalence)}`} accent="gold" />
              <MetricCard icon={<ShieldAlert size={16} />} label="RECALL @ TOP 20%" value={formatPercent(metrics?.recall20)} note="Share of churners in the highest-risk fifth" accent="coral" />
              <MetricCard icon={<Banknote size={16} />} label="EXPECTED LOST VALUE" value={currencyFormat.format(totalExpectedLoss)} note={`${currencyFormat.format(targetedValue)} attached to outreach actions`} accent="blue" />
            </>
          )}
        </section>

        <section className="chart-grid" aria-label="Interactive churn charts">
          <article className="chart-panel wide-panel">
            <div className="panel-heading"><div><p className="eyebrow">RISK DISTRIBUTION</p><h2>Score concentration</h2></div><span className="panel-context">{numberTick(riskBandBaseCount)} customers · click a band to filter</span></div>
            <div className="chart-area score-chart">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={scoreTrend} margin={{ top: 12, right: 12, left: -18, bottom: 2 }}>
                  <CartesianGrid stroke="#e8eeeb" vertical={false} />
                  <XAxis dataKey="range" tickLine={false} axisLine={false} tick={{ fill: '#798984', fontSize: 10 }} />
                  <YAxis yAxisId="count" tickLine={false} axisLine={false} tick={{ fill: '#798984', fontSize: 10 }} tickFormatter={numberTick} />
                  <YAxis yAxisId="rate" orientation="right" domain={[0, 1]} tickLine={false} axisLine={false} tick={{ fill: '#798984', fontSize: 10 }} tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                  <Tooltip content={chartTooltip} />
                  <Bar yAxisId="count" dataKey="count" name="Customers" radius={[3, 3, 0, 0]} cursor="pointer" onClick={(entry) => toggleFilter('riskBand', getRiskBand(entry.min + 0.001), FILTER_DEFAULTS.riskBand)}>
                    {scoreTrend.map((entry) => <Cell key={entry.range} fill={filters.riskBand !== FILTER_DEFAULTS.riskBand && getRiskBand(entry.min + 0.001) === filters.riskBand ? '#d4a12e' : entry.min >= 0.9 ? '#c84e37' : '#087f70'} />)}
                  </Bar>
                  {partition !== 'inference' && <Line yAxisId="rate" type="monotone" dataKey="actualRate" name="Observed churn rate" stroke="#d4a12e" strokeWidth={2.2} dot={{ r: 3, fill: '#d4a12e' }} connectNulls />}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-legend"><span><i className="legend-dot teal-dot" /> Predicted customer count</span>{partition !== 'inference' && <span><i className="legend-line gold-line" /> Observed churn rate</span>}<span className="chart-help"><CircleHelp size={13} /> Select a bar to filter risk band</span></div>
          </article>

          <article className="chart-panel">
            <div className="panel-heading"><div><p className="eyebrow">PORTFOLIO MIX</p><h2>Customer segment</h2></div><span className="panel-context">click a segment</span></div>
            <div className="chart-area donut-area">
              <div className="segment-stack" role="group" aria-label="Select customer segment">
                {segmentChartData.map((entry) => <button key={entry.name} className={filters.segment !== FILTER_DEFAULTS.segment && filters.segment !== entry.name ? 'muted-slice' : ''} style={{ width: `${(entry.count / Math.max(1, segmentBaseRows.length)) * 100}%`, backgroundColor: entry.fill }} title={`${entry.name}: ${numberTick(entry.count)} customers`} aria-label={`Filter by ${entry.name}, ${numberTick(entry.count)} customers`} onClick={() => toggleFilter('segment', entry.name, FILTER_DEFAULTS.segment)} />)}
              </div>
              <div className="donut-total"><strong>{numberTick(segmentBaseRows.length)}</strong><span>customers in mix</span></div>
            </div>
            <div className="segment-legend">{segmentChartData.map((item) => <button key={item.name} className={filters.segment !== FILTER_DEFAULTS.segment && filters.segment !== item.name ? 'muted-option' : ''} onClick={() => toggleFilter('segment', item.name, FILTER_DEFAULTS.segment)}><i style={{ background: item.fill }} /><span>{item.name.replaceAll('_', ' ')}</span><b>{numberTick(item.count)}</b></button>)}</div>
          </article>

          <article className="chart-panel">
            <div className="panel-heading"><div><p className="eyebrow">RETENTION PROGRAM</p><h2>Recommended action</h2></div><span className="panel-context">click an action to filter</span></div>
            <div className="chart-area action-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={actionChartData} layout="vertical" margin={{ top: 8, right: 14, left: 8, bottom: 0 }}>
                  <CartesianGrid stroke="#e8eeeb" horizontal={false} />
                  <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: '#798984', fontSize: 10 }} tickFormatter={numberTick} />
                  <YAxis type="category" dataKey="action" width={178} tickLine={false} axisLine={false} tick={{ fill: '#586a64', fontSize: 10 }} tickFormatter={(value) => value.replaceAll('_', ' ')} />
                  <Tooltip content={chartTooltip} />
                  <Bar dataKey="count" name="Customers" radius={[0, 3, 3, 0]} cursor="pointer" onClick={(entry) => toggleFilter('action', entry.action, FILTER_DEFAULTS.action)}>
                    {actionChartData.map((entry) => <Cell key={entry.action} fill={entry.fill} opacity={filters.action === FILTER_DEFAULTS.action || filters.action === entry.action ? 1 : 0.3} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </article>

          <article className="chart-panel benchmark-panel">
            <div className="panel-heading"><div><p className="eyebrow">NOTEBOOK BENCHMARKS</p><h2>Model comparison</h2></div><button className="text-button" onClick={() => setShowBenchmarks((value) => !value)}>{showBenchmarks ? 'Hide table' : 'View table'} <ArrowRight size={13} /></button></div>
            <p className="benchmark-note">Saved evaluation outputs · untouched validation and test cohorts · XGBoost is the selected model</p>
            <div className="benchmark-summary">
              <div><span>VALIDATION AUC</span><b>{formatPercent(BENCHMARKS.XGBoost.validation.auc, 2)}</b><small>+{((BENCHMARKS.XGBoost.validation.auc - BENCHMARKS.Logistic.validation.auc) * 100).toFixed(2)} pts vs logistic</small></div>
              <div><span>TEST AUC</span><b>{formatPercent(BENCHMARKS.XGBoost.test.auc, 2)}</b><small>held-out generalization</small></div>
              <div><span>TEST PR-AUC</span><b>{formatPercent(BENCHMARKS.XGBoost.test.prAuc, 2)}</b><small>imbalanced-class metric</small></div>
            </div>
            {showBenchmarks && <div className="benchmark-table-wrap"><table className="benchmark-table"><thead><tr><th>Model</th><th>Partition</th><th>ROC AUC</th><th>PR AUC</th><th>Recall @ 20%</th></tr></thead><tbody>{['Logistic', 'XGBoost'].flatMap((model) => ['validation', 'test'].map((set) => { const result = BENCHMARKS[model][set]; return <tr key={`${model}-${set}`}><td>{model}</td><td>{set}</td><td>{formatPercent(result.auc, 2)}</td><td>{formatPercent(result.prAuc, 2)}</td><td>{result.recall20 == null ? '—' : formatPercent(result.recall20)}</td></tr> }))}</tbody></table></div>}
          </article>
        </section>

        <section className="explain-grid" aria-label="Model explainability">
          <DriverImportancePanel importance={data.manifest.importance[partition]} labels={data.manifest.featureLabels} />
          <CalibrationPanel rows={filteredRows} />
          <GainsPanel rows={filteredRows} />
        </section>

        <section className="customer-section" id="customer-list">
          <div className="customer-section-head"><div><p className="eyebrow">PRIORITIZATION QUEUE</p><h2>{partition === 'inference' ? 'New customer recommendations' : `${currentPartition?.label} scored customers`}</h2><p>{numberTick(filteredRows.length)} matching records · sorted by expected lost value</p></div><label className="search-box"><span className="sr-only">Search customer cohort</span><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search segment or action" /></label></div>
          <div className="table-wrap">
            <table className="customer-table"><thead><tr><th>Record</th><th>Customer segment</th><th>Retention segment</th><th>Churn risk</th><th>Expected lost value</th><th>Tenure</th><th>Products</th><th>Recommended action</th></tr></thead><tbody>{rankedRows.map((row) => <tr key={`${partition}-${row.sourceIndex}`} className={selectedRow?.sourceIndex === row.sourceIndex ? 'selected-row' : ''} onClick={() => setSelectedRow(selectedRow?.sourceIndex === row.sourceIndex ? null : row)}><td><span className="record-id">#{numberTick(row.sourceIndex + 1)}</span></td><td>{row.customerSegment}</td><td><span className="segment-chip">{row.segment.replaceAll('_', ' ')}</span></td><td><span className={`risk-pill risk-${getRiskBand(row.churnScore).toLowerCase()}`}><i />{formatPercent(row.churnScore, 2)}</span></td><td className="money-cell">{currencyFormat.format(row.expectedLoss)}</td><td>{row.tenure} yr</td><td>{row.products}</td><td><span className="action-text">{row.action.replaceAll('_', ' ')}</span></td></tr>)}</tbody></table>
            {rankedRows.length === 0 && <div className="empty-table">No customers match these filters.</div>}
          </div>
          {selectedRow && <div className="selection-detail"><div><span className="eyebrow">SELECTED RECORD · #{numberTick(selectedRow.sourceIndex + 1)}</span><p>{selectedRow.segment.replaceAll('_', ' ')} / {getRiskBand(selectedRow.churnScore)} risk</p></div><strong>{currencyFormat.format(selectedRow.expectedLoss)} <small>expected lost value</small></strong><span className="selection-meta">Score {formatPercent(selectedRow.churnScore, 2)} · {selectedRow.tenure} year tenure · {selectedRow.products} products</span><div className="selection-drivers"><span className="eyebrow">TOP DRIVERS · SHAP LOG-ODDS</span><CustomerDrivers drivers={currentDrivers.drivers} error={currentDrivers.error} /></div></div>}
        </section>

        <footer className="page-footer"><span>Source: {data.manifest.datasets[datasetName].source} · {numberTick(activeRows.length)} scored rows in view</span><span>{partition === 'train' ? 'TRAINING METRICS ARE IN-SAMPLE AND MAY OVERSTATE PERFORMANCE' : partition === 'inference' ? 'INFERENCE BATCH HAS NO LABELED CHURN OUTCOME' : 'CHURN SCORE IS A MODEL ESTIMATE, NOT A GUARANTEE'}</span></footer>
      </main>
    </div>
  )
}

function MetricCard({ icon, label, value, note, accent = 'teal' }) {
  return (
    <article className={`metric-card metric-${accent}`}>
      <div className="metric-label"><span className="metric-icon">{icon}</span>{label}</div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-note">{note}</span>
    </article>
  )
}

export default App
