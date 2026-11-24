import fs from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool'

const root = new URL('../../', import.meta.url)
const outDir = new URL('./', import.meta.url)
const outPath = fileURLToPath(outDir)
const resultsPath = new URL('./model_comparison_results.csv', outDir)
const recommendationsPath = new URL('./improvement_recommendations.csv', outDir)

function parseCsv(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/)
  const columns = header.split(',')
  return lines.map((line) => {
    const values = line.split(',')
    return Object.fromEntries(columns.map((key, index) => [key, values[index]]))
  })
}

const results = parseCsv(await fs.readFile(resultsPath, 'utf8'))
const recommendations = parseCsv(await fs.readFile(recommendationsPath, 'utf8'))
const byDataset = Object.fromEntries(results.map((row) => [row.dataset, row]))
const test = byDataset.Test
const validation = byDataset.Validation
const fresh = byDataset['New synthetic stress test']

const metricDefs = [
  ['rmse_probability', 'RMSE of churn probability', 'Lower is better'],
  ['brier_score', 'Brier score', 'Lower is better'],
  ['accuracy', 'Accuracy', 'Higher is better'],
  ['precision', 'Precision', 'Higher is better'],
  ['recall', 'Recall', 'Higher is better'],
  ['f1', 'F1 score', 'Higher is better'],
  ['roc_auc', 'ROC-AUC', 'Higher is better'],
  ['pr_auc', 'PR-AUC', 'Higher is better'],
  ['recall_at_5pct', 'Recall at top 5%', 'Higher is better'],
  ['recall_at_10pct', 'Recall at top 10%', 'Higher is better'],
  ['recall_at_20pct', 'Recall at top 20%', 'Higher is better'],
]

const wb = Workbook.create()
const summary = wb.worksheets.add('Summary')
const metrics = wb.worksheets.add('Detailed Metrics')
const method = wb.worksheets.add('Methodology')
const dark = '#173A32'
const green = '#087F70'
const light = '#EAF4EF'
const amber = '#FBF3E5'
const red = '#FBEAE6'
const gray = '#6E8079'

for (const sheet of [summary, metrics, method]) {
  sheet.showGridLines = false
}

summary.getRange('A1').values = [['Bank churn model robustness evaluation']]
summary.getRange('A2').values = [['Saved XGBoost model compared across original partitions and a 10,000-record labeled synthetic stress test.']]
summary.getRange('A4:F4').values = [['Dataset', 'Records', 'Churn rate', 'ROC-AUC', 'RMSE', 'Accuracy']]
summary.getRange('A5:F8').values = ['Training', 'Validation', 'Test', 'New synthetic stress test'].map((name) => {
  const row = byDataset[name]
  return [name, Number(row.records), Number(row.churn_rate), Number(row.roc_auc), Number(row.rmse_probability), Number(row.accuracy)]
})
summary.getRange('H4:J4').values = [['Metric', 'New vs test', 'Interpretation']]
summary.getRange('H5:J15').values = metricDefs.map(([key, label, direction]) => {
  const delta = Number(fresh[key]) - Number(test[key])
  const interpretation = key === 'rmse_probability' || key === 'brier_score'
    ? (delta > 0.03 ? 'Worse on new cohort' : 'Comparable')
    : (delta < -0.03 ? 'Worse on new cohort' : 'Comparable')
  return [label, delta, interpretation]
})
summary.getRange('A11').values = [['Key finding']]
summary.getRange('A12:F13').merge()
summary.getRange('A12').values = [[`Performance degraded materially on the stress test: ROC-AUC fell from ${Number(test.roc_auc).toFixed(3)} to ${Number(fresh.roc_auc).toFixed(3)}, while RMSE rose from ${Number(test.rmse_probability).toFixed(3)} to ${Number(fresh.rmse_probability).toFixed(3)}. The new cohort also has a lower churn rate (${(Number(fresh.churn_rate) * 100).toFixed(1)}% vs ${(Number(test.churn_rate) * 100).toFixed(1)}%), so accuracy alone is not a reliable success measure.`]]
summary.getRange('A15').values = [['Recommended improvements']]
summary.getRange(`A16:F${15 + recommendations.length}`).values = recommendations.map((row) => [row.improvement_recommendation, null, null, null, null, null])
for (let row = 16; row <= 15 + recommendations.length; row += 1) summary.getRange(`A${row}:F${row}`).merge()

metrics.getRange('A1').values = [['Detailed evaluation metrics']]
metrics.getRange('A2').values = [['RMSE is calculated on predicted churn probabilities and equals the square root of the Brier score. Classification metrics use a 0.50 probability threshold.']]
const metricHeaders = Object.keys(results[0])
metrics.getRange('A4').values = [metricHeaders]
metrics.getRange(`A5:${String.fromCharCode(64 + metricHeaders.length)}${4 + results.length}`).values = results.map((row) => metricHeaders.map((key) => key === 'dataset' ? row[key] : Number(row[key])))

method.getRange('A1').values = [['Test design and limitations']]
method.getRange('A3:B11').values = [
  ['Item', 'Description'],
  ['Dataset size', '10,000 labeled synthetic customer records'],
  ['Data realism', 'Skewed income, balances, loans, product counts, zero-inflated complaints, categorical variation, missing values, and heavy-tail outliers'],
  ['Drift', 'Different feature distributions and a changed churn-generating process introduce population and concept drift'],
  ['Class balance', 'Churn prevalence is approximately 6.4%, compared with approximately 12.2% in the original labeled data'],
  ['Model', 'Existing saved XGBoost model at data/models/xgb_bank_churn.json'],
  ['Preprocessing', 'Same dropped columns, tenure buckets, balance flag, complaint flag, occupation frequency encoding, one-hot encoding, and feature alignment'],
  ['Interpretation', 'Use ROC-AUC, PR-AUC, RMSE/Brier score, and top-k recall together. Accuracy is affected by class imbalance'],
  ['Next step', 'Validate on real newly labeled outcomes before production decisions, then recalibrate and retrain if drift persists'],
]

for (const sheet of [summary, metrics, method]) {
  sheet.getUsedRange().format.font = { name: 'Arial', size: 10, color: '#172B2A' }
}
summary.getRange('A1:J1').format = { font: { name: 'Arial', size: 16, bold: true, color: dark } }
summary.getRange('A2:J2').format = { font: { name: 'Arial', size: 10, italic: true, color: gray } }
for (const sheet of [summary, metrics, method]) {
  sheet.getRange('A1:J1').format.borders = { bottom: { style: 'thin', color: green } }
}
for (const range of [summary.getRange('A4:F4'), summary.getRange('H4:J4'), metrics.getRange('A4:Q4'), method.getRange('A3:B3')]) {
  range.format = { fill: dark, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' }, horizontalAlignment: 'center', verticalAlignment: 'center' }
}
summary.getRange('A5:F8').format.borders = { preset: 'inside', style: 'thin', color: '#DCE5E0' }
summary.getRange('H5:J15').format.borders = { preset: 'inside', style: 'thin', color: '#DCE5E0' }
summary.getRange('A11:F11').format = { fill: green, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' } }
summary.getRange('A15:F15').format = { fill: green, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' } }
summary.getRange('A12:F13').format = { fill: light, wrapText: true, verticalAlignment: 'center' }
summary.getRange(`A16:F${15 + recommendations.length}`).format = { fill: amber, wrapText: true }
summary.getRange('C5:C8').format.numberFormat = '0.0%'
summary.getRange('D5:F8').format.numberFormat = '0.000'
summary.getRange('I5:I15').format.numberFormat = '0.000'
metrics.getRange(`B5:B${4 + results.length}`).format.numberFormat = '#,##0'
metrics.getRange(`C5:C${4 + results.length}`).format.numberFormat = '0.0%'
metrics.getRange(`D5:Q${4 + results.length}`).format.numberFormat = '0.000'
metrics.getRange('A4:Q8').format.borders = { preset: 'inside', style: 'thin', color: '#DCE5E0' }
method.getRange('A3:B11').format.borders = { preset: 'inside', style: 'thin', color: '#DCE5E0' }
method.getRange('A4:A11').format.font = { name: 'Arial', size: 10, bold: true, color: dark }
method.getRange('B4:B11').format.wrapText = true

summary.getRange('A:A').format.columnWidth = 28
summary.getRange('B:F').format.columnWidth = 15
summary.getRange('H:H').format.columnWidth = 28
summary.getRange('I:I').format.columnWidth = 14
summary.getRange('J:J').format.columnWidth = 24
summary.getRange('A12:F13').format.rowHeight = 42
summary.getRange(`A16:F${15 + recommendations.length}`).format.rowHeight = 45
metrics.getRange('A:A').format.columnWidth = 28
metrics.getRange('B:Q').format.columnWidth = 14
method.getRange('A:A').format.columnWidth = 22
method.getRange('B:B').format.columnWidth = 92
summary.freezePanes.freezeRows(4)
metrics.freezePanes.freezeRows(4)
method.freezePanes.freezeRows(3)
summary.getRange('E8').conditionalFormats.add('cellIs', { operator: 'greaterThan', formula: 0.2, format: { fill: red, font: { color: '#9C2F20', bold: true } } })
summary.getRange('D8').conditionalFormats.add('cellIs', { operator: 'lessThan', formula: 0.8, format: { fill: red, font: { color: '#9C2F20', bold: true } } })

wb.recalculate()
const check = await wb.inspect({ kind: 'table', sheetId: 'Summary', range: 'A1:J15', include: 'values,formulas', tableMaxRows: 15, tableMaxCols: 10, maxChars: 6000 })
console.log(check.ndjson)
const preview = await wb.render({ sheetName: 'Summary', autoCrop: 'all', scale: 1, format: 'png' })
await fs.writeFile(fileURLToPath(new URL('./model_evaluation_summary.png', outDir)), new Uint8Array(await preview.arrayBuffer()))
const xlsx = await SpreadsheetFile.exportXlsx(wb)
await xlsx.save(`${outPath}model_evaluation_results.xlsx`)
console.log('Saved workbook and preview to', outPath)
