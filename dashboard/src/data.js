const DATA_URL = `${import.meta.env.BASE_URL}data/`
const UNLABELED = 255

const ARRAY_TYPES = {
  float32: Float32Array,
  uint8: Uint8Array,
}

async function fetchData(file, parse) {
  const response = await fetch(`${DATA_URL}${file}`)
  if (!response.ok) throw new Error(`Could not load ${file} (${response.status})`)
  return parse(response)
}

function readColumns(buffer, layout) {
  return Object.fromEntries(layout.map(({ name, type, offset, length }) => [name, new ARRAY_TYPES[type](buffer, offset, length)]))
}

function toRows(columns, dictionaries, count) {
  const rows = new Array(count)
  for (let index = 0; index < count; index += 1) {
    rows[index] = {
      sourceIndex: index,
      churnScore: columns.churnScore[index],
      clv: columns.clv[index],
      expectedLoss: columns.expectedLoss[index],
      balance: columns.balance[index],
      tenure: columns.tenure[index],
      products: columns.products[index],
      churned: columns.churned[index] === UNLABELED ? null : columns.churned[index] === 1,
      segment: dictionaries.segment[columns.segment[index]],
      action: dictionaries.action[columns.action[index]],
      customerSegment: dictionaries.customerSegment[columns.customerSegment[index]],
    }
  }
  return rows
}

export async function loadDashboardData() {
  const manifest = await fetchData('manifest.json', (response) => response.json())
  const [labeled, inference] = await Promise.all(['labeled', 'inference'].map(async (name) => {
    const dataset = manifest.datasets[name]
    const buffer = await fetchData(`${name}.bin`, (response) => response.arrayBuffer())
    return toRows(readColumns(buffer, dataset.columns), manifest.dictionaries, dataset.rows)
  }))
  return { manifest, labeled, inference }
}

const driverRequests = new Map()

export function loadDrivers(manifest, name) {
  if (!driverRequests.has(name)) {
    const request = fetchData(`${name}-drivers.bin`, (response) => response.arrayBuffer())
      .then((buffer) => readColumns(buffer, manifest.datasets[name].drivers))
    request.catch(() => driverRequests.delete(name))
    driverRequests.set(name, request)
  }
  return driverRequests.get(name)
}

export function driversForRow(drivers, manifest, name, rowIndex) {
  const perRow = drivers.feature.length / manifest.datasets[name].rows
  return Array.from({ length: perRow }, (_, slot) => {
    const position = rowIndex * perRow + slot
    const feature = drivers.feature[position]
    return {
      name: manifest.featureNames[feature],
      label: manifest.featureLabels[feature],
      binary: manifest.binaryFeatures[feature],
      contribution: drivers.contribution[position],
      value: drivers.value[position],
    }
  })
}
