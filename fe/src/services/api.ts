export async function fetchEquity() {
  const res = await fetch('/api/equity')
  const data = await res.json()
  return (data as any[])
    .filter((d) => d.date && d.equity)
    .map((d) => ({ date: String(d.date), equity: Number(d.equity), cash: Number(d.cash || 0), positions: Number(d.positions || 0) }))
}

export async function fetchTrades() {
  const res = await fetch('/api/trades')
  const data = await res.json()
  return (data as any[])
    .filter((d) => d.symbol && d.entry_date)
    .map((d) => ({
      symbol: String(d.symbol),
      entry_date: String(d.entry_date),
      exit_date: String(d.exit_date),
      entry_price: Number(d.entry_price),
      exit_price: Number(d.exit_price),
      quantity: Number(d.quantity),
      actual_return: Number(d.actual_return),
      predicted_return: Number(d.predicted_return),
      hold_days: Number(d.hold_days),
      close_reason: String(d.close_reason),
      position_closed: String(d.position_closed),
    }))
}

export async function fetchStocks() {
  const res = await fetch('/api/stocks')
  const data = await res.json()
  return data
}

export async function fetchKline(symbol: string, start?: string, end?: string) {
  const url = new URL('/api/kline', window.location.origin)
  url.searchParams.set('symbol', symbol)
  if (start) url.searchParams.set('start', start)
  if (end) url.searchParams.set('end', end)
  const res = await fetch(url.toString())
  const data = await res.json()
  return data as { date: string; open: number; high: number; low: number; close: number; volume: number }[]
}

export async function fetchOperations() {
  const res = await fetch('/api/operations')
  const data = await res.json()
  return (data as any[]).map((d) => ({
    date: String(d.date),
    action: String(d.action),
    symbol: String(d.symbol),
    price: Number(d.price),
    quantity: Number(d.quantity),
    reason: d.reason ? String(d.reason) : '',
    partial_ratio: d.partial_ratio !== undefined ? Number(d.partial_ratio) : undefined,
    hold_days: d.hold_days !== undefined ? Number(d.hold_days) : undefined,
    predicted_return: d.predicted_return !== undefined ? Number(d.predicted_return) : undefined,
  }))
}

export async function fetchDBStats() {
  const res = await fetch('/api/db/stats')
  return res.json()
}

export async function fetchDaily(symbol: string, start?: string, end?: string) {
  const url = new URL('/api/daily', window.location.origin)
  url.searchParams.set('symbol', symbol)
  if (start) url.searchParams.set('start', start)
  if (end) url.searchParams.set('end', end)
  const res = await fetch(url.toString())
  const data = await res.json()
  return data as { date: string; open: number; high: number; low: number; close: number; volume: number; amount: number }[]
}
