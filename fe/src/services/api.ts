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
