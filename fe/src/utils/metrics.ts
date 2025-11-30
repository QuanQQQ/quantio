import type { EquityPoint, Trade } from '../store/appStore'

export function scaleEquity(equity: EquityPoint[], principal: number) {
  if (!equity.length) return []
  const first = equity[0].equity
  const scale = principal / first
  return equity.map(e => ({ ...e, equity: e.equity * scale }))
}

export function maxDrawdown(equity: EquityPoint[]) {
  let peak = -Infinity
  let mdd = 0
  for (const e of equity) {
    peak = Math.max(peak, e.equity)
    const dd = (e.equity / peak) - 1
    if (dd < mdd) mdd = dd
  }
  return mdd
}

export function annualReturn(equity: EquityPoint[]) {
  if (!equity.length) return 0
  const start = equity[0].equity
  const end = equity[equity.length - 1].equity
  const days = equity.length
  return Math.pow(end / start, 365 / days) - 1
}

export function summaryFromTrades(trades: Trade[]) {
  const total = trades.length
  const win = trades.filter(t => t.actual_return > 0).length
  const avg = total ? trades.reduce((s, t) => s + t.actual_return, 0) / total : 0
  return { total, winRate: total ? win / total : 0, avgReturn: avg }
}
