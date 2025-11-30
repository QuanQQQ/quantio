import { create } from 'zustand'
import dayjs, { Dayjs } from 'dayjs'
import { fetchEquity, fetchTrades, fetchStocks, fetchKline, fetchOperations } from '../services/api'

export type EquityPoint = { date: string; equity: number; cash?: number; positions?: number }
export type Trade = {
  symbol: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  quantity: number
  actual_return: number
  predicted_return: number
  hold_days: number
  close_reason: string
  position_closed: string
}
export type KlineBar = { date: string; open: number; high: number; low: number; close: number; volume: number }
export type StockItem = { symbol: string; name: string; sector?: string; listing_date?: string }
export type Operation = {
  date: string
  action: 'buy' | 'sell'
  symbol: string
  price: number
  quantity: number
  reason?: string
  partial_ratio?: number
  hold_days?: number
  predicted_return?: number
}

type State = {
  principal: number
  dateRange: [Dayjs, Dayjs]
  selectedSymbol?: string
  stocks: StockItem[]
  equity: EquityPoint[]
  trades: Trade[]
  kline: KlineBar[]
  operations: Operation[]
  loading: boolean
  init: () => Promise<void>
  setPrincipal: (v: number) => void
  setDateRange: (r: [Dayjs, Dayjs]) => void
  setSymbol: (s?: string) => Promise<void>
}

export const useAppStore = create<State>((set, get) => ({
  principal: 100000,
  dateRange: [dayjs('2023-08-21'), dayjs('2023-12-22')],
  stocks: [],
  equity: [],
  trades: [],
  kline: [],
  operations: [],
  loading: false,
  init: async () => {
    set({ loading: true })
    const [equity, trades, stocks, opsRaw] = await Promise.all([
      fetchEquity(),
      fetchTrades(),
      fetchStocks(),
      fetchOperations(),
    ])
    const operations: Operation[] = (opsRaw || []).map((o: any) => ({
      date: String(o.date),
      action: String(o.action) === 'buy' ? 'buy' : 'sell',
      symbol: String(o.symbol),
      price: Number(o.price),
      quantity: Number(o.quantity),
      reason: o.reason ? String(o.reason) : undefined,
      partial_ratio: o.partial_ratio !== undefined ? Number(o.partial_ratio) : undefined,
      hold_days: o.hold_days !== undefined ? Number(o.hold_days) : undefined,
      predicted_return: o.predicted_return !== undefined ? Number(o.predicted_return) : undefined,
    }))
    set({ equity, trades, stocks, operations, loading: false })
  },
  setPrincipal: (v) => set({ principal: v }),
  setDateRange: async (r) => {
    set({ dateRange: r })
    const symbol = get().selectedSymbol
    if (symbol) {
      const start = r[0].format('YYYYMMDD')
      const end = r[1].format('YYYYMMDD')
      const kline = await fetchKline(symbol, start, end)
      set({ kline })
    }
  },
  setSymbol: async (s) => {
    set({ selectedSymbol: s })
    if (s) {
      const [start, end] = get().dateRange
      const kline = await fetchKline(s, start.format('YYYYMMDD'), end.format('YYYYMMDD'))
      set({ kline })
    } else {
      set({ kline: [] })
    }
  }
}))
