import { Card, Table, DatePicker } from 'antd'
import { useMemo, useState } from 'react'
import { useAppStore } from '../store/appStore'

export default function HoldingsTable() {
  const trades = useAppStore(s => s.trades)
  const principal = useAppStore(s => s.principal)
  const [date, setDate] = useState<string | undefined>(undefined)

  const dailyMap = useMemo(() => {
    const map: Record<string, typeof trades> = {}
    for (const t of trades) {
      const d = String(t.entry_date)
      if (!map[d]) map[d] = []
      map[d].push(t)
    }
    return map
  }, [trades])

  const capitalByDate = useMemo(() => {
    const dates = Object.keys(dailyMap).sort()
    let capital = principal
    const out: Record<string, number> = {}
    for (const d of dates) {
      out[d] = capital
      const td = dailyMap[d]
      const per = capital / td.length
      let pnl = 0
      for (const t of td) pnl += per * (t.actual_return / 100)
      capital += pnl
    }
    return out
  }, [dailyMap, principal])

  const rows = useMemo(() => {
    const list = (date ? (dailyMap[date] || []) : trades).map(t => {
      const cap = capitalByDate[String(t.entry_date)] || principal
      const per = cap / (dailyMap[String(t.entry_date)]?.length || 1)
      const pnlAmt = per * (t.actual_return / 100)
      return {
        key: `${t.symbol}-${t.entry_date}`,
        date: t.entry_date,
        symbol: t.symbol,
        entry_price: t.entry_price,
        exit_price: t.exit_price,
        actual_return: t.actual_return,
        predicted_return: t.predicted_return,
        position_value: per,
        pnl_amount: pnlAmt
      }
    })
    return list
  }, [trades, capitalByDate, date, dailyMap, principal])

  return (
    <Card title="持仓明细">
      <div style={{ marginBottom: 8 }}>
        <DatePicker onChange={(d) => setDate(d ? d.format('YYYYMMDD') : undefined)} placeholder="按日期筛选" />
      </div>
      <Table size="small" dataSource={rows} pagination={{ pageSize: 10 }} columns={[
        { title: '日期', dataIndex: 'date' },
        { title: '股票代码', dataIndex: 'symbol' },
        { title: '买入价', dataIndex: 'entry_price', render: (v) => `¥${v.toFixed(2)}` },
        { title: '卖出价', dataIndex: 'exit_price', render: (v) => `¥${v?.toFixed(2)}` },
        { title: '实际收益率', dataIndex: 'actual_return', render: (v) => `${v.toFixed(2)}%` },
        { title: '预测收益率', dataIndex: 'predicted_return', render: (v) => `${v.toFixed(2)}%` },
        { title: '持仓金额', dataIndex: 'position_value', render: (v) => `¥${Math.round(v).toLocaleString()}` },
        { title: '盈亏金额', dataIndex: 'pnl_amount', render: (v) => `¥${Math.round(v).toLocaleString()}` }
      ]} />
    </Card>
  )
}
