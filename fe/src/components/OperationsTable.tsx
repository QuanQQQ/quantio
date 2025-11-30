import { Card, Table } from 'antd'
import { useMemo } from 'react'
import { useAppStore } from '../store/appStore'

export default function OperationsTable() {
  const ops = useAppStore(s => s.operations)
  const selectedSymbol = useAppStore(s => s.selectedSymbol)
  const dateRange = useAppStore(s => s.dateRange)

  const rows = useMemo(() => {
    const start = dateRange[0].format('YYYYMMDD')
    const end = dateRange[1].format('YYYYMMDD')
    return ops
      .filter(o => (!selectedSymbol || o.symbol === selectedSymbol) && o.date >= start && o.date <= end)
      .map((o, idx) => ({
        key: `${o.date}-${o.symbol}-${o.action}-${idx}`,
        ...o,
      }))
      .sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0))
  }, [ops, selectedSymbol, dateRange])

  return (
    <Card title="每日买卖操作" style={{ marginTop: 16 }}>
      <Table size="small" dataSource={rows} pagination={{ pageSize: 10 }} columns={[
        { title: '日期', dataIndex: 'date' },
        { title: '动作', dataIndex: 'action', render: (v) => (v === 'buy' ? '买入' : '卖出') },
        { title: '股票代码', dataIndex: 'symbol' },
        { title: '价格', dataIndex: 'price', render: (v) => `¥${Number(v).toFixed(2)}` },
        { title: '数量', dataIndex: 'quantity', render: (v) => Number(v).toFixed(2) },
        { title: '理由', dataIndex: 'reason' },
        { title: '比例', dataIndex: 'partial_ratio', render: (v) => (v !== undefined ? `${Math.round(Number(v) * 100)}%` : '') },
        { title: '持仓天数', dataIndex: 'hold_days' },
      ]} />
    </Card>
  )
}
