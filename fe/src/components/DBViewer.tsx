import { Card, Statistic, Row, Col, AutoComplete, DatePicker, Table } from 'antd'
import { useAppStore } from '../store/appStore'
import { useEffect, useMemo, useState } from 'react'
import { fetchDBStats, fetchDaily } from '../services/api'
import dayjs from 'dayjs'

export default function DBViewer() {
  const stocks = useAppStore(s => s.stocks)
  const [stats, setStats] = useState<any>({})
  const [symbol, setSymbol] = useState<string | undefined>(undefined)
  const [range, setRange] = useState<[any, any] | undefined>(undefined)
  const [rows, setRows] = useState<any[]>([])

  const options = useMemo(() => stocks.map(s => ({ value: s.symbol, label: `${s.symbol} - ${s.name}` })), [stocks])

  useEffect(() => {
    fetchDBStats().then(setStats).catch(() => setStats({}))
  }, [])

  useEffect(() => {
    if (!symbol) { setRows([]); return }
    const start = range?.[0]?.format('YYYYMMDD')
    const end = range?.[1]?.format('YYYYMMDD')
    fetchDaily(symbol, start, end).then(setRows).catch(() => setRows([]))
  }, [symbol, range])

  return (
    <div>
      <Card title="数据库统计" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col span={6}><Statistic title="日线行数" value={stats?.rows || 0} /></Col>
          <Col span={6}><Statistic title="股票数（stocks）" value={stats?.stocks_count || 0} /></Col>
          <Col span={6}><Statistic title="日线股票数" value={stats?.symbols_in_prices || 0} /></Col>
          <Col span={6}><Statistic title="日期范围" value={`${stats?.min_date || ''} ~ ${stats?.max_date || ''}`} /></Col>
        </Row>
      </Card>

      <Card title="日线数据浏览">
        <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
          <AutoComplete style={{ width: 360 }} options={options} placeholder="选择股票代码" onSelect={(v) => setSymbol(v)} allowClear onClear={() => setSymbol(undefined)} />
          <DatePicker.RangePicker onChange={(r) => setRange(r as any)} />
        </div>
        <Table size="small" dataSource={rows.map((r, i) => ({ key: `${r.date}-${i}`, ...r }))} pagination={{ pageSize: 20 }} columns={[
          { title: '日期', dataIndex: 'date' },
          { title: '开', dataIndex: 'open' },
          { title: '高', dataIndex: 'high' },
          { title: '低', dataIndex: 'low' },
          { title: '收', dataIndex: 'close' },
          { title: '成交量', dataIndex: 'volume' },
          { title: '成交额', dataIndex: 'amount' },
        ]} />
      </Card>
    </div>
  )
}
