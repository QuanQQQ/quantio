import { Typography, Space, InputNumber, DatePicker, AutoComplete } from 'antd'
import { useAppStore } from '../store/appStore'
import dayjs from 'dayjs'

export default function ControlsBar() {
  const principal = useAppStore(s => s.principal)
  const setPrincipal = useAppStore(s => s.setPrincipal)
  const dateRange = useAppStore(s => s.dateRange)
  const setDateRange = useAppStore(s => s.setDateRange)
  const stocks = useAppStore(s => s.stocks)
  const setSymbol = useAppStore(s => s.setSymbol)

  const options = stocks.map(s => ({ value: s.symbol, label: `${s.symbol} - ${s.name}` }))

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
      <Typography.Title level={4} style={{ margin: 0 }}>回测数据可视化</Typography.Title>
      <Space>
        <span>本金</span>
        <InputNumber value={principal} min={1000} step={1000} onChange={(v) => setPrincipal(v || 0)} />
      </Space>
      <Space>
        <span>时间范围</span>
        <DatePicker.RangePicker value={dateRange} onChange={(r) => r && setDateRange([r[0]!, r[1]!])} />
      </Space>
      <AutoComplete style={{ width: 360 }} options={options} placeholder="搜索股票代码" onSelect={(v) => setSymbol(v)} allowClear onClear={() => setSymbol(undefined)} />
    </div>
  )
}
