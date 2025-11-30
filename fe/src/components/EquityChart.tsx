import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Card } from 'antd'
import { useAppStore } from '../store/appStore'
import { scaleEquity } from '../utils/metrics'

export default function EquityChart() {
  const equity = useAppStore(s => s.equity)
  const principal = useAppStore(s => s.principal)
  const scaled = useMemo(() => scaleEquity(equity, principal), [equity, principal])

  const option = useMemo(() => ({
    tooltip: { trigger: 'axis', valueFormatter: (v: any) => `¥${Number(v).toLocaleString()}` },
    grid: { left: 40, right: 20, top: 20, bottom: 40 },
    xAxis: { type: 'category', data: scaled.map(e => e.date) },
    yAxis: { type: 'value' },
    dataZoom: [{ type: 'inside' }, { type: 'slider' }],
    series: [{
      type: 'line',
      name: '净值',
      smooth: true,
      showSymbol: false,
      areaStyle: { opacity: 0.1 },
      lineStyle: { width: 2 },
      data: scaled.map(e => e.equity)
    }]
  }), [scaled])

  return (
    <Card title="净值曲线">
      <ReactECharts option={option} style={{ height: 360 }} />
    </Card>
  )
}
