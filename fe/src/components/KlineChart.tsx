import { Card, Empty } from 'antd'
import ReactECharts from 'echarts-for-react'
import { useMemo } from 'react'
import { useAppStore } from '../store/appStore'

export default function KlineChart() {
  const kline = useAppStore(s => s.kline)
  const trades = useAppStore(s => s.trades)
  const symbol = useAppStore(s => s.selectedSymbol)

  const option = useMemo(() => {
    if (!kline.length) return null
    const dates = kline.map(k => k.date)
    const ohlc = kline.map(k => [k.open, k.close, k.low, k.high])
    const volume = kline.map(k => k.volume)

    const marks = trades
      .filter(t => t.symbol === symbol)
      .flatMap(t => [
        { name: '买', coord: [t.entry_date, t.entry_price], value: '买入', itemStyle: { color: '#52c41a' } },
        { name: '卖', coord: [t.exit_date, t.exit_price], value: t.close_reason, itemStyle: { color: '#f5222d' } }
      ])

    return {
      tooltip: { trigger: 'axis' },
      axisPointer: { type: 'cross' },
      grid: [{ left: 40, right: 20, top: 20, height: 260 }, { left: 40, right: 20, top: 300, height: 80 }],
      xAxis: [{ type: 'category', data: dates }, { type: 'category', gridIndex: 1, data: dates }],
      yAxis: [{ scale: true }, { gridIndex: 1 }],
      dataZoom: [{ type: 'inside', xAxisIndex: [0, 1] }, { type: 'slider', xAxisIndex: [0, 1] }],
      series: [
        { type: 'candlestick', name: 'K线', data: ohlc, markPoint: { data: marks } },
        { type: 'bar', name: '成交量', xAxisIndex: 1, yAxisIndex: 1, data: volume, itemStyle: { color: '#8884d8' } }
      ]
    }
  }, [kline, trades, symbol])

  return (
    <Card title="K线与买卖点">
      {option ? <ReactECharts option={option} style={{ height: 400 }} /> : <Empty description="请选择股票代码" />}
    </Card>
  )
}
