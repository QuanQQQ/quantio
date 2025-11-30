import { Card, Empty } from 'antd'
import ReactECharts from 'echarts-for-react'
import { useMemo } from 'react'
import { useAppStore } from '../store/appStore'

export default function KlineChart() {
  const kline = useAppStore(s => s.kline)
  const operations = useAppStore(s => s.operations)
  const trades = useAppStore(s => s.trades)
  const symbol = useAppStore(s => s.selectedSymbol)
  const dateRange = useAppStore(s => s.dateRange)

  const option = useMemo(() => {
    if (!kline.length) return null
    const dates = kline.map(k => k.date)
    const ohlc = kline.map(k => [k.open, k.close, k.low, k.high])
    const volume = kline.map(k => k.volume)

    // Prefer operations log (buy/sell) for marking; fallback to trades if operations missing
    let marks: any[] = []
    const startStr = dateRange[0].format('YYYYMMDD')
    const endStr = dateRange[1].format('YYYYMMDD')
    const inRange = (d: string) => d >= startStr && d <= endStr
    const ops = operations.filter(o => o.symbol === symbol && inRange(o.date))
    if (ops.length) {
      marks = ops.map(o => ({
        name: o.action === 'buy' ? '买' : (o.partial_ratio && o.partial_ratio < 1 ? '卖(半)' : '卖'),
        coord: [o.date, o.price],
        value: o.reason || (o.action === 'buy' ? '开仓' : '平仓'),
        itemStyle: { color: o.action === 'buy' ? '#52c41a' : '#f5222d' },
        symbol: o.action === 'buy' ? 'pin' : (o.partial_ratio && o.partial_ratio < 1 ? 'circle' : 'diamond'),
        symbolSize: 50,
      }))
    } else {
      marks = trades
        .filter(t => t.symbol === symbol)
        .flatMap(t => [
          { name: '买', coord: [t.entry_date, t.entry_price], value: '买入', itemStyle: { color: '#52c41a' }, symbol: 'pin', symbolSize: 50 },
          { name: t.position_closed === 'partial' ? '卖(半)' : '卖', coord: [t.exit_date, t.exit_price], value: t.close_reason, itemStyle: { color: '#f5222d' }, symbol: t.position_closed === 'partial' ? 'circle' : 'diamond', symbolSize: 40 }
        ])
    }

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
  }, [kline, trades, operations, symbol, dateRange])

  return (
    <Card title="K线与买卖点">
      {option ? <ReactECharts option={option} style={{ height: 400 }} /> : <Empty description="请选择股票代码" />}
    </Card>
  )
}
