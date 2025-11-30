import { Card, Statistic, Row, Col } from 'antd'
import { useMemo } from 'react'
import { useAppStore } from '../store/appStore'
import { scaleEquity, maxDrawdown, annualReturn, summaryFromTrades } from '../utils/metrics'

export default function IndicatorsPanel() {
  const equity = useAppStore(s => s.equity)
  const principal = useAppStore(s => s.principal)
  const trades = useAppStore(s => s.trades)

  const scaled = useMemo(() => scaleEquity(equity, principal), [equity, principal])
  const mdd = useMemo(() => maxDrawdown(scaled), [scaled])
  const annual = useMemo(() => annualReturn(scaled), [scaled])
  const summary = useMemo(() => summaryFromTrades(trades), [trades])

  return (
    <Card style={{ marginBottom: 16 }}>
      <Row gutter={16}>
        <Col span={4}><Statistic title="总交易" value={summary.total} /></Col>
        <Col span={4}><Statistic title="胜率" value={summary.winRate * 100} precision={2} suffix="%" /></Col>
        <Col span={4}><Statistic title="平均单笔收益" value={summary.avgReturn} precision={2} suffix="%" /></Col>
        <Col span={6}><Statistic title="最大回撤" value={mdd * 100} precision={2} suffix="%" /></Col>
        <Col span={6}><Statistic title="年化收益" value={annual * 100} precision={2} suffix="%" /></Col>
      </Row>
    </Card>
  )
}
