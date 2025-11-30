import { useEffect } from 'react'
import { Layout, Typography, InputNumber, DatePicker, AutoComplete, ConfigProvider } from 'antd'
import dayjs, { Dayjs } from 'dayjs'
import 'dayjs/locale/zh-cn'
import { useAppStore } from './store/appStore'
import ControlsBar from './components/ControlsBar'
import EquityChart from './components/EquityChart'
import KlineChart from './components/KlineChart'
import HoldingsTable from './components/HoldingsTable'
import IndicatorsPanel from './components/IndicatorsPanel'
import OperationsTable from './components/OperationsTable'

const { Header, Content, Footer } = Layout

export default function App() {
  const init = useAppStore(s => s.init)

  useEffect(() => {
    init()
  }, [init])

  return (
    <ConfigProvider theme={{ token: { colorPrimary: '#667eea' } }}>
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ background: '#fff', borderBottom: '1px solid #eee' }}>
          <ControlsBar />
        </Header>
        <Content style={{ padding: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <KlineChart />
            </div>
            <div>
              <IndicatorsPanel />
              <EquityChart />
            </div>
          </div>
        </Content>
        <Footer style={{ background: '#fff' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <HoldingsTable />
            <OperationsTable />
          </div>
        </Footer>
      </Layout>
    </ConfigProvider>
  )
}
