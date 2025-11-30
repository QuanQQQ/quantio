# 回测可视化前端实现计划

## 项目概览
- 在 `fe/` 目录实现一个专业的回测可视化页面，支持净值曲线、K线买卖点、持仓明细与搜索筛选。
- 前端单页应用，响应式布局，数据懒加载，满足缩放/平移交互与指标展示。

## 技术栈
- 前端框架：React + TypeScript + Vite
- 图表库：ECharts（折线/柱状/K线，含 dataZoom）
- UI 组件：Ant Design（布局、输入框、表格、日期选择器、AutoComplete）
- 状态管理：Zustand（轻量全局状态）
- CSV 解析：Papaparse（或 d3-dsv）
- 后端接口：FastAPI（复用现有数据库函数，按需提供数据）

## 目录结构
```
fe/
  ├─ index.html
  ├─ vite.config.ts
  ├─ package.json
  ├─ src/
  │   ├─ main.tsx
  │   ├─ App.tsx
  │   ├─ components/
  │   │   ├─ EquityChart.tsx
  │   │   ├─ KlineChart.tsx
  │   │   ├─ IndicatorsPanel.tsx
  │   │   ├─ HoldingsTable.tsx
  │   │   └─ ControlsBar.tsx
  │   ├─ services/
  │   │   ├─ api.ts              // 封装 fetch
  │   │   ├─ equity.ts           // 读取/计算净值
  │   │   ├─ trades.ts           // 解析交易记录
  │   │   └─ kline.ts            // 拉取日线
  │   ├─ store/
  │   │   └─ appStore.ts         // 全局状态
  │   ├─ utils/
  │   │   ├─ metrics.ts          // 最大回撤、年化收益等
  │   │   └─ date.ts             // 日期工具
  │   └─ styles/
  │       └─ app.css
```
后端新增：
```
src/api.py  // FastAPI 服务，基于现有数据库模块
```

## 数据源与访问
- 股票日线：从 SQLite `data/stock_data.db` 读取，复用 `src/database.py` 中方法
  - `get_all_stocks()` 用于股票搜索源（代码参考 src/database.py:130）
  - `get_stock_daily(symbol, start, end)` 提供 K 线数据（代码参考 src/database.py:172）
- 交易记录：读取 `/Users/bytedance/Documents/project/quantio/backtest_trades_dynamic.csv`
- 净值曲线：读取 `/Users/bytedance/Documents/project/quantio/backtest_equity_curve.csv`

接口设计（FastAPI）：
- `GET /api/stocks` → `{ symbol, name, sector, listing_date }[]`
- `GET /api/kline?symbol&start&end` → OHLCV 列表
- `GET /api/trades` → trades_dynamic.csv 解析后的结构
- `GET /api/equity` → equity_curve.csv 解析后的结构

说明：纯浏览器无法直接访问本机路径的 SQLite/CSV，因此以轻量 API 暴露数据；若仅本地静态预览，可将 `.db/.csv` 作为静态文件并用 `sql.js` 在前端解析（备选实现）。

## 关键模块
- 净值曲线展示区（EquityChart）
  - 折线图展示净值；开启 `dataZoom`（缩放/平移）与 tooltip
  - 指标显示（IndicatorsPanel）：最大回撤、年化收益、累计收益、胜率、平均单笔收益
  - 本金变更联动：按首个净值点缩放整条曲线并重算指标
- K 线与买卖点（KlineChart）
  - 专业 K 线（ECharts candlestick）+ 成交量副图
  - 在 `entry_date`/`exit_date` 叠加买卖标记（颜色区分 `close_reason`：止损/止盈/到期）
  - 支持切换股票代码与时间范围
- 持仓明细表格（HoldingsTable）
  - 按日期筛选显示当日持仓；字段：日期、股票、数量、持仓市值、盈亏等
  - 计算方法：用 trades 动态记录 + 对应日线收盘价重构持仓；懒加载按需查询所需股票的日线
- 控制区（ControlsBar）
  - 本金输入框、时间范围选择器、股票搜索（AutoComplete）
  - 全局状态驱动图表与表格刷新

## 指标计算
- 最大回撤（MDD）：遍历净值序列，维护高水位 `peak`，取最小 `(equity/peak - 1)`
- 年化收益：`annual = (equity_end / equity_start)^(365 / days) - 1`
- 累计收益：`equity_end / equity_start - 1`
- 胜率/平均单笔收益：基于 trades 动态记录聚合
- 本金变更：以 `scale = principal / first_equity` 缩放净值与日度 PnL，联动重算指标

## 交互与性能
- 懒加载：仅在选择股票时请求其日线；持仓计算按需查询涉及股票范围
- 虚拟表格：`antd` Table 配合分页或 `react-window`
- 图表优化：ECharts 大数据模式、禁用多余阴影；异步数据请求并显示骨架屏
- 状态管理：Zustand 存储本金、选股、时间范围、已缓存数据

## 页面布局
- 顶部：标题 + 本金输入区 + 时间范围 + 股票搜索
- 中部左：K 线与买卖点
- 中部右：净值曲线 + 指标卡片
- 底部：持仓明细表格（日期筛选）

## 开发步骤
1. 初始化 `fe`（Vite React TS、ECharts、AntD、Zustand、Papaparse）
2. 搭建 FastAPI（`src/api.py`）并复用 `database.py` 提供数据接口
3. 封装前端 `services`，实现 CSV/接口数据拉取与解析
4. 完成 `EquityChart` 与指标计算，联动本金变更
5. 完成 `KlineChart`，叠加买卖点标记与颜色区分
6. 完成 `HoldingsTable`，实现日期筛选与懒加载计算
7. 打通控制区交互（本金/时间范围/搜索）与全局状态
8. 响应式布局与性能优化（懒加载/缓存/虚拟列表）
9. 本地验证与数据一致性检查（与 CSV/DB 对比）

## 验证与交付
- 对比 `backtest_equity_curve.csv` 与前端净值曲线一致性
- 随机抽样股票核对 K 线与买卖点是否正确
- 持仓表在多个日期的数值与 DB 日线价格匹配
- 提供运行说明：启动 FastAPI，再启动 Vite 开发服务器，浏览器访问 `http://localhost:5173/`。

如确认该计划，我将按上述步骤实施并交付可运行的前端页面。