# 目标
构建一个基于 Python 的交互式回测可视化页面，展示净值曲线、K 线买卖点、持仓明细与关键指标，支持本金、时间范围与股票代码交互。

## 技术方案
- 应用框架：Streamlit（已有依赖）
- 图表库：Plotly（折线、K 线、柱状）
- 数据源：
  - 回测交易记录：`backtest_trades_dynamic.csv`
  - 净值曲线：`backtest_equity_curve.csv`
  - 股票日线：SQLite `data/stock_data.db`，复用现有函数：
    - `get_all_stocks()`（src/database.py:130）用于搜索源
    - `get_stock_daily(symbol, start, end)`（src/database.py:172）用于 K 线

## 文件与结构
- 新增：`src/backtest_dashboard.py`（独立运行的 Streamlit 页面）
- 页面分区：
  1. 顶部：标题、本金输入、时间范围选择、股票搜索
  2. 中部左：K 线图 + 买卖点标注（来自交易 CSV）
  3. 中部右：净值曲线（来自净值 CSV）+ 指标卡片
  4. 底部：持仓明细表格（按日期筛选）

## 功能细节
- 数据加载与处理：
  - 读取 CSV 为 DataFrame，字段映射：交易（symbol, entry_date, exit_date, entry_price, exit_price, quantity, actual_return, predicted_return, hold_days, close_reason, position_closed），净值（date, equity, cash, positions）
  - 懒加载 K 线：选择股票后，调用 `get_stock_daily` 按选定时间范围查询 OHLCV
- 净值曲线与指标：
  - 折线图支持缩放/平移（Plotly 原生交互）
  - 指标：最大回撤（高水位回撤）、年化收益、累计收益、胜率、平均单笔收益
  - 本金联动：以首点净值为基准进行线性缩放并重算指标
- K 线与买卖点：
  - Plotly Candlestick + Volume 副图
  - 在 `entry_date`/`exit_date` 添加 Scatter 标记；`close_reason` 用不同颜色/形状区分（止损/止盈/到期）
- 持仓明细：
  - 表格展示：日期、股票、数量、买入价、卖出价、实际/预测收益率、持仓金额、盈亏金额
  - 计算方式：按每日日内等权分配本金到该日交易记录并计算盈亏（与现有 HTML 逻辑一致），支持按日期筛选

## 交互与性能
- 本金、时间范围、股票选择均触发相应图表和表格刷新
- 懒加载 K 线数据，避免全市场扫描
- 对 CSV/DB 读取做异常提示（缺文件、空数据）

## 验证
- 对比 `backtest_equity_curve.csv` 与净值图一致性
- 随机抽样交易核对 K 线买卖点位置
- 表格金额与盈亏在随机日期的数值合理性检查

## 实施步骤
1. 新建 `src/backtest_dashboard.py`，搭建 Streamlit 页面骨架与布局
2. 编写数据加载模块（CSV→DataFrame）与指标计算函数
3. 完成净值曲线与指标卡片
4. 完成 K 线图与买卖点标注，打通股票搜索与时间范围
5. 完成持仓明细表格与日期筛选
6. 端到端本地验证与边界情况处理

完成后，运行方式：`streamlit run src/backtest_dashboard.py`。