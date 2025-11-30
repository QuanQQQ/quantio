"""
测试股票过滤功能
"""
import sys
import os

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

from database import get_all_stocks

print("=" * 70)
print("测试股票过滤功能")
print("=" * 70)
print()

# 不过滤 - 获取所有股票
print("1. 获取所有股票（不过滤）：")
all_stocks = get_all_stocks(filter_tradable=False)
print(f"   总数: {len(all_stocks)} 只股票")
print()

# 过滤 - 只获取可交易的股票
print("2. 获取可交易股票（过滤创业板、科创板、北交所）：")
tradable_stocks = get_all_stocks(filter_tradable=True)
print()

# 显示被过滤掉的股票示例
print("3. 被过滤掉的股票示例：")
filtered_out = all_stocks[~all_stocks['symbol'].isin(tradable_stocks['symbol'])]
if len(filtered_out) > 0:
    print(f"   共 {len(filtered_out)} 只股票被过滤")
    print()
    print("   前10只被过滤的股票：")
    for _, stock in filtered_out.head(10).iterrows():
        code = stock['symbol'].split('.')[0]
        board = ""
        if code.startswith('300'):
            board = "创业板"
        elif code.startswith('688'):
            board = "科创板"
        elif code.startswith('8') or code.startswith('4'):
            board = "北交所"
        print(f"   - {stock['symbol']}: {stock['name']} ({board})")
else:
    print("   未找到被过滤的股票")

print()
print("=" * 70)
print("测试完成")
print("=" * 70)
