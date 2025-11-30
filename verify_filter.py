"""
验证数据生成时的股票过滤功能
"""
import sys
import os

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

from database import get_all_stocks

print("=" * 70)
print("验证数据生成时的股票过滤")
print("=" * 70)
print()

# 模拟 generate_training_data 中的调用
all_stocks = get_all_stocks()  # 默认 filter_tradable=True
symbols = all_stocks['symbol'].tolist()

print(f"✓ 将使用 {len(symbols)} 只可交易股票生成训练数据")
print()

# 检查是否还有不可交易的股票
chiNext = [s for s in symbols if s.split('.')[0].startswith('300')]
star = [s for s in symbols if s.split('.')[0].startswith('688')]
bse = [s for s in symbols if len(s.split('.')[0]) == 6 and s.split('.')[0][0] in ['8', '4']]

print(f"✓ 创业板股票: {len(chiNext)} 只")
print(f"✓ 科创板股票: {len(star)} 只")
print(f"✓ 北交所股票: {len(bse)} 只")
print()

if chiNext or star or bse:
    print("⚠ 警告：仍有不可交易股票！")
else:
    print("✓ 确认：所有不可交易股票已被过滤！")

print()
print("=" * 70)
print("验证完成")
print("=" * 70)
