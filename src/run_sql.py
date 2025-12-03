"""
通用 SQLite 运行器：执行任意 SQL（查询/非查询），支持：
- 指定数据库路径（默认 data/stock_data.db）
- 从命令行传入 SQL、从 .sql 文件读取或从 stdin 读取
- 绑定参数（CSV 或 JSON），兼容 '?' 占位符
- 查询结果输出到控制台、或保存为 CSV/JSON
- 控制台展示行数与预览

用法示例：
  # 运行查询并展示结果
  python src/run_sql.py --sql "SELECT COUNT(*) FROM daily_prices"

  # 带参数查询（CSV 参数，对 '?' 占位符）
  python src/run_sql.py --sql "SELECT * FROM daily_prices WHERE symbol=? AND date>=?" \
                        --params 000001.SZ,20240101

  # 从文件执行 DDL/DML（多语句）
  python src/run_sql.py --file scripts/fix_index.sql

  # 查询结果保存为 CSV
  python src/run_sql.py --sql "SELECT * FROM daily_prices LIMIT 1000" --output out.csv

  # 查询结果保存为 JSON
  python src/run_sql.py --sql "SELECT symbol,date,close FROM daily_prices LIMIT 10" --format json
"""

import argparse
import os
import sys
import sqlite3
import json
from typing import List, Optional

import pandas as pd


DEFAULT_DB = os.path.join("data", "stock_data.db")


def _strip_comments(sql: str) -> str:
    lines = []
    for line in sql.splitlines():
        s = line.strip()
        if s.startswith("--"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _is_select_like(sql: str) -> bool:
    s = _strip_comments(sql).lstrip()
    if not s:
        return False
    first = s.split(None, 1)[0].upper()
    return first in {"SELECT", "WITH", "PRAGMA", "EXPLAIN"}


def _read_sql_from_args(args) -> str:
    if args.sql:
        return args.sql
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    # stdin
    data = sys.stdin.read()
    return data


def _parse_params(args) -> Optional[List]:
    if args.params_json:
        try:
            val = json.loads(args.params_json)
            if isinstance(val, list):
                return val
            raise ValueError("--params-json 需要是 JSON 数组，如 [""000001.SZ"", ""20240101""]")
        except Exception as e:
            raise SystemExit(f"解析 --params-json 失败: {e}")
    if args.params:
        return [p.strip() for p in args.params.split(',')]
    return None


def run_query(conn: sqlite3.Connection, sql: str, params: Optional[List], fmt: str, output: Optional[str], max_rows: int):
    # pandas 对多语句不友好，若是查询但包含分号多语句，尝试提取第一句
    clean = _strip_comments(sql)
    parts = [p.strip() for p in clean.split(';') if p.strip()]
    if len(parts) >= 2 and _is_select_like(parts[0]):
        sql = parts[0]

    df = pd.read_sql_query(sql, conn, params=params)
    total = len(df)
    print(f"查询成功，返回 {total} 行，{df.shape[1]} 列。")
    if output:
        if fmt == 'csv':
            df.to_csv(output, index=False, encoding='utf-8-sig')
            print(f"结果已保存到 CSV: {output}")
        elif fmt == 'json':
            df.to_json(output, orient='records', force_ascii=False)
            print(f"结果已保存到 JSON: {output}")
        else:
            print("不支持的格式，仅支持 csv/json。")
    else:
        # 控制台预览
        preview = df.head(max_rows)
        # 尽量以表格形式输出
        try:
            from tabulate import tabulate  # 可选依赖
            print(tabulate(preview, headers='keys', tablefmt='github', showindex=False))
        except Exception:
            # 回退为 Pandas 自带打印
            print(preview.to_string(index=False))


def run_script(conn: sqlite3.Connection, sql: str):
    before = conn.total_changes
    conn.executescript(sql)
    conn.commit()
    changed = conn.total_changes - before
    print(f"脚本执行成功。受影响记录数: {changed}")


def main():
    parser = argparse.ArgumentParser(description="运行任意 SQLite SQL（查询/非查询）")
    parser.add_argument("--db", type=str, default=DEFAULT_DB, help="数据库路径，默认 data/stock_data.db")
    parser.add_argument("--sql", type=str, help="要执行的 SQL（单条或多条；查询建议单条）")
    parser.add_argument("--file", type=str, help="从 .sql 文件读取 SQL（可多语句）")
    parser.add_argument("--params", type=str, help="以逗号分隔的参数列表，对 '?' 占位符绑定")
    parser.add_argument("--params-json", type=str, help="JSON 数组形式的参数列表，如 '[""000001.SZ"", ""20240101""]'")
    parser.add_argument("--output", type=str, help="将查询结果保存到文件（仅查询模式）")
    parser.add_argument("--format", type=str, default='csv', choices=['csv','json'], help="保存格式：csv 或 json")
    parser.add_argument("--max-rows", type=int, default=50, help="控制台预览行数（仅查询模式）")

    args = parser.parse_args()

    if not (args.sql or args.file):
        # 允许 stdin
        print("未提供 --sql 或 --file，将从 stdin 读取 SQL。按 Ctrl+D 结束输入。")

    sql = _read_sql_from_args(args)
    if not sql or not sql.strip():
        print("空 SQL，退出。")
        return

    if not os.path.exists(args.db):
        print(f"数据库不存在: {args.db}")
        return

    params = _parse_params(args)

    conn = sqlite3.connect(args.db)
    try:
        if _is_select_like(sql):
            run_query(conn, sql, params, fmt=args.format, output=args.output, max_rows=args.max_rows)
        else:
            if params:
                # 单条非查询语句带参数
                cur = conn.cursor()
                before = conn.total_changes
                cur.execute(sql, params)
                conn.commit()
                changed = conn.total_changes - before
                print(f"语句执行成功。受影响记录数: {changed}")
            else:
                # 多语句脚本
                run_script(conn, sql)
    except Exception as e:
        print(f"执行失败: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

