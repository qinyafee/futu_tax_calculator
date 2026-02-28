#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SHEET_NAME = "证券-交易流水"
COLUMN_TO_DROP = "账户号码"


def _iter_stock_xlsx(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("*stock.xlsx"))
    return [p for p in paths if not p.name.startswith(".~lock.")]


def extract_one(xlsx_path: Path, output_dir: Path) -> Path:
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)

    if COLUMN_TO_DROP in df.columns:
        df = df.drop(columns=[COLUMN_TO_DROP])

    # 数量/面值 列中如有负值，统一转为正数。应对卖出方向的负值
    if "数量/面值" in df.columns:
        df["数量/面值"] = df["数量/面值"].abs()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{xlsx_path.stem}_{SHEET_NAME}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"从 *stock.xlsx 提取 {SHEET_NAME} sheet 为 CSV（删除 {COLUMN_TO_DROP} 列）"
    )
    parser.add_argument(
        "--input-dir",
        default="mydata",
        help="包含 *stock.xlsx 的目录（默认：mydata）",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="输出 CSV 的目录（默认：data）",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = (repo_root / args.input_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    xlsx_files = _iter_stock_xlsx(input_dir)
    if not xlsx_files:
        print(f"未找到文件: {input_dir}/*stock.xlsx")
        return 0

    outputs: list[Path] = []
    for xlsx_path in xlsx_files:
        out = extract_one(xlsx_path, output_dir)
        outputs.append(out)
        print(f"✅ {xlsx_path.name} -> {out.relative_to(repo_root)}")

    print(f"完成：共生成 {len(outputs)} 个 CSV，输出目录：{output_dir.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

