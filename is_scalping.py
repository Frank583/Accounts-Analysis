from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from typing import Optional, Sequence, List, Tuple


# ========= 参数区（可按需调整） =========
SHORT_SECONDS = 120            # “秒单”阈值：< 3分钟
COUNT_SHARE_TH = 0.2          # 短单数量占比阈值（20%）
PROFIT_SHARE_TH = 0.1         # 短单利润占整体利润占比阈值（10%）
SCALPING_SECONDS = 60            # “秒单”阈值：< 1分钟
SINGLE_TRADE_PROFIT_TH = 200  # 单笔盈利阈值（USD）
# =====================================

def prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    """规范字段 & 计算每笔持仓秒数、是否短单等。"""
    # 必要列校验
    required = {"Ticket", "Open Time", "Type", "Volume", "Close Time", "Profit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    # 过滤dataframe，仅保留该账户交易记录数据
    df = extract_dataframe(df)
    # 时间列转为 datetime
    df["Open Time"]  = pd.to_datetime(df["Open Time"], errors="coerce")
    df["Close Time"] = pd.to_datetime(df["Close Time"], errors="coerce")

    # 持仓时长（秒）
    hold_seconds = (df["Close Time"] - df["Open Time"]).dt.total_seconds()
    # 若存在负值/缺失，将其赋值为0
    hold_seconds = hold_seconds.clip(lower=0).fillna(0)
    df["hold_seconds"] = hold_seconds

    # 是否“秒单”（短于阈值）
    df["is_short"] = df["hold_seconds"] < SHORT_SECONDS
    df['is_scalping'] = df["hold_seconds"] < SCALPING_SECONDS

    # 是否单笔大利单
    df["Profit"] = df["Profit"].astype(str).str.replace(" ", "", regex=False)
    df["Profit"] = pd.to_numeric(df["Profit"], errors='coerce')

    # 是否為美分賬戶
    df["is_cent_related"] = df["Item"].str.lower().str.contains("cent")
    if df["is_cent_related"].any():
        df["is_1m_big_win"] = (df['is_scalping']) & (df["Profit"] > SINGLE_TRADE_PROFIT_TH * 100)
    else:
        df["is_1m_big_win"] = (df['is_scalping']) & (df["Profit"] > SINGLE_TRADE_PROFIT_TH)
    print(df[df["is_1m_big_win"] == True])
    report_rows = df[df["is_1m_big_win"] == True]
    report_rows.to_excel("C:\\Users\\Frank W\\BTC_ETH_BCH_users\\1450435_rows.xlsx")
    print("----------------------------------")

    # 交易手術
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
    # 盈利手術
    df['profit_volume'] = df[df['Profit'] > 0]['Volume']
    return df


def extract_dataframe(df):
    # 将带有“Closed P/L:	”的行作为结尾行
    keyword = "Closed P/L:"
    df = df.astype(str)
    row_match = df.apply(lambda row: row.str.contains(keyword, case=False, na=False)).any(axis=1)
    match_indices = df.index[row_match]
    if not match_indices.empty:
        cutoff_index = match_indices[0]
        df = df.loc[:cutoff_index - 2]  # "Closed P/L:"前两行
    else:
        df = df.copy()
    # 統計'buy'和'sell'的交易
    df = df[df['Type'].astype(str).isin(['buy', 'sell'])]
    # 删除Ticket无效行
    df = df.dropna(subset=['Ticket'])
    return df


def aggregate_by_account(df: pd.DataFrame) -> pd.DataFrame:
    """按交易品種聚合，计算各项指标与是否触发规则。"""
    g = df.groupby("Item", as_index=False)
    # 逐账户基础统计
    base = g.agg(
        total_trades=("Profit", "size"),
        total_holding_seconds=("hold_seconds", "sum"),
        total_profit=("Profit", "sum"),
        short_trades_2m=("is_short", "sum"),
        short_trades_1m=("is_scalping", "sum"),
        trades_2m_profit=("Profit", lambda x: x[df.loc[x.index, "is_short"]].sum()),
        has_1m_big_win=("is_1m_big_win", "any"),
        total_lots=("Volume", "sum"),
        trades_2m_lots=("Volume", lambda x: x[df.loc[x.index, "is_short"]].sum()),
        profitable_lots=("profit_volume", "sum"),

        trades_long=("Type", lambda x: (df.loc[x.index, "Type"] == "buy").sum()),  # 多单交易笔数
        trades_short=("Type", lambda x: (df.loc[x.index, "Type"] == "sell").sum()),  # 空单交易笔数
        total_profit_long=("Profit", lambda x: x[df.loc[x.index, "Type"] == "buy"].sum()),  # 多单累计利润
        total_profit_short=("Profit", lambda x: x[df.loc[x.index, "Type"] == "sell"].sum()),  # 空单累计利润
        profit_volume_long=("profit_volume", lambda x: x[df.loc[x.index, "Type"] == "buy"].sum()),  # 多单盈利手数
        profit_volume_short=("profit_volume", lambda x: x[df.loc[x.index, "Type"] == "sell"].sum()),  # 空单盈利手数
    )

    # 平均持仓时间
    base['avg_holding_seconds'] = base['total_holding_seconds'] / base['total_trades']
    # 平均每手利潤
    base['profit_per_lot'] = base['total_profit'] / base['total_lots']
    # 3分钟内平仓交易数量占比
    base["trades_2m_percent"] = base["short_trades_2m"] / base["total_trades"]
    # 净利润
    profit = base["total_profit"]
    profit = profit.dropna()
    base["profit_2m_percent"] = base["trades_2m_profit"] / profit
    base["profit_2m_percent"] = base["profit_2m_percent"].fillna(0.0)  # for multiple accounts

    # 条件A：持仓时间低于3分钟 & 短单数量占比 & 短单利润占比
    cond_A = (base["short_trades_2m"].astype(bool)) & (base["trades_2m_percent"] >= COUNT_SHARE_TH) & (base["profit_2m_percent"] >= PROFIT_SHARE_TH)
    # 条件B：持仓时间低于1分钟 & 存在单笔盈利 > 阈值
    cond_B = base["has_1m_big_win"]

    base["is_scalping"] = cond_A | cond_B
    base["reason"] = np.select(
        [cond_A & cond_B, cond_A, cond_B],
        ["持仓时间低于2分钟 & 短单数量占比20% & 短单利润占比10% + 持仓时间低于1分钟 & 存在单笔盈利超过200",
         "持仓时间低于2分钟 & 短单数量占比20% & 短单利润占比10%", "持仓时间低于1分钟 & 存在单笔盈利超过200"],
        default=""
    )

    # ---- 多空盈利占比分布计算 ----
    # 防止 total_profit 为 0 导致除以 0
    base["profit_long_percent"] = 0.0
    base["profit_short_percent"] = 0.0

    # 仅对非空 total_profit 计算占比
    nonzero_mask = base["total_profit"] != 0
    base.loc[nonzero_mask, "profit_long_percent"] = (
            base.loc[nonzero_mask, "total_profit_long"] / base.loc[nonzero_mask, "total_profit"]
    )
    base.loc[nonzero_mask, "profit_short_percent"] = (
            base.loc[nonzero_mask, "total_profit_short"] / base.loc[nonzero_mask, "total_profit"]
    )

    # 多/空交易占比
    base["trades_long_percent"] = base["trades_long"] / base["total_trades"]
    base["trades_short_percent"] = base["trades_short"] / base["total_trades"]

    return base.sort_values(["is_scalping", "trades_2m_percent", "profit_2m_percent"], ascending=[False, False, False])


def detect_seconds_scalping(df: pd.DataFrame) -> pd.DataFrame:
    df1 = prepare_trades(df)
    report = aggregate_by_account(df1)

    # 将前 12 个饼图保存到本地文件夹
    saved = plot_profit_share_grid(report, cols=4, figsize=(16,9), save_path="all_symbols_profit_grid.png", show=True)
    print(saved)

    return report


def prepare_trades2(df: pd.DataFrame) -> pd.DataFrame:
    # 必要列校验
    required = {"Position", "Symbol", "Open Time", "Close Time", "Type", "Volume", "Profit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    # 过滤dataframe，仅保留该账户交易记录数据
    df = extract_dataframe2(df)
    # 时间列转为 datetime
    df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")
    df["Close Time"] = pd.to_datetime(df["Close Time"], errors="coerce")

    # 持仓时长（秒）
    hold_seconds = (df["Close Time"] - df["Open Time"]).dt.total_seconds()
    # 若存在负值/缺失，将其赋值为0
    hold_seconds = hold_seconds.clip(lower=0).fillna(0)
    df["hold_seconds"] = hold_seconds

    # 是否“秒单”（短于阈值）
    df["is_short"] = df["hold_seconds"] < SHORT_SECONDS
    df['is_scalping'] = df["hold_seconds"] < SCALPING_SECONDS

    # 是否单笔大利单
    df["Profit"] = df["Profit"].astype(str).str.replace(" ", "", regex=False)
    df["Profit"] = pd.to_numeric(df["Profit"], errors='coerce')

    # 是否為美分賬戶
    df["is_cent_related"] = df["Symbol"].str.lower().str.contains("cent")
    if df["is_cent_related"].any():
        df["is_1m_big_win"] = (df['is_scalping']) & (df["Profit"] > SINGLE_TRADE_PROFIT_TH * 100)
    else:
        df["is_1m_big_win"] = (df['is_scalping']) & (df["Profit"] > SINGLE_TRADE_PROFIT_TH)
    print(df[df["is_1m_big_win"] == True])
    # report_rows = df[df["is_1m_big_win"] == True]
    # report_rows.to_excel("C:\\Users\\Frank W\\BTC_ETH_BCH_users\\1457819_rows.xlsx")
    print("----------------------------------")

    # 交易手術
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
    # 盈利手術
    df['profit_volume'] = df[df['Profit'] > 0]['Volume']
    return df


def extract_dataframe2(df):
    # 将带有“Orders”的行作为结尾行
    keyword = "Orders"
    df = df.astype(str)
    row_match = df.apply(lambda row: row.str.contains(keyword, case=False, na=False)).any(axis=1)
    match_indices = df.index[row_match]
    if not match_indices.empty:
        cutoff_index = match_indices[0]
        df = df.loc[:cutoff_index - 2]  # "Orders"前两行
    else:
        df = df.copy()
    # 統計'buy'和'sell'的交易
    df = df[df['Type'].astype(str).isin(['buy', 'sell'])]
    # 删除Ticket无效行
    df = df.dropna(subset=['Position'])
    return df


def aggregate_by_account2(df: pd.DataFrame) -> pd.DataFrame:
    """按交易品種聚合，计算各项指标与是否触发规则。"""
    g = df.groupby("Symbol", as_index=False)
    # 逐账户基础统计
    base = g.agg(
        total_trades=("Profit", "size"),
        total_holding_seconds=("hold_seconds", "sum"),
        total_profit=("Profit", "sum"),
        short_trades_2m=("is_short", "sum"),
        short_trades_1m=("is_scalping", "sum"),
        trades_2m_profit=("Profit", lambda x: x[df.loc[x.index, "is_short"]].sum()),
        has_1m_big_win=("is_1m_big_win", "any"),
        total_lots=("Volume", "sum"),
        trades_2m_lots=("Volume", lambda x: x[df.loc[x.index, "is_short"]].sum()),
        profitable_lots=("profit_volume", "sum"),

        trades_long=("Type", lambda x: (df.loc[x.index, "Type"] == "buy").sum()),  #多单交易笔数
        trades_short=("Type", lambda x: (df.loc[x.index, "Type"] == "sell").sum()),  #空单交易笔数
        total_profit_long=("Profit", lambda x: x[df.loc[x.index, "Type"] == "buy"].sum()),  #多单累计利润
        total_profit_short=("Profit", lambda x: x[df.loc[x.index, "Type"] == "sell"].sum()),  #空单累计利润
        profit_volume_long=("profit_volume", lambda x: x[df.loc[x.index, "Type"] == "buy"].sum()),  #多单盈利手数
        profit_volume_short=("profit_volume", lambda x: x[df.loc[x.index, "Type"] == "sell"].sum()),  #空单盈利手数

    )

    # 平均持仓时间
    base['avg_holding_seconds'] = base['total_holding_seconds'] / base['total_trades']
    # 平均每手利潤
    base['profit_per_lot'] = base['total_profit'] / base['total_lots']
    # 3分钟内平仓交易数量占比
    base["trades_2m_percent"] = base["short_trades_2m"] / base["total_trades"]
    # 净利润
    profit = base["total_profit"]
    profit = profit.dropna()
    base["profit_2m_percent"] = base["trades_2m_profit"] / profit
    base["profit_2m_percent"] = base["profit_2m_percent"].fillna(0.0)  # for multiple accounts

    # 条件A：持仓时间低于3分钟 & 短单数量占比 & 短单利润占比
    cond_A = (base["short_trades_2m"].astype(bool)) & (base["trades_2m_percent"] >= COUNT_SHARE_TH) & (base["profit_2m_percent"] >= PROFIT_SHARE_TH)
    # 条件B：持仓时间低于1分钟 & 存在单笔盈利 > 阈值
    cond_B = base["has_1m_big_win"]

    base["is_scalping"] = cond_A | cond_B
    base["reason"] = np.select(
        [cond_A & cond_B, cond_A, cond_B],
        ["持仓时间低于2分钟 & 短单数量占比20% & 短单利润占比10% + 持仓时间低于1分钟 & 存在单笔盈利超过200",
         "持仓时间低于2分钟 & 短单数量占比20% & 短单利润占比10%", "持仓时间低于1分钟 & 存在单笔盈利超过200"],
        default=""
    )

    return base.sort_values(["is_scalping", "trades_2m_percent", "profit_2m_percent"], ascending=[False, False, False])


def detect_seconds_scalping_mt5(df: pd.DataFrame) -> pd.DataFrame:
    df2 = prepare_trades2(df)
    report = aggregate_by_account2(df2)
    return report



# def plot_profit_share_pies(
#     base: pd.DataFrame,
#     *,
#     symbol_col: str = "Symbol",
#     profit_long_col: str = "total_profit_long",
#     profit_short_col: str = "total_profit_short",
#     profit_long_pct_col: str = "profit_long_percent",
#     profit_short_pct_col: str = "profit_short_percent",
#     top_n: Optional[int] = 12,
#     sort_by_abs_profit: bool = True,
#     figsize: Tuple[float, float] = (6, 6),
#     cmap: Optional[Sequence[str]] = None,
#     save_dir: Optional[str] = None,
#     show: bool = True,
#     explode_long: float = 0.05,
#     explode_short: float = 0.0,
# ) -> List[str]:
#     """
#     为每个 Symbol 绘制多/空盈利占比饼图并返回保存路径列表（如果 save_dir 提供）。
#     图上同时显示 total_buy_profit 和 total_sell_profit（若存在）。
#     """
#     if cmap is None:
#         cmap = ["#4CAF50", "#F44336"]
#
#     df = base.copy()
#
#     # 准备百分比列（若不存在则从绝对利润或 total_profit 计算）
#     if profit_long_pct_col not in df.columns or profit_short_pct_col not in df.columns:
#         df[profit_long_pct_col] = 0.0
#         df[profit_short_pct_col] = 0.0
#         total_profit = df.get("total_profit")
#         if total_profit is None:
#             df[profit_long_pct_col] = np.nan
#             df[profit_short_pct_col] = np.nan
#         else:
#             nonzero = total_profit != 0
#             df.loc[nonzero, profit_long_pct_col] = df.loc[nonzero, profit_long_col] / df.loc[nonzero, "total_profit"]
#             df.loc[nonzero, profit_short_pct_col] = df.loc[nonzero, profit_short_col] / df.loc[nonzero, "total_profit"]
#             df[[profit_long_pct_col, profit_short_pct_col]] = df[[profit_long_pct_col, profit_short_pct_col]].fillna(0.0)
#
#     # 选择 top_n
#     if top_n is not None:
#         if sort_by_abs_profit and "total_profit" in df.columns:
#             df["_abs_profit_sort"] = df["total_profit"].abs()
#             df = df.sort_values("_abs_profit_sort", ascending=False).head(top_n).drop(columns=["_abs_profit_sort"])
#         else:
#             df = df.head(top_n)
#
#     saved_paths: List[str] = []
#     for _, row in df.iterrows():
#         sym = row.get(symbol_col, "")
#         # 获取 buy/sell 绝对利润值（如果列名不同请传入正确列名）
#         buy_val = float(row.get(profit_long_col, 0.0) or 0.0)
#         sell_val = float(row.get(profit_short_col, 0.0) or 0.0)
#
#         # 优先使用百分比列（可能为负），否则回退到按绝对值计算占比
#         long_pct = float(row.get(profit_long_pct_col, 0.0) or 0.0)
#         short_pct = float(row.get(profit_short_pct_col, 0.0) or 0.0)
#
#         # 若百分比全为0或不存在，则用绝对值贡献计算
#         if np.isclose(long_pct + short_pct, 0.0):
#             a_abs = abs(buy_val)
#             b_abs = abs(sell_val)
#             s_abs = a_abs + b_abs
#             if s_abs == 0:
#                 continue
#             sizes = np.array([a_abs, b_abs]) / s_abs
#             long_pct, short_pct = sizes[0], sizes[1]
#         else:
#             arr = np.array([long_pct, short_pct], dtype=float)
#             arr = np.nan_to_num(arr, nan=0.0)
#             arr_abs = np.abs(arr)
#             s = arr_abs.sum()
#             if s == 0:
#                 continue
#             sizes = arr_abs / s
#
#         # 标签中加入数值显示，带符号（正为 +，负为 -）
#         def fmt_val(v: float) -> str:
#             sign = "+" if v > 0 else ("-" if v < 0 else "")
#             return f"{sign}{abs(v):.2f}"
#
#         labels = [
#             f"Long {long_pct:.1%} ({fmt_val(buy_val)})",
#             f"Short {short_pct:.1%} ({fmt_val(sell_val)})"
#         ]
#         explode = (explode_long, explode_short)
#
#         fig, ax = plt.subplots(figsize=figsize)
#         wedges, texts, autotexts = ax.pie(
#             sizes,
#             labels=labels,
#             autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
#             startangle=90,
#             colors=list(cmap),
#             explode=explode,
#             wedgeprops={"edgecolor": "w"}
#         )
#         ax.axis("equal")
#         ax.set_title(f"{sym} Profit Share", fontsize=12)
#
#         # 在图下方显示总净利与 buy/sell 总利（带符号）
#         total_profit_val = row.get("total_profit", None)
#         lines = []
#         if total_profit_val is not None:
#             lines.append(f"Total profit: {total_profit_val:.2f}")
#         lines.append(f"Total buy profit: {buy_val:+.2f}")
#         lines.append(f"Total sell profit: {sell_val:+.2f}")
#         ax.text(0, -1.18, " | ".join(lines), ha="center", fontsize=9)
#
#         plt.tight_layout()
#
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             safe_sym = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(sym))
#             fname = os.path.join(save_dir, f"{safe_sym}_profit_share.png")
#             fig.savefig(fname, dpi=150)
#             saved_paths.append(fname)
#
#         if show:
#             plt.show()
#         else:
#             plt.close(fig)
#
#     return saved_paths
def plot_profit_share_grid(
    base: pd.DataFrame,
    *,
    symbol_col: str = "Symbol",
    profit_buy_col: str = "total_profit_long",
    profit_sell_col: str = "total_profit_short",
    profit_buy_pct_col: str = "profit_long_percent",
    profit_sell_pct_col: str = "profit_short_percent",
    top_n: Optional[int] = None,            # keep top_n optional
    sort_by_abs_profit: bool = True,
    cols: int = 4,
    figsize: Tuple[float, float] = (16, 9),
    cmap: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    explode_buy: float = 0.05,
    explode_sell: float = 0.0,
) -> Optional[str]:
    """
    Draw one figure containing a grid of pie charts, one pie per symbol.
    Uses all symbols from `base` unless `top_n` is provided to limit the count.
    """
    if cmap is None:
        cmap = ["#4CAF50", "#F44336"]

    df = base.copy()

    # If top_n is provided, select top_n symbols by abs(total_profit) or first top_n rows
    if top_n is not None:
        if sort_by_abs_profit and "total_profit" in df.columns:
            df = df.assign(_abs_profit_sort=df["total_profit"].abs()).sort_values("_abs_profit_sort", ascending=False).head(top_n).drop(columns=["_abs_profit_sort"])
        else:
            df = df.head(top_n)

    df = df.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return None

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # normalize axes to flat list
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for ax in axes_flat[n:]:
        ax.axis("off")

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes_flat[i]
        sym = row.get(symbol_col, "")

        buy_val = float(row.get(profit_buy_col, 0.0) or 0.0)
        sell_val = float(row.get(profit_sell_col, 0.0) or 0.0)

        a_abs = abs(buy_val)
        b_abs = abs(sell_val)
        s_abs = a_abs + b_abs
        if s_abs == 0:
            ax.text(0.5, 0.5, f"{sym}\nno profit data", ha="center", va="center")
            ax.axis("off")
            continue
        sizes = np.array([a_abs, b_abs]) / s_abs

        signed_buy_pct = row.get(profit_buy_pct_col, None)
        signed_sell_pct = row.get(profit_sell_pct_col, None)
        if signed_buy_pct is None or signed_sell_pct is None:
            total_profit = row.get("total_profit", None)
            if total_profit and total_profit != 0:
                signed_buy_pct = buy_val / total_profit
                signed_sell_pct = sell_val / total_profit
            else:
                signed_buy_pct = sizes[0]
                signed_sell_pct = sizes[1]

        def fmt_val(v: float) -> str:
            return f"{v:+.2f}"

        labels = [
            f"Buy {signed_buy_pct:.1%} ({fmt_val(buy_val)})",
            f"Sell {signed_sell_pct:.1%} ({fmt_val(sell_val)})",
        ]

        explode = (explode_buy, explode_sell)
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            startangle=90,
            colors=list(cmap),
            explode=explode,
            wedgeprops={"edgecolor": "w"},
            textprops={"fontsize": 9},
        )
        ax.axis("equal")
        ax.set_title(f"{sym}", fontsize=10)

        total_profit_val = row.get("total_profit", None)
        each_symbol = row.get("Item", None);
        parts = []
        if total_profit_val is not None:
            parts.append(f"{each_symbol}")
            parts.append(f"Total: {total_profit_val:+.2f}")
        parts.append(f"Buy: {buy_val:+.2f}")
        parts.append(f"Sell: {sell_val:+.2f}")
        ax.text(0, -1.5, " | ".join(parts), ha="center", fontsize=8)
        # ax.text(0, -1.25, str(sym), ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if not show:
            plt.close(fig)
        return save_path

    if show:
        plt.show()
    else:
        plt.close(fig)
    return None







if __name__ == "__main__":
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Increase display width to prevent line wrapping
    '''暫時從本地文件讀取'''
    filename = "C:\\Users\\Frank W\\OneDrive - Logtec Innovation Limited\\Desktop\\Data Integration\\Abnormal Accounts\\Statement_3682924.htm"
    # filename = "C:\\Users\\Frank W\\OneDrive - Logtec Innovation Limited\\Documents\\ReportHistory-904943.html"
    # filename = "C:\\Users\\Frank W\\BTC_ETH_BCH_users\\Statement_3681909.htm"
    # MT4 report files
    if filename.endswith(".htm"):
        trades = pd.read_html(filename, header=2)[0]
        report = detect_seconds_scalping(trades)
    # MT5 report files
    elif filename.endswith(".html"):
        trades = pd.read_html(filename, header=8)[0]
        trades.rename(columns={
            'Time': 'Open Time',
            'Volume': 'Undefined',
            'Profit': 'Volume',
            'Unnamed: 16': 'Close Time',
            'Unnamed: 20': 'Profit'
        }, inplace=True)
        filtered_trades = trades[['Open Time', 'Position', 'Symbol', 'Type', 'Volume', 'Close Time', 'Profit']]
        report = detect_seconds_scalping_mt5(filtered_trades)
    # Excel .csv file
    elif filename.endswith(".csv"):
        trades = pd.read_csv(filename, header=8)[0]
        trades.rename(columns={
            'Time': 'Open Time',
            'Volume': 'Undefined',
            'Profit': 'Volume',
            'Unnamed: 16': 'Close Time',
            'Unnamed: 20': 'Profit'
        }, inplace=True)
        filtered_trades = trades[['Open Time', 'Position', 'Symbol', 'Type', 'Volume', 'Close Time', 'Profit']]
        report = detect_seconds_scalping_mt5(filtered_trades)

    print(report)
    today = datetime.today().date()
    # print(today)
    # report.to_excel("C:\\Users\\Frank W\\BTC_ETH_BCH_users\\1450435.xlsx")
