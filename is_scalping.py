import pandas as pd
import numpy as np

# ========= 参数区（可按需调整） =========
SHORT_SECONDS = 180            # “秒单”阈值：< 3分钟
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
        short_trades_3m=("is_short", "sum"),
        short_trades_1m=("is_scalping", "sum"),
        trades_3m_profit=("Profit", lambda x: x[df.loc[x.index, "is_short"]].sum()),
        has_1m_big_win=("is_1m_big_win", "any"),
        total_lots=("Volume", "sum"),
        trades_3m_lots=("Volume", lambda x: x[df.loc[x.index, "is_short"]].sum()),
        profitable_lots=("profit_volume", "sum"),
    )

    # 平均持仓时间
    base['avg_holding_seconds'] = base['total_holding_seconds'] / base['total_trades']
    # 平均每手利潤
    base['profit_per_lot'] = base['total_profit'] / base['total_lots']
    # 3分钟内平仓交易数量占比
    base["trades_3m_percent"] = base["short_trades_3m"] / base["total_trades"]
    # 净利润
    profit = base["total_profit"]
    profit = profit.dropna()
    base["profit_3m_percent"] = base["trades_3m_profit"] / profit
    base["profit_3m_percent"] = base["profit_3m_percent"].fillna(0.0)  # for multiple accounts

    # 条件A：持仓时间低于3分钟 & 短单数量占比 & 短单利润占比
    cond_A = (base["short_trades_3m"].astype(bool)) & (base["trades_3m_percent"] >= COUNT_SHARE_TH) & (base["profit_3m_percent"] >= PROFIT_SHARE_TH)
    # 条件B：持仓时间低于1分钟 & 存在单笔盈利 > 阈值
    cond_B = base["has_1m_big_win"]

    base["is_scalping"] = cond_A | cond_B
    base["reason"] = np.select(
        [cond_A & cond_B, cond_A, cond_B],
        ["持仓时间低于3分钟 & 短单数量占比20% & 短单利润占比10% + 持仓时间低于1分钟 & 存在单笔盈利超过200",
         "持仓时间低于3分钟 & 短单数量占比20% & 短单利润占比10%", "持仓时间低于1分钟 & 存在单笔盈利超过200"],
        default=""
    )

    return base.sort_values(["is_scalping", "trades_3m_percent", "profit_3m_percent"], ascending=[False, False, False])


def detect_seconds_scalping(df: pd.DataFrame) -> pd.DataFrame:
    df1 = prepare_trades(df)
    report = aggregate_by_account(df1)
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
        short_trades_3m=("is_short", "sum"),
        short_trades_1m=("is_scalping", "sum"),
        trades_3m_profit=("Profit", lambda x: x[df.loc[x.index, "is_short"]].sum()),
        has_1m_big_win=("is_1m_big_win", "any"),
        total_lots=("Volume", "sum"),
        trades_3m_lots=("Volume", lambda x: x[df.loc[x.index, "is_short"]].sum()),
        profitable_lots=("profit_volume", "sum"),
    )

    # 平均持仓时间
    base['avg_holding_seconds'] = base['total_holding_seconds'] / base['total_trades']
    # 平均每手利潤
    base['profit_per_lot'] = base['total_profit'] / base['total_lots']
    # 3分钟内平仓交易数量占比
    base["trades_3m_percent"] = base["short_trades_3m"] / base["total_trades"]
    # 净利润
    profit = base["total_profit"]
    profit = profit.dropna()
    base["profit_3m_percent"] = base["trades_3m_profit"] / profit
    base["profit_3m_percent"] = base["profit_3m_percent"].fillna(0.0)  # for multiple accounts

    # 条件A：持仓时间低于3分钟 & 短单数量占比 & 短单利润占比
    cond_A = (base["short_trades_3m"].astype(bool)) & (base["trades_3m_percent"] >= COUNT_SHARE_TH) & (base["profit_3m_percent"] >= PROFIT_SHARE_TH)
    # 条件B：持仓时间低于1分钟 & 存在单笔盈利 > 阈值
    cond_B = base["has_1m_big_win"]

    base["is_scalping"] = cond_A | cond_B
    base["reason"] = np.select(
        [cond_A & cond_B, cond_A, cond_B],
        ["持仓时间低于3分钟 & 短单数量占比20% & 短单利润占比10% + 持仓时间低于1分钟 & 存在单笔盈利超过200",
         "持仓时间低于3分钟 & 短单数量占比20% & 短单利润占比10%", "持仓时间低于1分钟 & 存在单笔盈利超过200"],
        default=""
    )

    return base.sort_values(["is_scalping", "trades_3m_percent", "profit_3m_percent"], ascending=[False, False, False])


def detect_seconds_scalping_mt5(df: pd.DataFrame) -> pd.DataFrame:
    df2 = prepare_trades2(df)
    report = aggregate_by_account2(df2)
    return report


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Increase display width to prevent line wrapping
    '''暫時從本地文件讀取'''
    filename = "C:\\Users\\Frank W\\OneDrive - Logtec Innovation Limited\\Desktop\\Data Integration\\Abnormal Accounts\\Statement_1457318.htm"
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

    print(report)


