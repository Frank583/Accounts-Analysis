import pytz
import asyncio
import aiomysql
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import date, datetime, timedelta
import MetaTrader5 as mt5

# timezone = pytz.timezone("Etc/UTC")
# date_from = datetime(2025, 9, 5, 1, 12, tzinfo=timezone)
# date_to   = datetime(2025, 9, 5, 1, 14, tzinfo=timezone)

#账户信息（自定义）
# login = 3680216
login = 0
comment_str = "期迹%"
account_type = 1
today = date.today().strftime("%Y-%m-%d")
yesterday = (date.today() - timedelta(1)).strftime("%Y-%m-%d")
date_str = date.today().strftime("2025-09-21 09:00:00")
end_date_str = date.today().strftime("2025-09-22 09:00:00")
# end_date_str = ""
month_str = date_str.split("-")[1]


# param = (date_str, comment_str)
#query all trades by 'comment' when login is 0
def build_trade_query(login, date_str, end_date_str, comment_str, month_str):
    base_query = (
        "SELECT login, open_time, cmd, symbol, lots, open_price, close_time, close_price, business_group, "
        "profit_usd AS profit, comment FROM cpt.trade_transaction25" + month_str
    )

    if login == 0:
        if end_date_str:
            query = (
                base_query +
                # " WHERE DATE(close_time) >= %s "
                " WHERE close_time >= %s "
                "AND close_time <= %s "
                "AND comment LIKE %s;"
            )
            param = (date_str, end_date_str, comment_str)
        else:
            query = (
                base_query +
                " WHERE close_time >= %s "
                "AND comment LIKE %s;"
            )
            param = (date_str, comment_str)
    else:
        if end_date_str:
            query = (
                base_query +
                " WHERE login = %s "
                "AND close_time >= %s "
                "AND close_time <= %s "
                "AND comment IS NOT NULL AND comment <> '';"
            )
            param = (login, date_str, end_date_str)
        else:
            query = (
                base_query +
                " WHERE login = %s "
                "AND close_time >= %s "
                "AND comment IS NOT NULL AND comment <> '';"
            )
            param = (login, date_str)

    return query, param
# if login == 0:
#     # query range
#     if end_date_str:
#         query = (  #交易记录（包括交易量和盈亏）
#             "select login, open_time, cmd, symbol, lots, open_price, close_time, close_price, profit_usd as profit, comment "
#             "from cpt.trade_transaction25" + month_str
#             + " where DATE(close_time) >= %s "
#               "and DATE(close_time) < %s and comment like %s;")
#         param = (date_str, end_date_str, comment_str)
#     else:
#         query = (
#             "select login, open_time, cmd, symbol, lots, open_price, close_time, close_price, profit_usd as profit, comment "
#             "from cpt.trade_transaction25" + month_str
#             + " where DATE(close_time) >= %s and comment like %s;")
#         param = (date_str, comment_str)
# else:
#     if end_date_str:
#         query = (
#                 "select login, open_time, cmd, symbol, lots, open_price, close_time, close_price, profit_usd as profit, comment "
#                 "from cpt.trade_transaction25" + month_str + " where login = %s "
#                 "and DATE(close_time) >= %s and DATE(close_time) < %s "
#                 "and comment IS NOT NULL and comment <> '';")
#         param = (login, date_str, end_date_str)
#     else:
#         query = (
#                 "select login, open_time, cmd, symbol, lots, open_price, close_time, close_price, profit_usd as profit, comment "
#                 "from cpt.trade_transaction25" + month_str + " where login = %s and DATE(close_time) >= %s "
#                 "and comment IS NOT NULL "
#                 "and comment <> '';")
#         param = (login, date_str)



async def query_from_database(pool):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            query, param = build_trade_query(login, date_str, end_date_str, comment_str, month_str)
            await cur.execute(query, param)
            transaction = await cur.fetchall()

            if login == 0:
                # Extract all login values inside 'transaction' resultset
                login_values = [row[0] for row in transaction]
                # WHERE login IN (%s, %s, ...)
                login_placeholders = ', '.join(['%s'] * len(login_values))

                if end_date_str:
                    query_deposit = (
                        f"SELECT login, account_type, usd_amount, completion_time FROM account_deposit_transaction "
                        f"WHERE login IN ({login_placeholders}) "
                        f"AND account_type = %s AND completion_time >= %s "
                        f"AND completion_time <= %s;"
                    )

                    query_withdraw = (
                        f"select login, account_type, usd_amount, completion_time from account_withdraw_transaction "
                        f"where login in ({login_placeholders}) "
                        f"and account_type = %s and completion_time >= %s "
                        f"and completion_time <= %s;")

                    params = login_values + [account_type, date_str, end_date_str]

                else:
                    query_deposit = (
                        f"select login, account_type, usd_amount, completion_time from account_deposit_transaction "
                        f"where login in ({login_placeholders}) "
                        f"and account_type = %s and completion_time >= %s;")

                    query_withdraw = (
                        f"select login, account_type, usd_amount, completion_time from account_withdraw_transaction "
                        f"where login in ({login_placeholders}) "
                        f"and account_type = %s and completion_time >= %s;")

                    params = login_values + [account_type, date_str]

            else:
                if end_date_str:
                    # 出入金记录
                    query_deposit = (
                        f"select login, account_type, usd_amount, completion_time from account_deposit_transaction "
                        f"where login = %s and account_type = %s and completion_time >= %s "
                        f"and completion_time <= %s;")

                    query_withdraw = (
                        f"select login, account_type, usd_amount, completion_time from account_withdraw_transaction "
                        f"where login = %s and account_type = %s and completion_time >= %s "
                        f"and completion_time <= %s;")

                    params = (login, account_type, date_str, end_date_str)

                else:
                    # 出入金记录
                    query_deposit = (
                        f"select login, account_type, usd_amount, completion_time from account_deposit_transaction "
                        f"where login = %s and account_type = %s and completion_time >= %s;")

                    query_withdraw = (
                        f"select login, account_type, usd_amount, completion_time from account_withdraw_transaction "
                        f"where login = %s and account_type = %s and completion_time >= %s;")

                    params = (login, account_type, date_str)


            await cur.execute(query_deposit, params)
            deposit = await cur.fetchall()

            await cur.execute(query_withdraw, params)
            withdraw = await cur.fetchall()

            return transaction, deposit, withdraw


# async def get_transaction_data(pool):
#     async with pool.acquire() as conn:
#         async with conn.cursor() as cur:
#             await cur.execute(query, param)
#             transaction = await cur.fetchall()
#             return transaction
#
# async def get_deposit_data(pool):
#     async with pool.acquire() as conn:
#         async with conn.cursor() as cur:
#             await cur.execute(query_deposit, params)
#             deposit = await cur.fetchall()
#             return deposit
#
# async def get_withdraw_data(pool):
#     async with pool.acquire() as conn:
#         async with conn.cursor() as cur:
#             await cur.execute(query_withdraw, params)
#             withdraw = await cur.fetchall()
#             return withdraw


async def run_task():
    pool = await aiomysql.create_pool(
        host='proxy.cptmarkets.com',
        port=50713,
        user='new_frank.w',
        password='XH5mJ!MEHi#QFJN3',
        db='cpt',
        minsize=1,
        maxsize=3
    )

    results = await asyncio.gather(
        # get_transaction_data(pool),
        # get_deposit_data(pool),
        # get_withdraw_data(pool)
        query_from_database(pool)
    )

    transaction, deposit, withdraw = results[0]

    # list_results = list(results[0])
    columns = ['login', 'open_time', 'cmd', 'symbol', 'lots', 'open_price', 'close_time', 'close_price', 'business_group', 'profit', 'comment']
    df = pd.DataFrame(transaction, columns=columns)
    decimal_cols = ['lots', 'open_price', 'close_price', 'profit']
    df[decimal_cols] = df[decimal_cols].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df['close_time'] = pd.to_datetime(df['close_time'])

    #find the top 30 users by total lots traded
    # grouped = df.groupby("login")["lots"].sum()
    summary = df.groupby("login").agg({
        "lots": "sum",
        "profit": "sum",
        "comment": "count"  # number of trades
    }).rename(columns={
        "lots": "total_lots",
        "profit": "total_profit",
        "comment": "trade_count"
    })
    # Step 2: Extract business group info
    login_group = df[['login', 'business_group']].drop_duplicates(subset='login')
    # Step 3: Merge business group into summary
    summary = summary.reset_index().merge(login_group, on='login').set_index('login')

    lots_top_30 = summary.sort_values(by="total_lots", ascending=False).head(30)
    profit_top_30 = summary.sort_values(by="total_profit", ascending=False).head(30)
    # Optional: convert to DataFrame for display
    lots_top_30_df = lots_top_30.reset_index()
    profit_top_30_df = profit_top_30.reset_index()
    print('期迹EA交易量前30账户:')
    # print(lots_top_30_df)
    lots_top_30_df.to_csv('lots_top_30_df.csv', index=False)
    print('期迹EA盈利前30账户:')
    # print(profit_top_30_df)
    profit_top_30_df.to_csv('profit_top_30_df.csv', index=False)

    total_lots = df['lots'].sum()
    total_profit = df['profit'].sum()

    # # apply retracement calculation
    # df["retracement"] = df.apply(trade_retracement, axis=1)
    # # group by lots
    # summary = df.groupby("lots").agg(
    #     trade_count=("profit", "count"),
    #     total_profit=("profit", "sum"),
    #     avg_profit=("profit", "mean"),
    #     avg_retracement=("retracement", "mean"),
    #     win_rate=("profit", lambda x: (x > 0).mean())
    # ).reset_index()
    # print(summary)

    print(df)
    print("----------------Total Lots:", total_lots)
    print("----------------Total Profit:", total_profit)
    print("")

    #入金
    columns1 = ['login', 'account_type', 'usd_amount', 'completion_time']
    decimal_cols = ['usd_amount']
    df1 = pd.DataFrame(deposit, columns=columns1)
    df1[decimal_cols] = df1[decimal_cols].astype(float)
    df1['completion_time'] = pd.to_datetime(df1['completion_time'])
    total_deposit = df1['usd_amount'].sum()
    print('入金:')
    print(df1)
    print("----------------Total Deposit:", total_deposit)
    print("")

    #出金
    df2 = pd.DataFrame(withdraw, columns=columns1)
    df2[decimal_cols] = df2[decimal_cols].astype(float)
    df2['completion_time'] = pd.to_datetime(df2['completion_time'])
    total_withdraw = df2['usd_amount'].sum()
    print('出金:')
    print(df2)
    print("----------------Total Withdraw:", total_withdraw)
    print("")

    # # df.to_csv('3677595.csv', index=False)

    pool.close()
    await pool.wait_closed()


# def trade_retracement(row):
#     # fetch minute bars between open and close
#     # rates = mt5.copy_rates_range(
#     #     symbol=row["symbol"],
#     #     timeframe=mt5.TIMEFRAME_M1,
#     #     from_date=row["open_dt"],
#     #     to_date=row["close_dt"]
#     # )
#     rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M1, date_from, date_to)
#
#     if rates is None or len(rates) == 0:
#         return 0.0
#
#     data = pd.DataFrame(rates)
#     if row["cmd"] == 1:  # buy
#         worst_low = data["low"].min()
#         return row["open_price"] - worst_low
#     else:  # sell
#         worst_high = data["high"].max()
#         return worst_high - row["open_price"]


def scheduled_job():
    asyncio.run(run_task())


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None) # Show all rows
    pd.set_option('display.width', 2000)  # Increase display width to prevent line wrapping
    pd.set_option('display.max_colwidth', None) # Prevent column content from being truncated
    scheduled_job()
