from sqlalchemy import create_engine, text, bindparam
import pandas as pd

batch_size = 50

if __name__ == "__main__":
    url = "mysql+pymysql://kpg_frank.w:XH5mJ!MEHi#QFJN3@52.79.211.19:23808/dbtahoe"

    engine = create_engine(url, future=True)

    with engine.connect() as conn:
        # 1. Get account ids in groups ZD or HD
        accounts_stmt = text("""
            SELECT login
            FROM dbtahoe.mt4_users
            WHERE `enable` = 1 and `group` LIKE :g1
        """)
        account_rows = conn.execute(accounts_stmt, {"g1": "%HD"}).mappings().all()
        login_ids = [row["login"] for row in account_rows]

        if not login_ids:
            print("No logins found in groups ZD or HD")
        else:
            # 2. Query mt4_trades in dbtahoe for those accounts
            # Use a parameter list safely with SQLAlchemy's expanding parameter
            trades_stmt = text("""            
                WITH totals AS (
                  SELECT
                    t.LOGIN,
                    COALESCE(SUM(t.VOLUME),0) / 100.0 AS total_lot,
                    COALESCE(SUM(t.PROFIT),0)      AS total_profit,
                    COALESCE(SUM(t.SWAPS),0)       AS total_swaps,
                    COALESCE(SUM(t.PROFIT),0) + COALESCE(SUM(t.SWAPS),0) AS net_profit
                  FROM dbtahoe.mt4_trades t
                  WHERE t.LOGIN IN :logins
                    AND t.CMD <> 6
                  GROUP BY t.LOGIN
                )
                SELECT
                  tot.LOGIN,
                  tot.total_lot,
                  tot.total_profit,
                  tot.total_swaps,
                  tot.net_profit,
                  COALESCE(w.week_net_profit, 0) AS recent_week_net_profit
                FROM totals tot
                LEFT JOIN (
                  -- recent week aggregation using a fixed week range (same for all logins)
                  SELECT
                    LOGIN,
                    SUM(PROFIT) AS week_net_profit
                  FROM dbtahoe.mt4_trades
                  WHERE LOGIN IN :logins
                    AND CMD <> 6
                    AND CLOSE_TIME BETWEEN '2025-10-20 00:00:00' AND '2025-10-20 23:59:59'  -- <- use your week bounds (parameters)
                  GROUP BY LOGIN
                ) w
                  ON w.LOGIN = tot.LOGIN
                ORDER BY tot.LOGIN;
            """).bindparams(bindparam("logins", expanding=True))

            rows = []
            for i in range(0, len(login_ids), batch_size):
                batch = login_ids[i: i + batch_size]
                batch_rows = conn.execute(trades_stmt, {"logins": batch}).mappings().all()
                rows.extend(batch_rows)

            df = pd.DataFrame([dict(r) for r in rows])
            df.to_excel("C:\\Users\\custom_folders\\HD.xlsx")

            # ensure every requested login appears, fill missing with zeros
            requested = set(int(x) for x in login_ids)
            found = set(int(x) for x in df['login'].tolist()) if not df.empty else set()
            missing = requested - found
            if missing:
                missing_df = pd.DataFrame([{
                    'login': m, 'total_profit': 0.0, 'total_swaps': 0.0, 'net_profit': 0.0
                } for m in missing])
                df = pd.concat([df, missing_df], ignore_index=True)

            df = df.astype({'login': int, 'total_profit': float, 'total_swaps': float, 'net_profit': float})
            df = df.sort_values('login').reset_index(drop=True)

            # print(f"Total trades found: {len(trades)}")
