"""Load a project CSV into Supabase customers and customer_snapshots.

Usage:
  SUPABASE_DB_URL=... python jobs/seed_csv.py data/robustness_test/synthetic_banking_stress_test.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import psycopg


DB_URL = os.environ.get("SUPABASE_DB_URL")
if not DB_URL:
    raise SystemExit("SUPABASE_DB_URL is required")
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/robustness_test/synthetic_banking_stress_test.csv")
CHUNK_SIZE = 2_000


def clean(value):
    return None if pd.isna(value) else value


def main() -> None:
    total = 0
    with psycopg.connect(DB_URL) as connection:
        cursor = connection.cursor()
        for frame in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):
            customer_rows = [(str(value),) for value in frame["CustomerId"]]
            cursor.executemany(
                "insert into public.customers (external_customer_id) values (%s) "
                "on conflict (external_customer_id) do update set updated_at = now()",
                customer_rows,
            )
            ids = [str(value) for value in frame["CustomerId"]]
            customer_map = dict(connection.execute(
                "select external_customer_id, customer_id from public.customers where external_customer_id = any(%s)",
                (ids,),
            ).fetchall())
            snapshot_rows = []
            for row in frame.to_dict("records"):
                snapshot_rows.append((
                    customer_map[str(row["CustomerId"])],
                    clean(row.get("Gender")), clean(row.get("Marital Status")),
                    clean(row.get("Number of Dependents")), clean(row.get("Occupation")),
                    clean(row.get("Income")), clean(row.get("Education Level")),
                    clean(row.get("Customer Tenure")), clean(row.get("Customer Segment")),
                    clean(row.get("Preferred Communication Channel")), clean(row.get("Credit Score")),
                    clean(row.get("Credit History Length")), clean(row.get("Outstanding Loans")),
                    clean(row.get("Balance")), clean(row.get("NumOfProducts")), clean(row.get("NumComplaints")),
                ))
            cursor.executemany(
                "insert into public.customer_snapshots "
                "(customer_id, gender, marital_status, number_of_dependents, occupation, income, education_level, "
                "customer_tenure, customer_segment, preferred_communication_channel, credit_score, "
                "credit_history_length, outstanding_loans, balance, num_of_products, num_complaints) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                snapshot_rows,
            )
            total += len(frame)
            print(f"loaded={total}", flush=True)
        connection.commit()
        print("customers=" + str(connection.execute("select count(*) from public.customers").fetchone()[0]))
        print("snapshots=" + str(connection.execute("select count(*) from public.customer_snapshots").fetchone()[0]))


if __name__ == "__main__":
    main()
