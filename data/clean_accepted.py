"""
Preprocessing pipeline for LendingClub accepted loans dataset.

================================================================================
NOTE ON DERIVED FEATURE SELECTION
================================================================================
Tree-based models (XGBoost, LightGBM, Random Forest) automatically discover
thresholds and pairwise interactions on raw features, so most "engineered"
features add noise rather than signal. The 6 derived features below are kept
because each one provides information the model CANNOT trivially recover from
the raw columns by itself.

Selection criteria:
  (a) The ratio normalizes by a per-borrower scale (income, credit history
      size, etc.), which trees do not compute internally.
  (b) The feature encodes a domain-known risk signal that warrants giving the
      model a direct split point.
  (c) The transformation involves arithmetic the tree cannot perform on raw
      inputs (e.g. date subtraction).

Features kept and why:

  1. loan_to_income = loan_amnt / annual_inc
     Total debt burden relative to earning capacity. Captures affordability
     across borrowers of very different income levels — a $20k loan means
     something very different to someone earning $40k vs $200k.

  2. revolbal_to_income = revol_bal / annual_inc
     Existing revolving (credit card) debt stress, separate from the new loan.
     A high value means the borrower is already leveraged before taking on
     this installment loan.

  3. inquiries_per_account = inq_last_6mths / total_acc
     Normalizes recent credit-shopping intensity by credit history size.
     Three inquiries means very different things for someone with 5 accounts
     vs. 50 accounts; the raw count alone obscures this.

  4. total_interest_burden = int_rate * (term/ 12)
     Rough total interest cost as % of principal over the life of the loan.
     High values flag subprime pricing, which captures risk information that
     LendingClub's underwriting saw even when raw FICO does not reflect it.
     Tree models can find this interaction via depth-2 splits, but exposing
     it as a single feature helps shallow ensembles.

  5. credit_history_years = (issue_d - earliest_cr_line) / 365.25
     Length of credit history at the time the loan was issued. Requires date
     arithmetic on two columns; trees cannot derive this from the raw dates
     on their own. Longer history correlates with lower default risk.


Other things to watch out for:
  - All ratios use (denominator + 1) smoothing to avoid division by zero when
    annual_inc, total_acc, or loan_amnt edge cases produce zero/null.
  - Date parsing assumes the LendingClub format "Mon-YYYY" (e.g. "Dec-2015").
    If the parquet has these as already-parsed dates, the cast is a no-op.
  - `term` is parsed from "36 months" / "60 months" strings into integer
    `term_months` (needed for total_interest_burden).
  - Do NOT include the original `loan_status` as a model input — only `target`.
================================================================================
"""

import os
import polars as pl

RAW_PARQUET = "./raw_data/accepted_2007_to_2018Q4.parquet"
OUTPUT_PARQUET = "./processed_data/processed_accepted_df.parquet"
OUTPUT_CSV = "./processed_data/processed_accepted_df.csv"

FEATURES_FINAL = [
    # 1. Loan Characteristics (The Application)
    'loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'purpose',
    # 2. Borrower Demographics & Financials (The Person)
    'annual_inc', 'emp_length', 'home_ownership', 'dti',
    # 3. Credit History & Utilization (The Track Record)
    'fico_range_low', 'earliest_cr_line', 'open_acc', 'total_acc',
    'mort_acc', 'tot_cur_bal', 'revol_bal', 'revol_util',
    'bc_util', 'percent_bc_gt_75', 'inq_last_6mths',
    # 4. Red Flags
    'delinq_2yrs', 'acc_now_delinq', 'tax_liens', 'pub_rec',
    # 5. Metadata & Target Variables
    'issue_d', 'loan_status'
]

def csv_to_parquet(path):
    os.makedirs("raw_data", exist_ok=True)
    filename = os.path.basename(path)
    parquet_filename = filename.replace(".csv", ".parquet")
    parquet_path = os.path.join(".\\raw_data", parquet_filename)
    # lazy scan (does NOT load everything into RAM)
    lf = pl.scan_csv(
        path,
        infer_schema_length=1000,
        ignore_errors=True
    )
    print(parquet_path)
    # stream directly to parquet
    lf.sink_parquet(
        parquet_path,
        compression="snappy"
    )
    print(f"Saved Parquet file to: {parquet_path}")


def parse_lc_date(col_name: str) -> pl.Expr:
    """Parse LendingClub date strings of the form 'Dec-2015' into pl.Date.

    Safe if the column is already a date type — the Utf8 cast is non-strict
    and returns null on incompatible types, which downstream code handles.
    """
    return (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .str.to_date(format="%b-%Y", strict=False)
    )

def main():
    os.makedirs("./processed_data", exist_ok=True)

    # Verify if accepted polar df exists or not
    has_parquet = any(f.endswith(".parquet") for f in os.listdir("../data/raw_data"))
    if has_parquet:
        print("Parquet file already exists in data folder")
    else:
        print("No parquet file found. Converting CSV → Parquet...")
        csv_to_parquet("../data/raw_data/accepted_2007_to_2018Q4.csv")
    
    # Read accepted_df file
    df = pl.read_parquet(RAW_PARQUET)

    clean_df = df.select(FEATURES_FINAL)
    # Keep only known outcomes
    clean_df = clean_df.filter(
            pl.col("loan_status").is_in(["Fully Paid", "Charged Off"])
    )
    # Create binary target
    clean_df = clean_df.with_columns(
            pl.when(pl.col("loan_status") == "Charged Off")
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("target")
    )

    # ---- Clean raw columns ----
    # Parse date strings into actual Date type for arithmetic
    clean_df = clean_df.with_columns([
        parse_lc_date("issue_d").alias("issue_d_parsed"),
        parse_lc_date("earliest_cr_line").alias("earliest_cr_line_parsed"),
    ])

    clean_df = clean_df.with_columns(
        pl.col("sub_grade")
        .str.extract(r"(\d+)")
        .cast(pl.Int8)
    )

    clean_df = clean_df.with_columns(
        pl.col("emp_length")
        .str.replace("10\\+ years", "10 years")
        .str.replace("< 1 year", "0 years")
        .str.extract(r"(\d+)")
        .cast(pl.Int8)
        .alias("emp_length")
    )

    # term: "36 months"  -> int months (needed for interest burden)
    clean_df = clean_df.with_columns(
        pl.col("term").cast(pl.Utf8).str.extract(r"(\d+)").cast(pl.Int16)
    )

    clean_df = clean_df.with_columns(
        pl.col("home_ownership").replace({"NONE": "OTHER", "ANY": "OTHER"})
    )

    # ---- Impute missing data ----
    """
    the median is susceptible to extreme outliers and
    provides a more accurate and robust measure of central tendency.
    """
    clean_df = clean_df.with_columns([
        pl.col("emp_length").fill_null(pl.col("emp_length").median()),
        pl.col("percent_bc_gt_75").fill_null(pl.col("percent_bc_gt_75").median()),
        pl.col("mort_acc").fill_null(pl.col("mort_acc").median()),
        pl.col("dti").fill_null(pl.col("dti").median()),
        pl.col("bc_util").fill_null(pl.col("bc_util").median()),
        pl.col("revol_util").fill_null(pl.col("revol_util").median()),
        pl.col("tot_cur_bal").fill_null(pl.col("tot_cur_bal").median()),
        pl.col("pub_rec").fill_null(0),
        pl.col("inq_last_6mths").fill_null(0),
        pl.col("tax_liens").fill_null(0),
        pl.col("acc_now_delinq").fill_null(0)
    ])


    # ---- Derived columns ----
    # Income / debt-burden ratios
    clean_df = clean_df.with_columns([
        (pl.col("loan_amnt") / (pl.col("annual_inc") + 1))
            .alias("loan_to_income"),
        (pl.col("revol_bal") / (pl.col("annual_inc") + 1))
            .alias("revolbal_to_income"),
        (pl.col("loan_amnt") * pl.col("int_rate") / (12 * pl.col("annual_inc") + 1))
            .alias("payment_to_income"),
    ]) 

    # Credit behavior 
    clean_df = clean_df.with_columns([
        (pl.col("inq_last_6mths") / (pl.col("total_acc") + 1))
            .alias("inquiries_per_account"),
        (pl.col("revol_util") * pl.col("dti")).alias("revol_stress"),
        (pl.col("open_acc") / (pl.col("total_acc") + 1)).alias("active_accounts_ratio")
    ])

    # Loan-structure
    clean_df = clean_df.with_columns([
        (pl.col("int_rate") * (pl.col("term") / 12.0))
            .alias("total_interest_burden"),
    ])

    # Credit history length 
    clean_df = clean_df.with_columns([
        ((pl.col("issue_d_parsed") - pl.col("earliest_cr_line_parsed"))
            .dt.total_days() / 365.25).alias("credit_history_years")
        ])
    clean_df = clean_df.with_columns([
        pl.col("credit_history_years").fill_null(pl.col("credit_history_years").median()),
    ])
    
    # Red flags occurrences
    clean_df = clean_df.with_columns([
        (
            (pl.col("pub_rec") > 0).cast(pl.Int32) +
            (pl.col("tax_liens") > 0).cast(pl.Int32) +
            (pl.col('delinq_2yrs') > 0).cast(pl.Int32) +
            (pl.col('acc_now_delinq') > 0).cast(pl.Int32))
            .alias("total_red_flags")
    ])

    # Drop raw date columns
    clean_df = clean_df.drop(["issue_d", "earliest_cr_line"])
    print(f"Final shape: {clean_df.shape}")
    print(f"Columns ({len(clean_df.columns)}): {clean_df.columns}")

    # ---- Null sanity check ----
    null_summary = (
        clean_df.null_count()
        .transpose(
            include_header=True,
            header_name="column",
            column_names=["null_count"]
        )
        .filter(pl.col("null_count") > 0)
    )

    if null_summary.height:
        print("\\nRemaining nulls:")
        print(null_summary)

    # Save polar data
    clean_df.write_parquet(OUTPUT_PARQUET)
    print(f"Saved to: {OUTPUT_PARQUET}")

    # Save csv data
    clean_df.write_csv(OUTPUT_CSV)
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
