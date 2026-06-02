import os
import polars as pl

RAW_PARQUET = "./raw_data/accepted_2007_to_2018Q4.parquet"
OUTPUT_PARQUET = "./processed_data/processed_accepted_df.parquet"
OUTPUT_CSV = "./processed_data/processed_accepted_df.csv"

FEATURES_FINAL = [
    'loan_amnt', 'term', 'int_rate',
    'grade', 'sub_grade', 'purpose',
    'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
    'open_acc', 'total_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'emp_length', 'home_ownership',
    'earliest_cr_line', 'fico_range_low',
    'last_pymnt_amnt', 'tot_cur_bal',
    'verification_status', 'pub_rec_bankruptcies', 'tax_liens',
    'acc_now_delinq', 'mort_acc', 'bc_util', 'percent_bc_gt_75',
    'issue_d', 'loan_status'
]

def main():
    os.makedirs("./processed_data", exist_ok=True)

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

    # Clean data
    clean_df = clean_df.with_columns(
        pl.col("home_ownership").replace({"NONE": "OTHER", "ANY": "OTHER"})
    )

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

    # Impute missing data
    clean_df = clean_df.with_columns([
        pl.col("emp_length").fill_null(pl.col("emp_length").median()),
        pl.col("percent_bc_gt_75").fill_null(pl.col("percent_bc_gt_75").mean()),
        pl.col("mort_acc").fill_null(pl.col("mort_acc").median()),
        pl.col("dti").fill_null(pl.col("dti").mean()),
        pl.col("bc_util").fill_null(pl.col("bc_util").mean()),
        pl.col("pub_rec_bankruptcies").fill_null(0),
        pl.col("inq_last_6mths").fill_null(0),
    ])

    # Derived columns
    clean_df = clean_df.with_columns([
        (pl.col("loan_amnt") / (pl.col("annual_inc") + 1)).alias("loan_to_income"),
        (pl.col("open_acc") / (pl.col("total_acc") + 1)).alias("active_accounts_ratio"),
        (pl.col("inq_last_6mths") / (pl.col("total_acc") + 1)).alias("inquiries_per_account"),
        (pl.col("revol_bal") / (pl.col("annual_inc") + 1)).alias("revolbal_to_income"),
    ])

    print(f"Final shape: {clean_df.shape}")

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
    df.write_csv(OUTPUT_CSV)
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
