"""
Betfair Tipster Statistics App

This Streamlit application allows users to upload a CSV file containing
Betfair tipster data and view a summary of key performance metrics for
each tipster. You can upload a new CSV each week to see the updated
statistics.

Expected CSV format
-------------------
The application expects the CSV to contain at least the following columns
(case insensitive):

* **Tipster** – the name or identifier of the tipster. In your data this
  corresponds to the column ``Tipster_Name``.
* **Bet type** – either ``BACK`` or ``LAY``. In your data this is
  ``BettingStrategy``.
* **Stake** – the stake placed on the bet. This is the ``Stake`` column in
  your CSV.
* **Price** – the odds at which the bet was placed (decimal odds). In your
  CSV this is the ``BSP`` column.
* **Result** – indicator of whether the bet won or lost. In your data
  ``RESULT`` holds ``1`` for a win and ``0`` for a loss.
* **Profit** – net profit per bet, already calculated. In your CSV this
  corresponds to ``PROFIT_BSP``. If a profit column is absent, the
  application will derive profit from stake, price and result as an
  approximation.

If your CSV uses slightly different column names, the application will
attempt to rename them automatically. You can also edit the mapping in
``normalize_columns`` below to suit your data.

Metrics
-------
For each tipster the following metrics are computed:

* **Total Bets** – number of bets placed by the tipster.
* **Total Units Profit** – sum of profits across all bets.
* **Strike Rate** – proportion of winning bets (Result = 1) to total bets.
* **Highest Winning Back Price** – highest BSP (odds) among winning back bets.
* **Lowest Lay Price** – lowest BSP across all lay bets (regardless of
  outcome).
* **Profit on Turnover (POT)** – total profit divided by total turnover
  (sum of stakes).

These metrics give a comprehensive picture of how each tipster performed
over the period covered by the CSV.

Usage
-----
Run the app with:

```
streamlit run betfair_app.py
```

Then upload your CSV file via the file uploader widget. The summary
statistics will be presented in an interactive table.
"""

import pandas as pd
from typing import Any, Dict


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to a standard set used by the application.

    This function attempts to map various possible column names found in
    betting data to a small set of expected names: ``tipster``, ``bet_type``,
    ``stake``, ``price``, ``result`` and ``profit``. If a mapping is not
    found, the original column name is kept.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with arbitrary column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns where possible.
    """
    # Map original lower-case column names to original names for lookup
    col_map: Dict[str, str] = {c.lower(): c for c in df.columns}
    rename_map: Dict[str, str] = {}

    # Map tipster column
    for name in ["tipster", "tipsters", "user", "contributor", "tipster_name"]:
        if name in col_map:
            rename_map[col_map[name]] = "tipster"
            break

    # Map bet type column
    for name in ["bet_type", "bettype", "type", "bettingstrategy"]:
        if name in col_map:
            rename_map[col_map[name]] = "bet_type"
            break

    # Map stake column
    for name in ["stake", "amount", "stake_units", "unit"]:
        if name in col_map:
            rename_map[col_map[name]] = "stake"
            break

    # Map price/odds column (BSP in your CSV)
    for name in ["price", "odds", "back_price", "lay_price", "bsp"]:
        if name in col_map:
            rename_map[col_map[name]] = "price"
            break

    # Map result column (mbr or RESULT indicates win/loss)
    for name in ["result", "outcome", "win", "won", "mbr", "result"]:
        if name.lower() in col_map:
            rename_map[col_map[name.lower()]] = "result"
            break

    # Map profit column (PROFIT_BSP in your CSV)
    for name in ["profit", "net", "pnl", "net_profit", "profit_bsp", "profit_bsp_rp", "profit_bestbet"]:
        if name in col_map:
            # The first match will be used; you can adjust the order if you prefer
            rename_map[col_map[name]] = "profit"
            break

    return df.rename(columns=rename_map)


def compute_profit(row: pd.Series) -> float:
    """Derive profit for a single bet when not explicitly provided.

    If the row already contains a ``profit`` value it will be returned as
    float. Otherwise the profit is approximated based on bet type, stake,
    price and result:

    * ``BACK`` bets: winning profit = (price − 1) × stake; losing profit = −stake
    * ``LAY`` bets: winning profit = stake; losing profit = −(price − 1) × stake

    The ``result`` column is treated as truthy for win (e.g. 1, ``True``,
    ``win``) and falsey for loss (e.g. 0, ``False``, ``loss``).

    Parameters
    ----------
    row : pd.Series
        A row from the dataframe.

    Returns
    -------
    float
        Profit for the bet.
    """
    # If profit provided, use it
    if "profit" in row and pd.notnull(row.get("profit")):
        try:
            return float(row.get("profit"))
        except Exception:
            pass

    # Determine win/loss
    result_val = row.get("result")
    won = False
    if isinstance(result_val, (int, float)):
        won = result_val > 0
    elif isinstance(result_val, str):
        won = result_val.strip().lower() in {"win", "won", "yes", "true", "1"}
    elif isinstance(result_val, bool):
        won = result_val

    # Bet type
    bet_type = str(row.get("bet_type", "")).strip().lower()
    # Stake and price as float
    try:
        stake_val = float(row.get("stake", 0))
    except Exception:
        stake_val = 0.0
    try:
        price_val = float(row.get("price", 0))
    except Exception:
        price_val = 0.0

    if bet_type == "back":
        return (price_val - 1) * stake_val if won else -stake_val
    elif bet_type == "lay":
        return stake_val if won else -(price_val - 1) * stake_val
    else:
        return 0.0


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance metrics for each tipster.

    This function does not depend on Streamlit and raises a ValueError
    if required columns are missing. It normalises column names, computes
    profits (if not supplied), determines winning bets, and aggregates
    statistics by tipster.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataframe containing bet records.

    Returns
    -------
    pd.DataFrame
        Summary statistics per tipster.

    Raises
    ------
    ValueError
        If the dataframe does not contain all required columns.
    """
    # Normalise column names
    df = normalize_columns(df)

    required_cols = {"tipster", "bet_type", "stake", "price", "result"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}. "
            "Update your CSV or adjust the mapping in normalize_columns()."
        )

    # Convert stake and price to numeric (coerce errors to NaN)
    df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Compute profit column if missing or not numeric
    df["profit"] = df.apply(compute_profit, axis=1)

    # Determine win indicator for strike rate
    def is_win(val: Any) -> bool:
        if isinstance(val, (int, float)):
            return val > 0
        if isinstance(val, str):
            return val.strip().lower() in {"win", "won", "yes", "true", "1"}
        if isinstance(val, bool):
            return val
        return False

    df["won"] = df["result"].apply(is_win)

    summary_rows = []
    for tipster, group in df.groupby("tipster"):
        total_bets = len(group)
        total_units_profit = group["profit"].sum()
        strike_rate = group["won"].mean() if total_bets else float("nan")
        # Highest winning back price
        winning_backs = group[(group["bet_type"].str.lower() == "back") & (group["won"])]
        highest_win_back_price = (
            winning_backs["price"].max() if not winning_backs.empty else float("nan")
        )
        # Lowest lay price
        lay_bets = group[group["bet_type"].str.lower() == "lay"]
        lowest_lay_price = (
            lay_bets["price"].min() if not lay_bets.empty else float("nan")
        )
        # Profit on turnover
        total_turnover = group["stake"].sum()
        pot = total_units_profit / total_turnover if total_turnover else float("nan")
        summary_rows.append(
            {
                "Tipster": tipster,
                "Total Bets": total_bets,
                "Total Units Profit": total_units_profit,
                "Strike Rate": strike_rate,
                "Highest Winning Back Price": highest_win_back_price,
                "Lowest Lay Price": lowest_lay_price,
                "Profit on Turnover": pot,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="Total Units Profit", ascending=False).reset_index(drop=True)
    return summary_df


def main() -> None:
    """Run the Streamlit application.

    Importing Streamlit within the function ensures that the rest of the module
    can be imported without requiring Streamlit to be installed. This makes
    testing helper functions easier in environments where Streamlit isn't
    available. When executed as a script (via ``streamlit run``) the
    ``streamlit`` package must be installed.
    """
    try:
        import streamlit as st  # type: ignore
    except ImportError:
        raise ImportError(
            "Streamlit is not installed. Please install it with `pip install streamlit` "
            "to run the web application."
        )

    st.set_page_config(page_title="Betfair Tipster Stats", layout="wide")
    st.title("Betfair Tipster Statistics")
    st.write(
        "Upload a CSV file with Betfair tipster data to see summary statistics "
        "for each contributor. The app calculates metrics such as total bets, "
        "profit, strike rate, highest winning back price, lowest lay price and "
        "profit on turnover."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], help="Upload your tipster results CSV"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            return

        try:
            summary_df = calculate_metrics(df)
        except ValueError as e:
            st.error(str(e))
            return

        if not summary_df.empty:
            display_df = summary_df.copy()
            display_df["Strike Rate"] = display_df["Strike Rate"].apply(
                lambda x: f"{x:.2%}" if pd.notnull(x) else ""
            )
            display_df["Profit on Turnover"] = display_df["Profit on Turnover"].apply(
                lambda x: f"{x:.2%}" if pd.notnull(x) else ""
            )
            st.subheader("Summary by Tipster")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No data available to display.")


if __name__ == "__main__":
    main()