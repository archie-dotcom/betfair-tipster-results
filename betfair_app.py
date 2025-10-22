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
from typing import Any, Dict, List
import os
import glob
import requests
import tempfile


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

    # Map date column (MeetingDate in your CSV)
    for name in ["date", "meetingdate", "meeting_date", "date_time"]:
        if name in col_map:
            rename_map[col_map[name]] = "date"
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

    The page layout and style are tuned to loosely resemble the Betfair app
    aesthetic shown in the reference images: a dark background with warm
    accent colours, rounded cards, and coloured tags for back/lay data. A small
    amount of custom CSS is injected via ``st.markdown`` to apply global
    styles (colours, spacing, table formatting, card layout). Each section
    (summary table, top‑five view, tipster detail) is rendered inside a
    container with a dark background and subtle box shadow.
    """
    try:
        import streamlit as st  # type: ignore
        import altair as alt  # type: ignore
    except ImportError:
        raise ImportError(
            "Streamlit is not installed. Please install it with `pip install streamlit` "
            "to run the web application."
        )

    st.set_page_config(page_title="Betfair Tipster Stats", layout="wide")
    # Inject custom CSS for dark theme and card/table styling.  These rules
    # emulate the dark aesthetic of the Betfair app: dark panels, warm
    # highlights, coloured tags and rounded corners.  The table is
    # rendered manually with HTML so that we can apply our own classes.
    st.markdown(
        """
        <style>
        /* Global background and text colours */
        .stApp {
            background-color: #111111;
            color: #F5F5F5;
        }
        /* Container around content to add padding */
        .main .block-container {
            padding: 1rem 2rem;
        }
        /* Generic card styling */
        .card {
            background-color: #1b1b1b;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .card h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            color: #EBA808;
        }
        /* Accent line for cards */
        .accent-line {
            height: 3px;
            background-color: #EBA808;
            border-radius: 3px;
            margin-top: -0.5rem;
            margin-bottom: 0.5rem;
        }
        /* Table styling */
        .bet-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        .bet-table th, .bet-table td {
            border: 1px solid #333333;
            padding: 0.5rem;
        }
        .bet-table th {
            background-color: #222222;
            color: #EBA808;
            font-weight: bold;
        }
        .bet-table tr:nth-child(even) {
            background-color: #2a2a2a;
        }
        .bet-table tr:nth-child(odd) {
            background-color: #1e1e1e;
        }
        /* Metric tags similar to Back/Lay labels */
        .bet-tag {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.5rem;
            color: #ffffff;
        }
        .bet-tag.back {
            background-color: #224CA8;
        }
        .bet-tag.lay {
            background-color: #E06EB7;
        }
        /* Metric container styling */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
        }
        .metric-box {
            background-color: #1b1b1b;
            border-radius: 8px;
            padding: 0.8rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .metric-title {
            font-size: 0.8rem;
            color: #AAAAAA;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: bold;
            color: #F5F5F5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Betfair Tipster Statistics")
    st.write(
        "Upload weekly CSV files containing tipster data and view performance metrics."
    )

    # Determine where to load CSV data from.  By default the app reads all
    # CSV files in the local ``data`` directory.  Alternatively, you can
    # specify one or more Google Drive file IDs via the environment variable
    # ``BETFAIR_DRIVE_FILE_IDS``.  When set, the app will download each
    # referenced file from Google Drive before stacking them into a single
    # DataFrame.  If both ``BETFAIR_DATA_DIR`` and ``BETFAIR_DRIVE_FILE_IDS``
    # are unset, the app falls back to the ``data`` folder.
    DATA_DIR = os.environ.get("BETFAIR_DATA_DIR", "data")
    DRIVE_FILE_IDS: List[str] = []
    file_ids_env = os.environ.get("BETFAIR_DRIVE_FILE_IDS", "").strip()
    if file_ids_env:
        # Accept comma-separated list of file IDs or URLs.  If a full URL is
        # provided, extract the file ID from it.
        for token in file_ids_env.split(","):
            token = token.strip()
            if not token:
                continue
            # If token looks like a full Google Drive URL, attempt to parse the ID
            if "/d/" in token:
                try:
                    # URL pattern: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
                    parts = token.split("/d/")
                    file_id = parts[1].split("/")[0]
                except Exception:
                    file_id = token
            else:
                file_id = token
            DRIVE_FILE_IDS.append(file_id)

    def download_drive_file(file_id: str) -> str:
        """Download a Google Drive file to a temporary location.

        This helper uses the public download endpoint
        ``https://docs.google.com/uc?export=download`` to fetch the file.  For
        larger files Google prompts with a confirmation token which must be
        included in a subsequent request.  The function handles this by
        checking for a confirmation cookie in the initial response.

        Parameters
        ----------
        file_id : str
            The ID of the Google Drive file to download.

        Returns
        -------
        str
            Path to the downloaded file on the local filesystem.
        """
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={"id": file_id}, stream=True)
        # Check for confirmation token (used when file is large or not yet confirmed)
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
        if token:
            response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
        # Save content to a temporary file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    tmp_file.write(chunk)
        return tmp_path

    def load_data() -> pd.DataFrame:
        data_frames: List[pd.DataFrame] = []
        # If file IDs provided, download each from Google Drive
        if DRIVE_FILE_IDS:
            for fid in DRIVE_FILE_IDS:
                try:
                    file_path = download_drive_file(fid)
                    df_part = pd.read_csv(file_path)
                    df_part["_source"] = f"drive:{fid}"
                    data_frames.append(df_part)
                except Exception as exc:
                    st.warning(f"Failed to download or read Google Drive file {fid}: {exc}")
        # Also load any local CSVs from the directory
        pattern = os.path.join(DATA_DIR, "*.csv")
        for file_path in sorted(glob.glob(pattern)):
            try:
                df_part = pd.read_csv(file_path)
                df_part["_source"] = os.path.basename(file_path)
                data_frames.append(df_part)
            except Exception as exc:
                st.warning(f"Failed to read {file_path}: {exc}")
        if data_frames:
            df_all = pd.concat(data_frames, ignore_index=True)
            return df_all
        else:
            return pd.DataFrame()

    df_raw = load_data()
    if df_raw.empty:
        st.info(
            "No data found. Please upload your weekly CSVs into the 'data' folder, "
            "set the BETFAIR_DATA_DIR environment variable to a folder containing CSVs, "
            "or provide Google Drive file IDs via the BETFAIR_DRIVE_FILE_IDS environment variable."
        )
        return

    # Compute summary metrics
    try:
        summary_df = calculate_metrics(df_raw)
    except ValueError as e:
        st.error(str(e))
        return

    # Format the summary for display
    display_df = summary_df.copy()
    display_df["Strike Rate"] = display_df["Strike Rate"].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else ""
    )
    display_df["Profit on Turnover"] = display_df["Profit on Turnover"].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else ""
    )

    # User interface: Tabs for different views.  We wrap each tab’s content
    # in a card div to provide padding and a dark background.
    tab1, tab2, tab3 = st.tabs(["All Stats", "Top 5", "Tipster View"])

    # 1. All stats: full summary table
    with tab1:
        st.markdown(
            "<div class='card'><h3>Summary by Tipster</h3><div class='accent-line'></div>",
            unsafe_allow_html=True,
        )
        # Convert display_df to HTML so that we can apply our custom table classes
        html_table = display_df.to_html(classes="bet-table", index=False, escape=False)
        st.markdown(html_table, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. Top 5 comparison
    with tab2:
        st.markdown(
            "<div class='card'><h3>Top 5 Tipsters</h3><div class='accent-line'></div>",
            unsafe_allow_html=True,
        )
        metric = st.selectbox(
            "Select metric for ranking",
            options=["Total Units Profit", "Strike Rate", "Profit on Turnover"],
            index=0,
            key="top_metric",
        )
        # Sort by selected metric (numeric sorting). Remove rows with NaN.
        temp = summary_df.dropna(subset=[metric]).sort_values(by=metric, ascending=False).head(5)
        # Prepare data for chart. For percentages multiply by 100
        chart_df = temp[["Tipster", metric]].copy()
        # Choose colour based on metric type
        metric_color_map = {
            "Total Units Profit": "#EBA808",
            "Strike Rate": "#224CA8",
            "Profit on Turnover": "#E06EB7",
        }
        bar_color = metric_color_map.get(metric, "#EBA808")
        # Create bar chart using Altair
        bar = (
            alt.Chart(chart_df)
            .mark_bar(color=bar_color)
            .encode(
                x=alt.X("Tipster:N", sort=-chart_df[metric].values),
                y=alt.Y(f"{metric}:Q", title=metric),
                tooltip=["Tipster", metric],
            )
            .properties(height=350)
        )
        st.altair_chart(bar, use_container_width=True)
        # Display table below the chart with formatted percentages if applicable
        temp_disp = temp.copy()
        if metric == "Strike Rate":
            temp_disp[metric] = (temp_disp[metric] * 100).round(2).astype(str) + "%"
        elif metric == "Profit on Turnover":
            temp_disp[metric] = (temp_disp[metric] * 100).round(2).astype(str) + "%"
        # Render mini-table
        html_table2 = temp_disp[["Tipster", metric]].to_html(classes="bet-table", index=False, escape=False)
        st.markdown(html_table2, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Tipster-specific view
    with tab3:
        st.markdown(
            "<div class='card'><h3>Tipster Detail</h3><div class='accent-line'></div>",
            unsafe_allow_html=True,
        )
        tipster_names = summary_df["Tipster"].tolist()
        selected_tipster = st.selectbox("Select a tipster", options=tipster_names, key="tipster_select")
        # Filter original data for the selected tipster
        tip_df = df_raw.copy()
        tip_df = normalize_columns(tip_df)
        tip_df = tip_df[tip_df["tipster"] == selected_tipster]
        # Compute metrics for this tipster
        try:
            tip_summary = calculate_metrics(tip_df)
        except ValueError:
            tip_summary = pd.DataFrame()
        # Display metrics in custom metric boxes
        if not tip_summary.empty:
            row = tip_summary.iloc[0]
            # Build HTML for metrics grid
            metrics_html = "<div class='metric-grid'>"
            metrics = [
                ("Total Bets", f"{int(row['Total Bets'])}"),
                ("Units Profit", f"{row['Total Units Profit']:.2f}"),
                ("Strike Rate", f"{row['Strike Rate']:.2%}"),
                ("POT", f"{row['Profit on Turnover']:.2%}"),
                ("Highest Win Back", f"{row['Highest Winning Back Price']:.2f}" if pd.notnull(row['Highest Winning Back Price']) else "N/A"),
                ("Lowest Lay", f"{row['Lowest Lay Price']:.2f}" if pd.notnull(row['Lowest Lay Price']) else "N/A"),
            ]
            for title, value in metrics:
                metrics_html += f"<div class='metric-box'><div class='metric-title'>{title}</div><div class='metric-value'>{value}</div></div>"
            metrics_html += "</div>"
            st.markdown(metrics_html, unsafe_allow_html=True)
        # Line graph of cumulative units over time
        # Prepare data with date and profit
        if "date" not in tip_df.columns:
            # Try to parse MeetingDate if present
            if "MeetingDate" in tip_df.columns:
                tip_df["date"] = pd.to_datetime(tip_df["MeetingDate"], dayfirst=True, errors="coerce")
        # Ensure date column exists and is datetime
        if "date" in tip_df.columns:
            tip_df["date"] = pd.to_datetime(tip_df["date"], errors="coerce")
            tip_df = tip_df.dropna(subset=["date"])
            tip_df = tip_df.sort_values("date")
            # Compute cumulative units profit
            tip_df["profit"] = tip_df.apply(compute_profit, axis=1)
            tip_df["cumulative_profit"] = tip_df["profit"].cumsum()
            line = (
                alt.Chart(tip_df)
                .mark_line(color="#EBA808")
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("cumulative_profit:Q", title="Cumulative Units Profit"),
                    tooltip=["date:T", "cumulative_profit:Q"],
                )
                .properties(height=350)
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.info("No date information available to plot cumulative profit.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()