# Run:
#   pip install streamlit pandas numpy plotly
#   streamlit run main.py

from __future__ import annotations

from pathlib import Path
from datetime import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import textwrap


# Styling
BG = "#070A12"
PANEL = "rgba(255,255,255,0.04)"
PANEL_BORDER = "rgba(255,255,255,0.08)"
GRID = "rgba(255,255,255,0.06)"
TXT = "rgba(245,247,255,0.92)"
MUTED = "rgba(245,247,255,0.72)"

ACCENT = "#7AA2FF"
ACCENT_SOFT = "rgba(122,162,255,0.25)"
GREEN = "#2EE59D"
GREEN_SOFT = "rgba(46,229,157,0.25)"
RED = "#FF5C8A"
RED_SOFT = "rgba(255,92,138,0.22)"
AMBER = "#FFD166"


def apply_glass_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial", size=12, color=TXT),
        margin=dict(l=12, r=12, t=54 if title else 18, b=12),
        legend=dict(
            orientation="h",
            y=1.02, x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(10,12,20,0.85)",
            bordercolor="rgba(255,255,255,0.12)",
            font=dict(color=TXT),
        ),
    )
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                x=0.01, xanchor="left",
                y=0.98, yanchor="top",
                font=dict(size=16, color=TXT),
            )
        )

    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
        ticks="outside", tickcolor="rgba(255,255,255,0.10)", tickfont=dict(color=MUTED),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
        ticks="outside", tickcolor="rgba(255,255,255,0.10)", tickfont=dict(color=MUTED),
    )
    return fig


def apply_cinematic_dark_pro(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#05070F",
        plot_bgcolor="#05070F",
        font=dict(family="Inter, Segoe UI, Arial", size=13, color=TXT),
        margin=dict(l=18, r=18, t=74, b=18),
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=18, color=TXT)),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(10,12,20,0.92)",
            bordercolor="rgba(255,255,255,0.14)",
            font=dict(color=TXT),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(color="rgba(245,247,255,0.70)"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        showline=False,
        tickfont=dict(color="rgba(245,247,255,0.70)"),
    )
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="rgba(255,255,255,0.10)", width=1),
        fillcolor="rgba(0,0,0,0)",
        layer="below",
    )
    return fig

# One-click data loading
DATA_DIR = Path(__file__).parent / "data"

TIMEFRAME_FILES = {
    "5m": "NIFTY 50_5minute.csv",
    "15m": "NIFTY 50_15minute.csv",
    "30m": "NIFTY 50_30minute.csv",
}

FORCED_EXIT_BY_TF = {
    "5m": time(15, 10),
    "15m": time(15, 15),
    "30m": time(15, 0),
}

START_T = time(9, 15)
LAST_ENTRY_T = time(14, 45)

def available_timeframes() -> list[str]:
    tfs = []
    for tf, fname in TIMEFRAME_FILES.items():
        if (DATA_DIR / fname).exists():
            tfs.append(tf)
    return tfs

@st.cache_data(show_spinner=False)
def load_csv_from_data(tf: str) -> pd.DataFrame:
    path = DATA_DIR / TIMEFRAME_FILES[tf]
    return pd.read_csv(path)

# Strategy core

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def apply_slippage(entry_price: float, exit_price: float, side: int, slip_pts: float):
    entry_adj = entry_price + side * slip_pts
    exit_adj = exit_price - side * slip_pts
    return entry_adj, exit_adj

def _parse_datetime_column_fast(df: pd.DataFrame) -> pd.Series:
    s = df["date"].astype(str).str.strip()
    mask = s.str.match(r"^\d{4}-\d{2}-\d{2}\s+\d{6}$")
    if mask.any():
        hh = s[mask].str.slice(-6, -4)
        mm = s[mask].str.slice(-4, -2)
        ss = s[mask].str.slice(-2, None)
        s.loc[mask] = s[mask].str.slice(0, 10) + " " + hh + ":" + mm + ":" + ss
    return pd.to_datetime(s, errors="coerce")

def prepare_df(df: pd.DataFrame, ema_len: int) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = _parse_datetime_column_fast(df)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    df["emaH"] = ema(df["high"], ema_len)
    df["emaL"] = ema(df["low"], ema_len)
    df["green"] = df["close"] > df["open"]
    df["red"] = df["close"] < df["open"]
    df["t"] = df["date"].dt.time
    df["d"] = df["date"].dt.date
    df["year"] = df["date"].dt.year
    return df

def generate_signals(df: pd.DataFrame, start_t: time, last_entry_t: time) -> tuple[np.ndarray, np.ndarray]:
    in_entry = (df["t"] >= start_t) & (df["t"] <= last_entry_t)

    pierce_long = (df["low"].shift(1) < df["emaH"].shift(1)) & (df["close"].shift(1) > df["emaH"].shift(1))
    pierce_short = (df["high"].shift(1) > df["emaL"].shift(1)) & (df["close"].shift(1) < df["emaL"].shift(1))

    long_sig = (
            in_entry
            & (df["close"].shift(2) < df["emaH"].shift(2))
            & df["green"].shift(1).fillna(False)
            & pierce_long
            & df["green"]
            & (df["high"] > df["high"].shift(1))
            & (df["close"] > df["emaH"])
    ).to_numpy()

    short_sig = (
            in_entry
            & (df["close"].shift(2) > df["emaL"].shift(2))
            & df["red"].shift(1).fillna(False)
            & pierce_short
            & df["red"]
            & (df["low"] < df["low"].shift(1))
            & (df["close"] < df["emaL"])
    ).to_numpy()

    return long_sig, short_sig

def run_backtest(
        df_raw: pd.DataFrame,
        ema_len: int,
        rr_mult: float,
        start_t: time,
        last_entry_t: time,
        forced_exit_bar_time: time,
        lot_size: int,
        n_contracts: int,
        slip_pts_side: float,
        fee_rt_total: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = prepare_df(df_raw, ema_len)
    long_sig, short_sig = generate_signals(df, start_t, last_entry_t)

    dates = df["date"].to_numpy()
    times = df["t"].to_numpy()

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    emaH = df["emaH"].to_numpy()
    emaL = df["emaL"].to_numpy()

    trades = []
    position = 0
    entry_i = -1
    entry_p = np.nan
    signal_i = -1
    target = np.nan

    n = len(df)
    for i in range(3, n - 2):
        # forced exit at bar close (time check)
        if position != 0 and times[i] == forced_exit_bar_time:
            trades.append((dates[entry_i], dates[i], position, float(entry_p), float(c[i]), "time_exit_close"))
            position = 0
            continue

        if position == 0:
            if np.isnan(emaH[i]) or np.isnan(emaL[i]):
                continue

            if long_sig[i]:
                entry_i = i + 1
                entry_p = o[entry_i]
                signal_i = i
                risk = entry_p - emaL[signal_i]
                target = entry_p + rr_mult * risk
                position = 1

            elif short_sig[i]:
                entry_i = i + 1
                entry_p = o[entry_i]
                signal_i = i
                risk = emaH[signal_i] - entry_p
                target = entry_p - rr_mult * risk
                position = -1

        else:
            if position == 1 and h[i] >= target:
                trades.append((dates[entry_i], dates[i], position, float(entry_p), float(target), "target"))
                position = 0
                continue
            if position == -1 and l[i] <= target:
                trades.append((dates[entry_i], dates[i], position, float(entry_p), float(target), "target"))
                position = 0
                continue

            # 2 close stop
            if i >= entry_i + 2:
                if position == 1 and (c[i - 1] < emaL[i - 1]) and (c[i] < emaL[i]):
                    trades.append((dates[entry_i], dates[i], position, float(entry_p), float(c[i]), "stop_2cl_close"))
                    position = 0
                    continue
                if position == -1 and (c[i - 1] > emaH[i - 1]) and (c[i] > emaH[i]):
                    trades.append((dates[entry_i], dates[i], position, float(entry_p), float(c[i]), "stop_2cl_close"))
                    position = 0
                    continue

    tr = pd.DataFrame(trades, columns=["entry_time", "exit_time", "side", "entry_price", "exit_price", "exit_reason"])
    if len(tr) == 0:
        return df, tr, pd.DataFrame(), pd.DataFrame()

    mult = lot_size * n_contracts

    entry_adj, exit_adj = zip(*tr.apply(
        lambda r: apply_slippage(r["entry_price"], r["exit_price"], int(r["side"]), slip_pts_side),
        axis=1
    ))
    tr["entry_price_adj"] = np.array(entry_adj, dtype=float)
    tr["exit_price_adj"] = np.array(exit_adj, dtype=float)

    tr["gross_points"] = tr["side"] * (tr["exit_price"] - tr["entry_price"])
    tr["net_points"] = tr["side"] * (tr["exit_price_adj"] - tr["entry_price_adj"])

    tr["gross_rupees"] = tr["gross_points"] * mult
    tr["net_rupees_before_fees"] = tr["net_points"] * mult
    tr["fees_rupees"] = float(fee_rt_total)
    tr["net_rupees"] = tr["net_rupees_before_fees"] - tr["fees_rupees"]

    tr = tr.sort_values("exit_time").reset_index(drop=True)
    tr["cum_net_rupees"] = tr["net_rupees"].cumsum()
    tr["peak_net"] = tr["cum_net_rupees"].cummax()
    tr["dd_net_rupees"] = tr["cum_net_rupees"] - tr["peak_net"]
    tr["year"] = pd.to_datetime(tr["exit_time"]).dt.year

    wins = tr["net_rupees"] > 0
    losses = tr["net_rupees"] < 0
    pf = tr.loc[wins, "net_rupees"].sum() / (-tr.loc[losses, "net_rupees"].sum()) if losses.any() else np.inf

    overall = pd.DataFrame([{
        "trades": len(tr),
        "win_rate_net": float(wins.mean()),
        "avg_net_rupees": float(tr["net_rupees"].mean()),
        "median_net_rupees": float(tr["net_rupees"].median()),
        "profit_factor_net": float(pf),
        "max_dd_net_rupees": float(tr["dd_net_rupees"].min()),
        "total_net_rupees": float(tr["net_rupees"].sum()),
    }])

    yearly = (tr.groupby("year")
              .agg(trades=("net_rupees", "size"),
                   net_pnl=("net_rupees", "sum"),
                   avg_trade=("net_rupees", "mean"),
                   win_rate=("net_rupees", lambda s: (s > 0).mean()))
              .reset_index())

    return df, tr, overall, yearly

# Enrich trades
def enrich_trades(tr: pd.DataFrame) -> pd.DataFrame:
    tx = tr.copy()
    tx["entry_time"] = pd.to_datetime(tx["entry_time"])
    tx["exit_time"] = pd.to_datetime(tx["exit_time"])
    tx["duration_min"] = (tx["exit_time"] - tx["entry_time"]).dt.total_seconds() / 60.0
    tx["month"] = tx["exit_time"].dt.month
    tx["year"] = tx["exit_time"].dt.year
    tx["month_start"] = tx["exit_time"].dt.to_period("M").dt.to_timestamp()
    return tx

# Plots
def fig_price_with_trades(df: pd.DataFrame, tr: pd.DataFrame, last_days: int = 12) -> go.Figure:
    dfx = df.copy()
    dfx["d"] = dfx["date"].dt.date
    last = sorted(dfx["d"].unique())[-last_days:]
    dfx = dfx[dfx["d"].isin(last)].copy()

    trw = tr[(tr["entry_time"] >= dfx["date"].min()) & (tr["entry_time"] <= dfx["date"].max())].copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dfx["date"], open=dfx["open"], high=dfx["high"], low=dfx["low"], close=dfx["close"],
        name="Price",
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        increasing_fillcolor="rgba(46,229,157,0.20)",
        decreasing_fillcolor="rgba(255,92,138,0.20)",
        line=dict(width=1),
    ))

    fig.add_trace(go.Scatter(
        x=dfx["date"], y=dfx["emaH"],
        name="EMA(high)",
        line=dict(width=1.6, color=ACCENT),
        hovertemplate="EMAH<br>%{x}<br>%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dfx["date"], y=dfx["emaH"],
        name="", showlegend=False,
        line=dict(width=6, color=ACCENT_SOFT),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=dfx["date"], y=dfx["emaL"],
        name="EMA(low)",
        line=dict(width=1.6, color="rgba(122,162,255,0.75)"),
        hovertemplate="EMAL<br>%{x}<br>%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dfx["date"], y=dfx["emaL"],
        name="", showlegend=False,
        line=dict(width=6, color="rgba(122,162,255,0.18)"),
        hoverinfo="skip",
    ))

    if len(trw):
        longs = trw[trw["side"] == 1]
        shorts = trw[trw["side"] == -1]

        if len(longs):
            fig.add_trace(go.Scatter(
                x=longs["entry_time"], y=longs["entry_price"],
                mode="markers", name="Long entry",
                marker=dict(symbol="triangle-up", size=12, color=GREEN, line=dict(width=1, color="rgba(0,0,0,0.35)")),
                hovertemplate="Long entry<br>%{x}<br>%{y:.2f}<extra></extra>",
            ))
        if len(shorts):
            fig.add_trace(go.Scatter(
                x=shorts["entry_time"], y=shorts["entry_price"],
                mode="markers", name="Short entry",
                marker=dict(symbol="triangle-down", size=12, color=RED, line=dict(width=1, color="rgba(0,0,0,0.35)")),
                hovertemplate="Short entry<br>%{x}<br>%{y:.2f}<extra></extra>",
            ))

        fig.add_trace(go.Scatter(
            x=trw["exit_time"], y=trw["exit_price"],
            mode="markers", name="Exit",
            marker=dict(symbol="x", size=9, color=AMBER),
            text=trw["exit_reason"],
            hovertemplate="Exit (%{text})<br>%{x}<br>%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(xaxis_rangeslider_visible=False)
    return apply_glass_theme(fig, title=f"Price + EMA band + trades (last {last_days} days)")

def fig_equity(tr: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.66, 0.34], vertical_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=tr["exit_time"], y=tr["cum_net_rupees"],
        mode="lines", name="Equity (NET ₹)",
        line=dict(width=2.2, color=ACCENT),
        hovertemplate="%{x}<br><b>₹%{y:,.0f}</b><extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tr["exit_time"], y=tr["cum_net_rupees"],
        mode="lines", name="", showlegend=False,
        line=dict(width=8, color=ACCENT_SOFT),
        hoverinfo="skip",
    ), row=1, col=1)

    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.12)", row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tr["exit_time"], y=tr["dd_net_rupees"],
        mode="lines", name="Drawdown (₹)",
        line=dict(width=2.0, color=RED),
        fill="tozeroy", fillcolor=RED_SOFT,
        hovertemplate="%{x}<br><b>₹%{y:,.0f}</b><extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        xaxis=dict(rangeselector=dict(buttons=list([
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ])))
    )
    return apply_glass_theme(fig, title="NET Equity & Drawdown")

def fig_yearly(y: pd.DataFrame) -> go.Figure:
    colors = np.where(y["net_pnl"] >= 0, GREEN, RED)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=y["year"], y=y["net_pnl"],
        name="Net PnL (₹)",
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.10)",
        marker_line_width=1,
        hovertemplate="Year %{x}<br><b>₹%{y:,.0f}</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=y["year"], y=100 * y["win_rate"],
        mode="lines+markers", name="Win rate (%)",
        line=dict(width=2.1, color=ACCENT),
        marker=dict(size=7, color=ACCENT),
        hovertemplate="Year %{x}<br><b>%{y:.2f}%</b><extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=y["year"], y=y["trades"],
        mode="lines+markers", name="Trades",
        line=dict(width=1.7, dash="dot", color="rgba(245,247,255,0.65)"),
        marker=dict(size=6, color="rgba(245,247,255,0.75)"),
        hovertemplate="Year %{x}<br><b>%{y}</b> trades<extra></extra>",
    ), secondary_y=True)

    fig.update_yaxes(title_text="Net PnL (₹)", secondary_y=False)
    fig.update_yaxes(title_text="Win rate (%) / Trades", secondary_y=True)
    return apply_glass_theme(fig, title="Yearly scaling (NET after costs)")

def fig_proof_matrix_plotly(grid: pd.DataFrame, color_by: str = "rdd") -> go.Figure:
    """
    Plotly proof matrix:
      rows = TF × RR
      cols = EMA
      cell text = PnL / Win% / MaxDD / ReturnDD
      cell color = Return/DD (default) or PnL
    """

    dfx = grid.copy()
    dfx["win_pct"] = 100.0 * pd.to_numeric(dfx["win_rate_net"], errors="coerce")
    dfx["pnl"] = pd.to_numeric(dfx["total_net_rupees"], errors="coerce")
    dfx["dd"] = pd.to_numeric(dfx["max_dd_net_rupees"], errors="coerce")
    dfx["rdd"] = pd.to_numeric(dfx["return_dd"], errors="coerce")

    # Create row label TF×RR
    dfx["row"] = dfx["tf"].astype(str).str.upper() + " | RR " + dfx["rr"].astype(int).astype(str)

    # Keep a consistent order
    row_order = []
    for tf in ["5m", "15m", "30m"]:
        for rr in [1, 2, 3]:
            row_order.append(f"{tf.upper()} | RR {rr}")

    col_order = [55, 89]

    def fmt_rupee_short(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        sgn = "-" if x < 0 else ""
        v = abs(float(x))
        if v >= 1e7:
            return f"{sgn}₹{v/1e7:.2f}Cr"
        if v >= 1e5:
            return f"{sgn}₹{v/1e5:.2f}L"
        if v >= 1e3:
            return f"{sgn}₹{v/1e3:.1f}K"
        return f"{sgn}₹{v:.0f}"

    # Build text matrix
    txt = []
    z = []

    # choose color metric
    metric = "rdd" if color_by == "rdd" else "pnl"

    # Pivot numeric metric for z
    piv_z = dfx.pivot(index="row", columns="ema", values=metric).reindex(index=row_order, columns=col_order)

    # Compute robust z range
    zz = piv_z.to_numpy(dtype=float)
    finite = zz[np.isfinite(zz)]
    if finite.size == 0:
        zmin, zmax = -1.0, 1.0
    else:
        # robust range
        q05 = float(np.quantile(finite, 0.05))
        q95 = float(np.quantile(finite, 0.95))
        if q95 <= q05:
            q95 = q05 + 1.0
        zmin, zmax = q05, q95

    # Pivot components for text
    piv_pnl = dfx.pivot(index="row", columns="ema", values="pnl").reindex(index=row_order, columns=col_order)
    piv_win = dfx.pivot(index="row", columns="ema", values="win_pct").reindex(index=row_order, columns=col_order)
    piv_dd = dfx.pivot(index="row", columns="ema", values="dd").reindex(index=row_order, columns=col_order)
    piv_rdd = dfx.pivot(index="row", columns="ema", values="rdd").reindex(index=row_order, columns=col_order)

    for r in row_order:
        row_txt = []
        for ema_v in col_order:
            pnl = piv_pnl.loc[r, ema_v]
            win = piv_win.loc[r, ema_v]
            dd = piv_dd.loc[r, ema_v]
            rdd = piv_rdd.loc[r, ema_v]

            if not np.isfinite(pnl) and not np.isfinite(win) and not np.isfinite(dd) and not np.isfinite(rdd):
                row_txt.append("—")
            else:
                rdd_s = "—" if not np.isfinite(rdd) else f"{float(rdd):.3f}"
                win_s = "—" if not np.isfinite(win) else f"{float(win):.1f}%"
                row_txt.append(
                    f"{fmt_rupee_short(pnl)}  |  {win_s}<br>"
                    f"DD {fmt_rupee_short(dd)}<br>"
                    f"R/DD {rdd_s}"
                )
        txt.append(row_txt)

    z = piv_z.to_numpy(dtype=float)

    # Color scale: red -> amber -> green
    colorscale = [
        [0.00, "rgba(255,92,138,0.88)"],
        [0.50, "rgba(255,209,102,0.55)"],
        [1.00, "rgba(46,229,157,0.88)"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        x=[f"EMA {c}" for c in col_order],
        y=row_order,
        text=txt,
        texttemplate="%{text}",
        textfont=dict(color=TXT, size=12),
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        xgap=6,
        ygap=6,
        hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>",
        colorbar=dict(
            title=dict(text=("Return/DD" if metric == "rdd" else "Net PnL"), font=dict(color=MUTED)),
            tickfont=dict(color=MUTED),
            len=0.88,
        ),
    ))

    fig.update_layout(
        height=560,  # fits in one screen
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(side="top"),
    )
    title = "EMA 55 vs EMA 89 — Proof Matrix (TF×RR)"
    subtitle = "Colored by Return/DD" if metric == "rdd" else "Colored by Net PnL"
    return apply_glass_theme(fig, title=f"{title} • {subtitle}")

# Cinematic plots
def fig_equity_cinematic(tr: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tr["exit_time"], y=tr["cum_net_rupees"],
        mode="lines",
        line=dict(width=3, color=ACCENT),
        hovertemplate="%{x}<br><b>₹%{y:,.0f}</b><extra></extra>",
    ))
    return apply_cinematic_dark_pro(fig, "Equity Curve (NET after costs)")

def fig_underwater_cinematic(tr: pd.DataFrame, initial_capital: float) -> go.Figure:
    if initial_capital <= 0:
        initial_capital = 1.0

    equity = initial_capital + tr["cum_net_rupees"].astype(float)
    peak = equity.cummax().replace(0, np.nan)
    dd_pct = (equity - peak) / peak * 100.0
    dd_pct = dd_pct.fillna(0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tr["exit_time"], y=dd_pct,
        mode="lines",
        fill="tozeroy",
        line=dict(width=2.2, color=RED),
        fillcolor="rgba(255,92,138,0.20)",
        hovertemplate="%{x}<br><b>%{y:.2f}%</b><extra></extra>",
    ))
    fig.update_yaxes(title_text="Drawdown (%)")
    return apply_cinematic_dark_pro(fig, "Underwater Curve (Drawdown %)")

def fig_monthly_pnl_cinematic(tr: pd.DataFrame) -> go.Figure:
    tx = enrich_trades(tr)
    m = tx.groupby("month_start", as_index=False).agg(net_pnl=("net_rupees", "sum"))
    colors = np.where(m["net_pnl"] >= 0, GREEN, RED)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=m["month_start"], y=m["net_pnl"],
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.10)",
        marker_line_width=1,
        hovertemplate="%{x|%b %Y}<br><b>₹%{y:,.0f}</b><extra></extra>",
    ))
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Net PnL (₹)")
    return apply_cinematic_dark_pro(fig, "Monthly Net PnL (after costs)")

def _inr_short(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sgn = "-" if x < 0 else ""
    v = abs(float(x))
    if v >= 1e7:
        return f"{sgn}₹{v / 1e7:.2f}Cr"
    if v >= 1e5:
        return f"{sgn}₹{v / 1e5:.2f}L"
    if v >= 1e3:
        return f"{sgn}₹{v / 1e3:.1f}K"
    return f"{sgn}₹{v:.0f}"

def fig_monthly_heatmap(tr: pd.DataFrame) -> go.Figure:
    tx = enrich_trades(tr)
    m = tx.groupby(["year", "month"], as_index=False)["net_rupees"].sum()
    pivot = m.pivot(index="year", columns="month", values="net_rupees").reindex(columns=range(1, 13))

    z = pivot.to_numpy(dtype=float)
    years = pivot.index.astype(str).tolist()
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    max_abs = float(np.nanmax(np.abs(z))) if np.isfinite(np.nanmax(np.abs(z))) else 1.0
    if max_abs == 0:
        max_abs = 1.0

    txt = [[_inr_short(z[i, j]) for j in range(z.shape[1])] for i in range(z.shape[0])]
    n_years = len(years)
    heat_h = max(560, 46 * n_years)
    font_sz = 13 if n_years <= 10 else 11

    colorscale = [
        [0.00, "rgba(255,92,138,0.88)"],
        [0.50, "rgba(255,209,102,0.55)"],
        [1.00, "rgba(46,229,157,0.88)"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        x=month_labels,
        y=years,
        text=txt,
        texttemplate="%{text}",
        textfont=dict(color=TXT, size=font_sz),
        colorscale=colorscale,
        zmin=-max_abs,
        zmax=max_abs,
        xgap=3,
        ygap=3,
        hovertemplate="Year %{y}<br>Month %{x}<br><b>%{text}</b><extra></extra>",
        colorbar=dict(
            title=dict(text="₹ (Net)", font=dict(color=MUTED)),
            tickfont=dict(color=MUTED),
            len=0.88,
        ),
    ))
    fig.update_layout(height=heat_h)
    fig.update_xaxes(side="top")
    return apply_cinematic_dark_pro(fig, "Monthly Consistency Heatmap (NET ₹)")

def fig_trade_distribution_cinematic(tr: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=tr["net_rupees"],
        nbinsx=70,
        marker=dict(color="rgba(122,162,255,0.85)"),
        marker_line=dict(color="rgba(255,255,255,0.10)", width=1),
        hovertemplate="PnL: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_xaxes(title_text="PnL per trade (₹)")
    fig.update_yaxes(title_text="Frequency")
    return apply_cinematic_dark_pro(fig, "Trade Outcome Distribution")

def fig_duration_hist_cinematic(tr: pd.DataFrame) -> go.Figure:
    tx = enrich_trades(tr)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=tx["duration_min"],
        nbinsx=50,
        marker=dict(color="rgba(46,229,157,0.65)"),
        marker_line=dict(color="rgba(255,255,255,0.10)", width=1),
        hovertemplate="Duration: %{x:.1f} min<br>Count: %{y}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Trade duration (minutes)")
    fig.update_yaxes(title_text="Frequency")
    return apply_cinematic_dark_pro(fig, "How Long Trades Last (Duration)")

def fig_rolling_stats_cinematic(tr: pd.DataFrame, window_trades: int = 50) -> go.Figure:
    pnl = tr["net_rupees"].astype(float)
    minp = max(10, window_trades // 4)

    roll_avg = pnl.rolling(window_trades, min_periods=minp).mean()
    roll_wr = (pnl > 0).rolling(window_trades, min_periods=minp).mean() * 100.0

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, row_heights=[0.52, 0.48])

    fig.add_trace(go.Scatter(
        x=tr["exit_time"], y=roll_avg,
        mode="lines",
        line=dict(width=2.6, color=ACCENT),
        hovertemplate="%{x}<br><b>₹%{y:,.0f}</b><extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text=f"Avg PnL (₹) | last {window_trades} trades", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=tr["exit_time"], y=roll_wr,
        mode="lines",
        line=dict(width=2.4, color=GREEN),
        hovertemplate="%{x}<br><b>%{y:.1f}%</b><extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(title_text=f"Win rate (%) | last {window_trades} trades", row=2, col=1)

    return apply_cinematic_dark_pro(fig, "Stability Over Time (Rolling stats)")

def fig_strategy_vs_buyhold(df: pd.DataFrame, tr: pd.DataFrame, initial_capital: float) -> go.Figure:
    if initial_capital <= 0:
        initial_capital = 1.0

    sx = tr.copy().sort_values("exit_time").reset_index(drop=True)
    strat_equity = initial_capital + sx["cum_net_rupees"].astype(float)
    strat_index = 100.0 * (strat_equity / float(strat_equity.iloc[0]))

    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    start = pd.to_datetime(sx["exit_time"]).min()
    end = pd.to_datetime(sx["exit_time"]).max()
    dfx = dfx[(dfx["date"] >= start) & (dfx["date"] <= end)].copy()
    if dfx.empty:
        dfx = df.copy().sort_values("date")

    bh = 100.0 * (dfx["close"].astype(float) / float(dfx["close"].iloc[0]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sx["exit_time"], y=strat_index,
        mode="lines",
        line=dict(width=3, color=ACCENT),
        name="Strategy",
        hovertemplate="%{x}<br><b>%{y:.2f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dfx["date"], y=bh,
        mode="lines",
        line=dict(width=2, color="rgba(245,247,255,0.70)"),
        name="Buy & Hold",
        hovertemplate="%{x}<br><b>%{y:.2f}</b><extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.02, bgcolor="rgba(0,0,0,0)")
    )
    fig.update_yaxes(title_text="Index (Start = 100)")
    return apply_cinematic_dark_pro(fig, "Strategy vs Buy & Hold (Indexed)")

# TF × RR heatmap grid
@st.cache_data(show_spinner=False)
def compute_tf_rr_grid(
        ema_len_: int,
        lot_: int,
        ncon_: int,
        slip_: float,
        fee_: float,
) -> pd.DataFrame:
    rows = []
    tfs_all = available_timeframes()
    rr_list = [1.0, 2.0, 3.0]

    for tf_ in tfs_all:
        df_in = load_csv_from_data(tf_)
        forced_exit_t_ = FORCED_EXIT_BY_TF.get(tf_, time(15, 10))

        for rr_ in rr_list:
            _, _, overall_, _ = run_backtest(
                df_raw=df_in,
                ema_len=int(ema_len_),
                rr_mult=float(rr_),
                start_t=START_T,
                last_entry_t=LAST_ENTRY_T,
                forced_exit_bar_time=forced_exit_t_,
                lot_size=int(lot_),
                n_contracts=int(ncon_),
                slip_pts_side=float(slip_),
                fee_rt_total=float(fee_),
            )

            if overall_ is None or overall_.empty:
                rows.append({"tf": tf_, "rr": rr_, "trades": 0, "win_rate_net": np.nan, "total_net_rupees": 0.0})
            else:
                om_ = overall_.iloc[0].to_dict()
                rows.append({
                    "tf": tf_,
                    "rr": float(rr_),
                    "trades": int(om_.get("trades", 0)),
                    "win_rate_net": float(om_.get("win_rate_net", np.nan)),
                    "total_net_rupees": float(om_.get("total_net_rupees", 0.0)),
                })

    out = pd.DataFrame(rows).sort_values(["tf", "rr"]).reset_index(drop=True)
    return out

def fig_tf_rr_heatmap(grid: pd.DataFrame, color_metric: str) -> go.Figure:
    tfs = ["5m", "15m", "30m"]
    tfs = [t for t in tfs if t in grid["tf"].unique()]
    rrs = [1.0, 2.0, 3.0]

    pnl = grid.pivot(index="tf", columns="rr", values="total_net_rupees").reindex(index=tfs, columns=rrs)
    acc = grid.pivot(index="tf", columns="rr", values="win_rate_net").reindex(index=tfs, columns=rrs)

    txt = []
    for tf in tfs:
        row_txt = []
        for rr in rrs:
            p = pnl.loc[tf, rr]
            a = acc.loc[tf, rr]
            if pd.isna(a):
                row_txt.append(f"₹{p:,.0f}<br>—")
            else:
                row_txt.append(f"₹{p:,.0f}<br>{100 * a:.1f}%")
        txt.append(row_txt)

    if color_metric == "acc":
        z = (acc * 100.0).to_numpy(dtype=float)
        zmin, zmax = 0.0, 100.0
        title = "All TF × RR — Accuracy (NET win rate)"
        colorscale = [
            [0.00, "rgba(255,92,138,0.85)"],
            [0.50, "rgba(255,209,102,0.85)"],
            [1.00, "rgba(46,229,157,0.85)"],
        ]
        cb_title = "Win%"
    else:
        z = pnl.to_numpy(dtype=float)
        max_abs = float(np.nanmax(np.abs(z))) if np.isfinite(np.nanmax(np.abs(z))) else 1.0
        if max_abs == 0:
            max_abs = 1.0
        zmin, zmax = -max_abs, max_abs
        title = "All TF × RR — Net PnL (₹)"
        colorscale = [
            [0.00, "rgba(255,92,138,0.85)"],
            [0.50, "rgba(255,209,102,0.45)"],
            [1.00, "rgba(46,229,157,0.85)"],
        ]
        cb_title = "₹ Net"

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        x=[f"RR {int(r)}" for r in rrs],
        y=tfs,
        text=txt,
        texttemplate="%{text}",
        textfont=dict(color=TXT, size=12),
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        hovertemplate="TF %{y}<br>%{x}<br>%{text}<extra></extra>",
        colorbar=dict(
            title=dict(text=cb_title, font=dict(color=MUTED)),
            tickfont=dict(color=MUTED),
            len=0.85,
        ),
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), xaxis=dict(side="top"))
    return apply_glass_theme(fig, title=title)

# EMA 55 vs 89 Grid (metrics)
@st.cache_data(show_spinner=False)
def compute_ema_tf_rr_grid(
        ema_list: tuple[int, ...],
        lot_: int,
        ncon_: int,
        slip_: float,
        fee_: float,
) -> pd.DataFrame:
    rows = []
    tfs_all = available_timeframes()
    rr_list = [1.0, 2.0, 3.0]

    for ema_len_ in ema_list:
        for tf_ in tfs_all:
            df_in = load_csv_from_data(tf_)
            forced_exit_t_ = FORCED_EXIT_BY_TF.get(tf_, time(15, 10))

            for rr_ in rr_list:
                _, tr_, overall_, _ = run_backtest(
                    df_raw=df_in,
                    ema_len=int(ema_len_),
                    rr_mult=float(rr_),
                    start_t=START_T,
                    last_entry_t=LAST_ENTRY_T,
                    forced_exit_bar_time=forced_exit_t_,
                    lot_size=int(lot_),
                    n_contracts=int(ncon_),
                    slip_pts_side=float(slip_),
                    fee_rt_total=float(fee_),
                )

                if overall_ is None or overall_.empty or tr_ is None or tr_.empty:
                    rows.append({
                        "ema": int(ema_len_),
                        "tf": tf_,
                        "rr": float(rr_),
                        "trades": 0,
                        "win_rate_net": np.nan,
                        "profit_factor_net": np.nan,
                        "avg_net_rupees": np.nan,
                        "median_net_rupees": np.nan,
                        "max_dd_net_rupees": 0.0,
                        "total_net_rupees": 0.0,
                    })
                else:
                    om_ = overall_.iloc[0].to_dict()
                    rows.append({
                        "ema": int(ema_len_),
                        "tf": tf_,
                        "rr": float(rr_),
                        "trades": int(om_.get("trades", 0)),
                        "win_rate_net": float(om_.get("win_rate_net", np.nan)),
                        "profit_factor_net": float(om_.get("profit_factor_net", np.nan)),
                        "avg_net_rupees": float(om_.get("avg_net_rupees", np.nan)),
                        "median_net_rupees": float(om_.get("median_net_rupees", np.nan)),
                        "max_dd_net_rupees": float(om_.get("max_dd_net_rupees", 0.0)),
                        "total_net_rupees": float(om_.get("total_net_rupees", 0.0)),
                    })

    out = pd.DataFrame(rows).sort_values(["ema", "tf", "rr"]).reset_index(drop=True)

    dd = out["max_dd_net_rupees"].astype(float).abs()
    pnl = out["total_net_rupees"].astype(float)
    out["return_dd"] = np.where(dd > 0, pnl / dd, np.nan)

    return out

# Proof Table rendering
def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _fmt_rupee(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"₹{x:,.0f}"

def _fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}%"

def _fmt_float(x: float, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{d}f}"

def _bar_html(value: float, vmin: float, vmax: float, good_high: bool = True) -> tuple[str, str]:
    """
    Returns (bg_bar_html, text_color)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "", "rgba(245,247,255,0.88)"

    if vmax <= vmin:
        t = 0.5
    else:
        t = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
    t = _clamp01(t)
    if not good_high:
        t = 1.0 - t

    # gradient from red -> amber -> green
    # pick color by t
    # 0..0.5: red->amber, 0.5..1: amber->green
    if t < 0.5:
        tt = t / 0.5
        col = f"rgba({int(255 - 0 * tt)},{int(92 + (209 - 92) * tt)},{int(138 + (102 - 138) * tt)},0.55)"
    else:
        tt = (t - 0.5) / 0.5
        col = f"rgba({int(255 + (46 - 255) * tt)},{int(209 + (229 - 209) * tt)},{int(102 + (157 - 102) * tt)},0.55)"

    width = int(8 + 92 * t)  # 8..100
    bar = f"""
    <div style="position:absolute; inset:0; padding:6px 10px;">
      <div style="height:100%; width:{width}%; background:{col}; border-radius:10px; filter: blur(0px);"></div>
    </div>
    """
    return bar, "rgba(245,247,255,0.92)"

def render_proof_table_html(df: pd.DataFrame, caption: str, big: bool = False) -> str:
    dfx = df.copy()

    # prep metrics
    dfx["win_pct"] = 100.0 * dfx["win_rate_net"]
    dfx["pf"] = dfx["profit_factor_net"]
    dfx["avg_trade"] = dfx["avg_net_rupees"]
    dfx["med_trade"] = dfx["median_net_rupees"]
    dfx["dd"] = dfx["max_dd_net_rupees"]
    dfx["pnl"] = dfx["total_net_rupees"]
    dfx["rdd"] = dfx["return_dd"]

    # ranges for bar scaling (robust to outliers)
    def robust_minmax(s: pd.Series) -> tuple[float, float]:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return 0.0, 1.0
        lo = float(s.quantile(0.05))
        hi = float(s.quantile(0.95))
        if hi == lo:
            hi = lo + 1.0
        return lo, hi

    pnl_min, pnl_max = robust_minmax(dfx["pnl"])
    dd_min, dd_max = robust_minmax(dfx["dd"].abs())
    win_min, win_max = 0.0, 100.0
    pf_min, pf_max = robust_minmax(dfx["pf"])
    rdd_min, rdd_max = robust_minmax(dfx["rdd"])

    # font sizes
    fs = "16px" if big else "13px"
    fsh = "13px" if big else "12px"
    pad = "12px 14px" if big else "10px 12px"
    rowh = "52px" if big else "44px"

    # CSS + table shell
    html = f"""
    <div style="
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 14px 14px 10px 14px;
        backdrop-filter: blur(10px);
        overflow-x: auto;
    ">
      <div style="display:flex; align-items:baseline; justify-content:space-between; margin-bottom:10px;">
        <div style="color: rgba(245,247,255,0.92); font-weight: 750; font-size: {"20px" if big else "16px"}; letter-spacing:0.2px;">
          {caption}
        </div>
        <div style="color: rgba(245,247,255,0.62); font-size:{fsh};">
          Return/DD = Net PnL ÷ |Max DD|
        </div>
      </div>

      <table style="
          width: 100%;
          border-collapse: separate;
          border-spacing: 0 8px;
          font-family: Inter, Segoe UI, Arial;
          font-size: {fs};
          color: rgba(245,247,255,0.92);
      ">
        <thead>
          <tr>
            {''.join([f'<th style="text-align:center; color:rgba(245,247,255,0.80); font-size:{fsh}; font-weight:700; padding: 0 10px;">{h}</th>' for h in [
        "EMA", "TF", "RR", "Trades", "Win %", "PF", "Return/DD", "Avg/Trade", "Median/Trade", "Max DD", "Net PnL"
    ]])}
          </tr>
        </thead>
        <tbody>
    """

    def chip(text: str, fg: str, bg: str) -> str:
        return f"""
        <span style="
            display:inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: {bg};
            border: 1px solid rgba(255,255,255,0.10);
            color: {fg};
            font-weight: 750;
            letter-spacing: 0.2px;
            font-size: {fsh};
        ">{text}</span>
        """

    # build rows
    for _, r in dfx.iterrows():
        ema_v = int(r["ema"])
        tf_v = str(r["tf"])
        rr_v = float(r["rr"])
        trades_v = int(r["trades"]) if np.isfinite(r["trades"]) else 0

        win_v = float(r["win_pct"]) if np.isfinite(r["win_pct"]) else np.nan
        pf_v = float(r["pf"]) if np.isfinite(r["pf"]) else np.nan
        rdd_v = float(r["rdd"]) if np.isfinite(r["rdd"]) else np.nan
        avg_v = float(r["avg_trade"]) if np.isfinite(r["avg_trade"]) else np.nan
        med_v = float(r["med_trade"]) if np.isfinite(r["med_trade"]) else np.nan
        dd_v = float(r["dd"]) if np.isfinite(r["dd"]) else np.nan
        pnl_v = float(r["pnl"]) if np.isfinite(r["pnl"]) else np.nan

        pnl_bar, _ = _bar_html(pnl_v, pnl_min, pnl_max, good_high=True)
        dd_bar, _ = _bar_html(abs(dd_v) if np.isfinite(dd_v) else np.nan, dd_min, dd_max, good_high=False)
        win_bar, _ = _bar_html(win_v, win_min, win_max, good_high=True)
        pf_bar, _ = _bar_html(pf_v, pf_min, pf_max, good_high=True)
        rdd_bar, _ = _bar_html(rdd_v, rdd_min, rdd_max, good_high=True)

        # quick chips
        ema_chip = chip(f"EMA {ema_v}", "rgba(245,247,255,0.92)", "rgba(122,162,255,0.14)")
        tf_chip = chip(tf_v.upper(), "rgba(245,247,255,0.92)", "rgba(255,255,255,0.08)")
        rr_chip = chip(f"RR {int(rr_v)}", "rgba(245,247,255,0.92)", "rgba(255,209,102,0.14)")

        html += f"""
          <tr style="height:{rowh};">
            <td style="text-align:center; padding:{pad};">{ema_chip}</td>
            <td style="text-align:center; padding:{pad};">{tf_chip}</td>
            <td style="text-align:center; padding:{pad};">{rr_chip}</td>

            <td style="text-align:center; padding:{pad}; background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius: 14px;">
              {trades_v:,}
            </td>

            <td style="position:relative; text-align:center; padding:{pad}; background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius: 14px; overflow:hidden;">
              {win_bar}
              <div style="position:relative; font-weight:750;">{_fmt_pct(win_v)}</div>
            </td>

            <td style="position:relative; text-align:center; padding:{pad}; background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius: 14px; overflow:hidden;">
              {pf_bar}
              <div style="position:relative; font-weight:750;">{_fmt_float(pf_v, 3)}</div>
            </td>

            <td style="position:relative; text-align:center; padding:{pad}; background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius: 14px; overflow:hidden;">
              {rdd_bar}
              <div style="position:relative; font-weight:750;">{_fmt_float(rdd_v, 3)}</div>
            </td>

            <td style="text-align:center; padding:{pad}; background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius: 14px;">
              {_fmt_rupee(avg_v)}
            </td>
            <td style="text-align:center; padding:{pad}; background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius: 14px;">
              {_fmt_rupee(med_v)}
            </td>

            <td style="position:relative; text-align:center; padding:{pad}; background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius: 14px; overflow:hidden;">
              {dd_bar}
              <div style="position:relative; font-weight:750;">{_fmt_rupee(dd_v)}</div>
            </td>

            <td style="position:relative; text-align:center; padding:{pad}; background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius: 14px; overflow:hidden;">
              {pnl_bar}
              <div style="position:relative; font-weight:800;">{_fmt_rupee(pnl_v)}</div>
            </td>
          </tr>
        """

    html += """
        </tbody>
      </table>
    </div>
    """
    return html

# Proof Matrix renderer
def _mx_fmt_rupee_short(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sgn = "-" if x < 0 else ""
    v = abs(float(x))
    if v >= 1e7:
        return f"{sgn}₹{v / 1e7:.2f}Cr"
    if v >= 1e5:
        return f"{sgn}₹{v / 1e5:.2f}L"
    if v >= 1e3:
        return f"{sgn}₹{v / 1e3:.1f}K"
    return f"{sgn}₹{v:.0f}"

def _mx_fmt_pct1(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.1f}%"

def _mx_robust_minmax(s: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    lo = float(s.quantile(0.05))
    hi = float(s.quantile(0.95))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi

def _mx_score_color(t: float) -> str:
    """
    0 -> red, 0.5 -> amber, 1 -> green
    """
    t = _clamp01(float(t))
    if t < 0.5:
        tt = t / 0.5
        r = 255
        g = int(92 + (209 - 92) * tt)
        b = int(138 + (102 - 138) * tt)
        return f"rgba({r},{g},{b},0.30)"
    tt = (t - 0.5) / 0.5
    r = int(255 + (46 - 255) * tt)
    g = int(209 + (229 - 209) * tt)
    b = int(102 + (157 - 102) * tt)
    return f"rgba({r},{g},{b},0.30)"

def render_proof_matrix_html(grid: pd.DataFrame, caption: str = "EMA 55 vs EMA 89 — Proof Matrix (TF × RR)") -> str:
    """
    Compact matrix:
      Rows  : TF×RR (<= 9)
      Cols  : EMA 55, EMA 89
      Cell  : Net PnL, Win%, Max DD, Return/DD

    """

    dfx = grid.copy()
    dfx["win_pct"] = 100.0 * pd.to_numeric(dfx["win_rate_net"], errors="coerce")
    dfx["pnl"] = pd.to_numeric(dfx["total_net_rupees"], errors="coerce")
    dfx["dd_abs"] = pd.to_numeric(dfx["max_dd_net_rupees"], errors="coerce").abs()
    dfx["pf"] = pd.to_numeric(dfx["profit_factor_net"], errors="coerce")
    dfx["rdd"] = pd.to_numeric(dfx["return_dd"], errors="coerce")

    pnl_lo, pnl_hi = _mx_robust_minmax(dfx["pnl"])
    rdd_lo, rdd_hi = _mx_robust_minmax(dfx["rdd"])
    pf_lo, pf_hi = _mx_robust_minmax(dfx["pf"])
    dd_lo, dd_hi = _mx_robust_minmax(dfx["dd_abs"])

    def _norm(v, lo, hi) -> float:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            return 0.5
        return _clamp01((float(v) - lo) / (hi - lo))

    def _score(row: pd.Series) -> float:
        s_pnl = _norm(row["pnl"], pnl_lo, pnl_hi)
        s_rdd = _norm(row["rdd"], rdd_lo, rdd_hi)
        s_pf = _norm(row["pf"], pf_lo, pf_hi)
        s_win = _norm(row["win_pct"], 0.0, 100.0)
        s_dd_bad = _norm(row["dd_abs"], dd_lo, dd_hi)  # 0 good -> 1 bad
        score = (0.35 * s_pnl + 0.35 * s_rdd + 0.20 * s_pf + 0.10 * s_win) - 0.18 * s_dd_bad
        return _clamp01(score)

    combos = []
    for tf in ["5m", "15m", "30m"]:
        for rr in [1.0, 2.0, 3.0]:
            if ((dfx["tf"] == tf) & (dfx["rr"] == rr)).any():
                combos.append((tf, rr))

    def _label_card(tf: str, rr: float) -> str:
        return f"""
        <div style="
          border:1px solid rgba(255,255,255,0.08);
          border-radius:16px;
          background: rgba(255,255,255,0.02);
          padding: 12px 12px;
          height: 96px;
          display:flex;
          flex-direction:column;
          justify-content:center;
          gap: 6px;
        ">
          <div style="font-weight:900; color:{TXT}; font-size:14px;">{tf.upper()}</div>
          <div style="font-weight:850; color:{MUTED}; font-size:12px;">RR {int(rr)}</div>
        </div>
        """

    def _empty_card() -> str:
        return """
        <div style="
          border:1px solid rgba(255,255,255,0.06);
          border-radius:16px;
          background: rgba(255,255,255,0.02);
          padding: 12px;
          height: 96px;
          display:flex; align-items:center; justify-content:center;
          color: rgba(245,247,255,0.45);
          font-weight:800;
        ">—</div>
        """

    def _metric_card(row: pd.Series | None) -> str:
        if row is None:
            return _empty_card()

        score = _score(row)
        bg = _mx_score_color(score)

        pnl = row["pnl"]
        win = row["win_pct"]
        dd = pd.to_numeric(row.get("max_dd_net_rupees"), errors="coerce")
        dd = float(dd) if np.isfinite(dd) else np.nan
        rdd = row["rdd"]

        rdd_txt = "—"
        if np.isfinite(rdd):
            rdd_txt = f"{float(rdd):.3f}"

        return f"""
        <div style="
          position:relative;
          border:1px solid rgba(255,255,255,0.10);
          border-radius:16px;
          background: rgba(255,255,255,0.02);
          overflow:hidden;
          padding: 10px 12px;
          height: 96px;
        ">
          <div style="position:absolute; inset:0; background:{bg};"></div>
          <div style="position:relative; display:grid; grid-template-columns: 1fr 1fr; row-gap: 8px; column-gap:10px;">
            <div style="font-weight:950; font-size:16px; color:{TXT};">{_mx_fmt_rupee_short(pnl)}</div>
            <div style="text-align:right; font-weight:900; font-size:14px; color:{TXT};">{_mx_fmt_pct1(win)}</div>

            <div style="font-size:12px; color:{MUTED};">
              Max DD: <span style="color:{TXT}; font-weight:900;">{_mx_fmt_rupee_short(dd)}</span>
            </div>
            <div style="text-align:right; font-size:12px; color:{MUTED};">
              R/DD: <span style="color:{TXT}; font-weight:900;">{rdd_txt}</span>
            </div>
          </div>
        </div>
        """

    html = f"""
    <div style="
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 14px;
        backdrop-filter: blur(10px);
    ">
      <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px;">
        <div style="font-weight:900; font-size:18px; color:{TXT};">{caption}</div>
        <div style="font-size:12px; color:{MUTED};">Cell tint = blended score (PnL + Return/DD + PF + Win% minus DD)</div>
      </div>

      <div style="display:grid; grid-template-columns: 140px 1fr 1fr; gap: 10px;">
        <div style="color:{MUTED}; font-size:12px; font-weight:900; padding: 6px 10px;">TF × RR</div>
        <div style="color:{MUTED}; font-size:12px; font-weight:950; padding: 6px 10px;">EMA 55</div>
        <div style="color:{MUTED}; font-size:12px; font-weight:950; padding: 6px 10px;">EMA 89</div>
    """

    for tf, rr in combos:
        r55 = dfx[(dfx["tf"] == tf) & (dfx["rr"] == rr) & (dfx["ema"] == 55)]
        r89 = dfx[(dfx["tf"] == tf) & (dfx["rr"] == rr) & (dfx["ema"] == 89)]
        row55 = r55.iloc[0] if len(r55) else None
        row89 = r89.iloc[0] if len(r89) else None

        html += _label_card(tf, rr)
        html += _metric_card(row55)
        html += _metric_card(row89)

    html += """
      </div>
    </div>
    """
    html = textwrap.dedent(html).strip()
    return html

# Streamlit UI
st.set_page_config(page_title="NIFTY50 Strategy", layout="wide")

st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{
        font-family: Inter, Segoe UI, Arial;
      }}
      .stApp {{
        background: radial-gradient(1200px 600px at 20% 10%, rgba(122,162,255,0.10), rgba(0,0,0,0)) ,
                    radial-gradient(900px 500px at 80% 30%, rgba(46,229,157,0.08), rgba(0,0,0,0)) ,
                    {BG};
      }}
      [data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.03);
        border-right: 1px solid rgba(255,255,255,0.06);
        backdrop-filter: blur(10px);
      }}
      .glass-card {{
        background: {PANEL};
        border: 1px solid {PANEL_BORDER};
        border-radius: 16px;
        padding: 14px 16px;
        backdrop-filter: blur(10px);
      }}
      .kpi {{
        font-size: 12px;
        color: {MUTED};
        margin-bottom: 4px;
      }}
      .kpi-val {{
        font-size: 22px;
        color: {TXT};
        font-weight: 650;
      }}
      .kpi-sub {{
        font-size: 12px;
        color: {MUTED};
        margin-top: 2px;
      }}
      .subtle-note {{
        font-size: 12px;
        color: rgba(245,247,255,0.60);
        margin-top: 6px;
      }}
      .hint {{
        font-size: 12px;
        color: rgba(245,247,255,0.62);
        margin-top: 6px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

LINKEDIN_URL = "https://www.linkedin.com/in/jpqtre/"

title_col, link_col = st.columns([0.88, 0.12], vertical_alignment="center")
with title_col:
    st.title("NIFTY50 Medium Frequency Strategy Back-test")
with link_col:
    st.markdown(
        f"""
        <div style="display:flex; justify-content:flex-end; align-items:center; height:64px;">
          <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer"
             style="display:inline-flex; align-items:center; text-decoration:none;">
            <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24"
                 fill="#0A66C2" style="filter: drop-shadow(0 0 6px rgba(10,102,194,0.35));">
              <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 1 1 0-4.124 2.062 2.062 0 0 1 0 4.124zM6.814 20.452H3.86V9h2.954v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.727v20.545C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.273V1.727C24 .774 23.2 0 22.222 0z"/>
            </svg>
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("Personal strategy backtested over the last 10 years.")
st.caption("Created by Jay Patel")

with st.sidebar:
    st.header("Data (one-click)")
    tfs = available_timeframes()
    if not tfs:
        st.error(
            f"No CSVs found in: {DATA_DIR}\n\n"
            f"Create ./data and add one or more of:\n"
            + "\n".join([f"- {v}" for v in TIMEFRAME_FILES.values()])
        )
        st.stop()

    tf = st.selectbox("Timeframe", tfs, index=0)
    st.caption(f"Loaded: data/{TIMEFRAME_FILES[tf]}")

    st.header("Strategy")
    ema_len = st.number_input("EMA length", min_value=5, max_value=300, value=89, step=1)
    rr_mult = st.selectbox("Risk:Reward", [1.0, 2.0, 3.0], index=1)

    st.header("Times (fixed)")
    forced_exit_t = FORCED_EXIT_BY_TF.get(tf, time(15, 10))
    st.write(f"Entry: {START_T} → {LAST_ENTRY_T}")
    st.write(f"Forced exit: {forced_exit_t} (bar close)")

    st.header("Sizing + costs")
    lot_size = st.number_input("Lot size", min_value=1, value=65, step=1)
    n_contracts = st.number_input("Contracts", min_value=1, value=20, step=1)
    slip = st.number_input("Slippage (pts/side)", min_value=0.0, value=0.25, step=0.05, format="%.2f")
    fee_rt = st.number_input("Fees (₹/round-trip total)", min_value=0.0, value=40.0, step=5.0)

    st.header("Visuals")
    last_days = st.slider("Price window (days)", 5, 60, 12, 1)
    roll_n = st.slider("Rolling window (trades)", 20, 150, 50, 5)
    initial_capital = st.number_input(
        "Capital for comparisons (₹)",
        min_value=1000.0,
        value=500000.0,
        step=50000.0,
        help="Used for Strategy vs Buy&Hold and Underwater (drawdown %) calculations.",
    )

    st.header("Proof Table")
    show_proof = st.checkbox("Show proof table in Overview", value=True)
    st.markdown(
        "<div class='hint'>Screenshots: Proof Table tab + Visuals tab.</div>",
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def cached_run(tf_: str, ema_len_: int, rr_: float, lot_: int, ncon_: int, slip_: float, fee_: float):
    df_in = load_csv_from_data(tf_)
    forced_exit_t_ = FORCED_EXIT_BY_TF.get(tf_, time(15, 10))
    return run_backtest(
        df_raw=df_in,
        ema_len=ema_len_,
        rr_mult=rr_,
        start_t=START_T,
        last_entry_t=LAST_ENTRY_T,
        forced_exit_bar_time=forced_exit_t_,
        lot_size=lot_,
        n_contracts=ncon_,
        slip_pts_side=slip_,
        fee_rt_total=fee_,
    )


with st.spinner("Running backtest..."):
    df, tr, overall, yearly = cached_run(
        tf,
        int(ema_len),
        float(rr_mult),
        int(lot_size),
        int(n_contracts),
        float(slip),
        float(fee_rt),
    )

if len(tr) == 0:
    st.error("No trades produced with these settings.")
    st.stop()

tr = tr.copy()
tr["entry_time"] = pd.to_datetime(tr["entry_time"])
tr["exit_time"] = pd.to_datetime(tr["exit_time"])

om = overall.iloc[0].to_dict()
k1, k2, k3, k4 = st.columns(4)

def kpi(col, label, val, sub=""):
    col.markdown(
        f"""
        <div class="glass-card">
          <div class="kpi">{label}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


kpi(k1, "Total NET (₹)", f"{om['total_net_rupees']:,.0f}",
    f"TF: {tf} | Contracts: {int(n_contracts)} | Lot: {int(lot_size)}")
kpi(k2, "Max DD NET (₹)", f"{om['max_dd_net_rupees']:,.0f}", f"Slip: {slip:.2f} pts/side | Fees: ₹{fee_rt:.0f}/RT")
kpi(k3, "Profit factor (NET)", f"{om['profit_factor_net']:.3f}", f"Trades: {int(om['trades'])}")
kpi(k4, "Win rate (NET)", f"{100 * om['win_rate_net']:.2f}%", f"RR: {rr_mult:g}")

tabs = st.tabs([
    "Overview",
    "Price + Trades",
    "Equity",
    "Yearly",
    "Strategy Visuals",
    "Proof Table",
    "Trades",
])

with tabs[0]:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Overall metrics")
        st.dataframe(overall, use_container_width=True, hide_index=True)

        st.markdown("### All timeframes × Risk:Reward")

        view_mode = st.radio(
            "Color cells by",
            options=["Net PnL (₹)", "Accuracy (%)"],
            horizontal=True,
            index=0,
        )

        with st.spinner("Computing TF × RR grid..."):
            grid = compute_tf_rr_grid(
                ema_len_=int(ema_len),
                lot_=int(lot_size),
                ncon_=int(n_contracts),
                slip_=float(slip),
                fee_=float(fee_rt),
            )

        metric_key = "pnl" if view_mode.startswith("Net") else "acc"
        st.plotly_chart(fig_tf_rr_heatmap(grid, color_metric=metric_key), use_container_width=True,
            key="plotly_1"
        )

        st.markdown(
            "<div class='subtle-note'>Each cell shows: Net PnL (₹) and accuracy (NET win rate). "
            "Costs/slippage/sizing match the sidebar settings.</div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.subheader("Yearly summary")
        st.dataframe(yearly, use_container_width=True, hide_index=True)

    if show_proof:
        st.markdown("## EMA 55 vs EMA 89 — Proof (TF × RR)")
        st.markdown(
            "<div class='subtle-note'>Compact matrix view.</div>",
            unsafe_allow_html=True,
        )

        ema_compare = (55, 89)
        with st.spinner("Computing EMA 55 vs 89 grid..."):
            grid2 = compute_ema_tf_rr_grid(
                ema_list=ema_compare,
                lot_=int(lot_size),
                ncon_=int(n_contracts),
                slip_=float(slip),
                fee_=float(fee_rt),
            )

        st.markdown("### EMA 55 vs EMA 89 — Proof Matrix (TF × RR)")

        mode2 = st.radio(
            "Color cells by",
            ["Return/DD", "Net PnL"],
            horizontal=True,
            index=0,
            key="proof_overview_color",
        )

        color_key2 = "rdd" if mode2.startswith("Return") else "pnl"
        st.plotly_chart(
            fig_proof_matrix_plotly(grid2, color_by=color_key2),
            use_container_width=True,
            key="plotly_2"
        )

with tabs[1]:
    st.subheader("Price + EMA band + entries/exits")
    st.plotly_chart(fig_price_with_trades(df, tr, last_days=last_days), use_container_width=True,
        key="plotly_3"
    )

with tabs[2]:
    st.subheader("Equity curve & drawdown (NET ₹)")
    st.plotly_chart(fig_equity(tr), use_container_width=True,
        key="plotly_4"
    )

with tabs[3]:
    st.subheader("Yearly scaling (NET after costs)")
    st.plotly_chart(fig_yearly(yearly), use_container_width=True,
        key="plotly_5"
    )

with tabs[4]:
    st.subheader("Strategy Visuals")
    left, right = st.columns([1, 1])

    with left:
        st.plotly_chart(fig_equity_cinematic(tr), use_container_width=True,
            key="plotly_6"
        )
        st.plotly_chart(fig_strategy_vs_buyhold(df, tr, float(initial_capital)), use_container_width=True,
            key="plotly_7"
        )
        st.plotly_chart(fig_underwater_cinematic(tr, float(initial_capital)), use_container_width=True,
            key="plotly_8"
        )
        st.plotly_chart(fig_rolling_stats_cinematic(tr, window_trades=int(roll_n)), use_container_width=True,
            key="plotly_9"
        )

    with right:
        st.plotly_chart(fig_monthly_pnl_cinematic(tr), use_container_width=True,
            key="plotly_10"
        )
        st.plotly_chart(fig_monthly_heatmap(tr), use_container_width=True,
            key="plotly_11"
        )
        st.plotly_chart(fig_trade_distribution_cinematic(tr), use_container_width=True,
            key="plotly_12"
        )
        st.plotly_chart(fig_duration_hist_cinematic(tr), use_container_width=True,
            key="plotly_13"
        )

    st.markdown(
        "<div class='subtle-note'> <b>Equity Curve</b>, <b>Strategy vs Buy&Hold</b>, "
        "<b>Monthly Heatmap</b>, <b>Underwater Curve</b>.</div>",
        unsafe_allow_html=True,
    )

with tabs[5]:
    st.subheader("Proof Matrix — One Screen Proof")

    ema_compare = (55, 89)
    with st.spinner("Computing proof grid..."):
        grid2 = compute_ema_tf_rr_grid(
            ema_list=ema_compare,
            lot_=int(lot_size),
            ncon_=int(n_contracts),
            slip_=float(slip),
            fee_=float(fee_rt),
        )

    mode = st.radio(
        "Color cells by",
        ["Return/DD", "Net PnL"],
        horizontal=True,
        index=0,
    )

    color_key = "rdd" if mode.startswith("Return") else "pnl"

    st.plotly_chart(
        fig_proof_matrix_plotly(grid2, color_by=color_key),
        use_container_width=True,
        key="proof_tab_matrix",
    )

    st.markdown(
        "<div class='subtle-note'>Cell text shows: Net PnL, Win%, Max DD, Return/DD.</div>",
        unsafe_allow_html=True,
    )

with tabs[6]:
    st.subheader("Trade log")
    st.dataframe(tr, use_container_width=True)
    st.download_button(
        "Download trades CSV",
        data=tr.to_csv(index=False).encode("utf-8"),
        file_name=f"trades_{tf}_rr_synthfut_costs.csv",
        mime="text/csv",
    )
