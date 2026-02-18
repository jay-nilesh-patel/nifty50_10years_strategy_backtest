# Run:
#   pip install streamlit pandas numpy plotly
#   streamlit run app.py

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st



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

# One-click data loading
DATA_DIR = Path(__file__).parent / "data"

TIMEFRAME_FILES = {
    "5m":  "NIFTY 50_5minute.csv",
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
    df = pd.read_csv(path)
    return df

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
        name="",
        showlegend=False,
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
        name="",
        showlegend=False,
        line=dict(width=6, color="rgba(122,162,255,0.18)"),
        hoverinfo="skip",
    ))

    if len(trw):
        longs = trw[trw["side"] == 1]
        shorts = trw[trw["side"] == -1]

        if len(longs):
            fig.add_trace(go.Scatter(
                x=longs["entry_time"], y=longs["entry_price"],
                mode="markers",
                name="Long entry",
                marker=dict(symbol="triangle-up", size=12, color=GREEN, line=dict(width=1, color="rgba(0,0,0,0.35)")),
                hovertemplate="Long entry<br>%{x}<br>%{y:.2f}<extra></extra>",
            ))
        if len(shorts):
            fig.add_trace(go.Scatter(
                x=shorts["entry_time"], y=shorts["entry_price"],
                mode="markers",
                name="Short entry",
                marker=dict(symbol="triangle-down", size=12, color=RED, line=dict(width=1, color="rgba(0,0,0,0.35)")),
                hovertemplate="Short entry<br>%{x}<br>%{y:.2f}<extra></extra>",
            ))

        fig.add_trace(go.Scatter(
            x=trw["exit_time"], y=trw["exit_price"],
            mode="markers",
            name="Exit",
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
        mode="lines",
        name="Equity (NET ₹)",
        line=dict(width=2.2, color=ACCENT),
        hovertemplate="%{x}<br><b>₹%{y:,.0f}</b><extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tr["exit_time"], y=tr["cum_net_rupees"],
        mode="lines",
        name="",
        showlegend=False,
        line=dict(width=8, color=ACCENT_SOFT),
        hoverinfo="skip",
    ), row=1, col=1)

    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.12)", row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tr["exit_time"], y=tr["dd_net_rupees"],
        mode="lines",
        name="Drawdown (₹)",
        line=dict(width=2.0, color=RED),
        fill="tozeroy",
        fillcolor=RED_SOFT,
        hovertemplate="%{x}<br><b>₹%{y:,.0f}</b><extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        xaxis=dict(rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ])
        ))
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
        mode="lines+markers",
        name="Win rate (%)",
        line=dict(width=2.1, color=ACCENT),
        marker=dict(size=7, color=ACCENT),
        hovertemplate="Year %{x}<br><b>%{y:.2f}%</b><extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=y["year"], y=y["trades"],
        mode="lines+markers",
        name="Trades",
        line=dict(width=1.7, dash="dot", color="rgba(245,247,255,0.65)"),
        marker=dict(size=6, color="rgba(245,247,255,0.75)"),
        hovertemplate="Year %{x}<br><b>%{y}</b> trades<extra></extra>",
    ), secondary_y=True)

    fig.update_yaxes(title_text="Net PnL (₹)", secondary_y=False)
    fig.update_yaxes(title_text="Win rate (%) / Trades", secondary_y=True)
    return apply_glass_theme(fig, title="Yearly scaling (NET after costs)")

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

            if overall_ is None or overall_.empty:
                rows.append({
                    "tf": tf_,
                    "rr": rr_,
                    "trades": 0,
                    "win_rate_net": np.nan,
                    "total_net_rupees": 0.0,
                })
            else:
                om_ = overall_.iloc[0].to_dict()
                rows.append({
                    "tf": tf_,
                    "rr": rr_,
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
                row_txt.append(f"₹{p:,.0f}<br>{100*a:.1f}%")
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
        zmin=zmin,
        zmax=zmax,
        hovertemplate="TF %{y}<br>%{x}<br>%{text}<extra></extra>",
        colorbar=dict(
            title=dict(text=cb_title, font=dict(color=MUTED)),  # <-- FIXED (no titlefont)
            tickfont=dict(color=MUTED),
            len=0.85,
        ),
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(side="top"),
    )
    return apply_glass_theme(fig, title=title)

# Streamlit UI
st.set_page_config(page_title="NIFTY EMA (Glass)", layout="wide")

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
    start_t = START_T
    last_entry_t = LAST_ENTRY_T
    forced_exit_t = FORCED_EXIT_BY_TF.get(tf, time(15, 10))
    st.write(f"Entry: {start_t} → {last_entry_t}")
    st.write(f"Forced exit: {forced_exit_t} (bar close)")

    st.header("Sizing + costs")
    lot_size = st.number_input("Lot size", min_value=1, value=65, step=1)
    n_contracts = st.number_input("Contracts", min_value=1, value=20, step=1)
    slip = st.number_input("Slippage (pts/side)", min_value=0.0, value=0.25, step=0.05, format="%.2f")
    fee_rt = st.number_input("Fees (₹/round-trip total)", min_value=0.0, value=40.0, step=5.0)

    st.header("Visuals")
    last_days = st.slider("Price window (days)", 5, 60, 12, 1)

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

kpi(k1, "Total NET (₹)", f"{om['total_net_rupees']:,.0f}", f"TF: {tf} | Contracts: {int(n_contracts)} | Lot: {int(lot_size)}")
kpi(k2, "Max DD NET (₹)", f"{om['max_dd_net_rupees']:,.0f}", f"Slip: {slip:.2f} pts/side | Fees: ₹{fee_rt:.0f}/RT")
kpi(k3, "Profit factor (NET)", f"{om['profit_factor_net']:.3f}", f"Trades: {int(om['trades'])}")
kpi(k4, "Win rate (NET)", f"{100*om['win_rate_net']:.2f}%", f"RR: {rr_mult:g}")

tabs = st.tabs(["Overview", "Price + Trades", "Equity", "Yearly", "Trades"])

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
        st.plotly_chart(fig_tf_rr_heatmap(grid, color_metric=metric_key), use_container_width=True)

        st.markdown(
            "<div class='subtle-note'>Each cell shows: Net PnL (₹) and accuracy (NET win rate). "
            "Costs/slippage/sizing match the sidebar settings.</div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.subheader("Yearly summary")
        st.dataframe(yearly, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Price + EMA band + entries/exits")
    st.plotly_chart(fig_price_with_trades(df, tr, last_days=last_days), use_container_width=True)

with tabs[2]:
    st.subheader("Equity curve & drawdown (NET ₹)")
    st.plotly_chart(fig_equity(tr), use_container_width=True)

with tabs[3]:
    st.subheader("Yearly scaling (NET after costs)")
    st.plotly_chart(fig_yearly(yearly), use_container_width=True)

with tabs[4]:
    st.subheader("Trade log")
    st.dataframe(tr, use_container_width=True)
    st.download_button(
        "Download trades CSV",
        data=tr.to_csv(index=False).encode("utf-8"),
        file_name=f"trades_{tf}_rr_synthfut_costs.csv",
        mime="text/csv",
    )
