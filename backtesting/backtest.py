import os
import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    model_name: str = "linear_regression"   # one of: linear_regression, random_forest, svr, neural_network, xgboost, ensemble
    target: str = "returns"                 # "returns" or "close" (MVP focuses on returns)
    threshold: float | None = None          # trade only if |pred| > threshold; None => auto from costs
    fee_bps: float = 5.0                    # per-turn fee in basis points (bps)
    slippage_bps: float = 2.0               # per-turn slippage in bps
    spread_bps: float = 0.0                # per-turn spread in bps (execution realism)
    allow_short: bool = False               # MVP default: long/flat only
    train_fraction: float = 0.8             # chronological split
    periods_per_year: int = 24 * 365        # hourly candles
    # Walk-forward (rolling) validation
    walk_forward: bool = False
    wf_train_bars: int = 600                # train window length (bars)
    wf_test_bars: int = 200                 # test window length (bars)
    wf_step_bars: int = 200                 # step between windows (bars)
    # Risk controls
    max_leverage: float = 1.0               # cap position magnitude (e.g. 1.0 = 100% notional)
    vol_target_annual: float | None = None  # e.g. 0.5 for 50% annualized; None disables
    vol_lookback: int = 48                  # bars for realized vol estimate (e.g. 2 days on 1h)


def _load_feature_list(project_root: str) -> list[str]:
    meta_path = os.path.join(project_root, "ml_data", "feature_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feats = meta.get("ml_features") or []
    if not feats:
        raise ValueError("ml_features missing in ml_data/feature_metadata.json. Run --analysis first.")
    return feats


def _load_smoothed_df(project_root: str) -> pd.DataFrame:
    from env import smoothed_data
    df = pd.read_csv(smoothed_data)

    # Make sure key numeric columns are numeric
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "num_trades"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse timestamps if present (optional)
    for tcol in ["open_time", "close_time"]:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")

    return df


def _prepare_backtest_frame(project_root: str, feature_cols: list[str]) -> pd.DataFrame:
    df = _load_smoothed_df(project_root)

    # Ensure all feature columns are numeric
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing feature column in smoothed_data: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Targets (no forward fill; avoids pandas FutureWarning)
    df["future_close"] = df["close"].shift(-1)
    df["returns"] = df["close"].pct_change(fill_method=None)
    df["future_returns"] = df["future_close"].pct_change(fill_method=None)

    # Drop rows with NaNs in features/targets we need
    required = ["close"] + feature_cols + ["future_close", "future_returns"]
    df = df.dropna(subset=required).copy()

    # Keep chronological order
    df = df.sort_index()
    return df


def _load_predict_fn(model_name: str):
    # Use each model's predict module so we reuse the saved artifacts consistently.
    mapping = {
        "linear_regression": ("Models.linear_regression.predict", "predict"),
        "random_forest": ("Models.random_forest.predict", "predict"),
        "svr": ("Models.svr.predict", "predict"),
        "neural_network": ("Models.neural_network.predict", "predict"),
        "xgboost": ("Models.xgboost.predict", "predict"),
        "ensemble": ("Models.ensemble.predict", "predict"),
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model_name: {model_name}")
    mod_path, fn = mapping[model_name]
    mod = __import__(mod_path, fromlist=[fn])
    return getattr(mod, fn)


def _signals_from_prediction(pred: np.ndarray, cfg: BacktestConfig) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    thr = cfg.threshold
    if thr is None:
        # Conservative default: require predicted return to exceed ~2x round-trip cost.
        # This reduces over-trading when signal is weak.
        thr = 2.0 * (cfg.fee_bps + cfg.slippage_bps + cfg.spread_bps) / 10000.0
    if cfg.allow_short:
        sig = np.where(pred > thr, 1, np.where(pred < -thr, -1, 0))
    else:
        sig = np.where(pred > thr, 1, 0)
    return sig.astype(int)


def _apply_position_sizing(df: pd.DataFrame, raw_pos: np.ndarray, cfg: BacktestConfig) -> np.ndarray:
    """
    Convert raw {-1,0,1} into sized positions with leverage cap and optional vol targeting.
    """
    pos = raw_pos.astype(float)
    pos = np.clip(pos, -cfg.max_leverage, cfg.max_leverage)

    if cfg.vol_target_annual is None:
        return pos

    # Vol targeting based on realized volatility of returns (using current-bar returns as proxy)
    r = df["returns"].to_numpy(dtype=float)
    vol = pd.Series(r).rolling(cfg.vol_lookback).std(ddof=1).to_numpy()
    # annualization factor
    ann = np.sqrt(cfg.periods_per_year)
    vol_ann = vol * ann
    # avoid div by zero
    scale = np.where((vol_ann > 1e-12) & np.isfinite(vol_ann), cfg.vol_target_annual / vol_ann, 0.0)
    scale = np.clip(scale, 0.0, cfg.max_leverage)  # never exceed max leverage
    return pos * scale


def _compute_strategy_returns(realized: np.ndarray, pos: np.ndarray, cfg: BacktestConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute strategy returns and costs.
    Costs are charged on position changes (turnover) and include:
    - fee_bps + slippage_bps + spread_bps per turn
    """
    prev = np.roll(pos, 1)
    prev[0] = 0.0
    turnover = np.abs(pos - prev)  # 0,1,2,... if leverage used
    cost_per_turn = (cfg.fee_bps + cfg.slippage_bps + cfg.spread_bps) / 10000.0
    costs = turnover * cost_per_turn
    strat_ret = pos * realized - costs
    return strat_ret, costs


def _compute_equity(period_returns: np.ndarray) -> np.ndarray:
    equity = np.empty(len(period_returns) + 1, dtype=float)
    equity[0] = 1.0
    for i, r in enumerate(period_returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    return equity


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def run_backtest(project_root: str, cfg: BacktestConfig, report_prefix: str) -> dict:
    feats = _load_feature_list(project_root)
    df = _prepare_backtest_frame(project_root, feats)

    split_idx = int(len(df) * cfg.train_fraction)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    X_test = df_test[feats].copy()
    predict_fn = _load_predict_fn(cfg.model_name)

    # Predict next-period target
    pred = predict_fn(X_test, target=cfg.target)
    sig = _signals_from_prediction(pred, cfg)
    pos = _apply_position_sizing(df_test, sig, cfg)

    # Realized next-period return for PnL
    realized = df_test["future_returns"].to_numpy(dtype=float)

    strat_ret, costs = _compute_strategy_returns(realized, pos, cfg)
    equity = _compute_equity(strat_ret)

    # Metrics
    total_return = float(equity[-1] - 1.0)
    avg = float(np.mean(strat_ret)) if len(strat_ret) else 0.0
    vol = float(np.std(strat_ret, ddof=1)) if len(strat_ret) > 1 else 0.0
    sharpe = float((avg / vol) * np.sqrt(cfg.periods_per_year)) if vol > 0 else 0.0
    mdd = _max_drawdown(equity)
    hit_rate = float(np.mean(strat_ret > 0)) if len(strat_ret) else 0.0
    exposure = float(np.mean(pos != 0)) if len(pos) else 0.0
    turns = int(np.sum(np.abs(np.diff(np.r_[0.0, pos])) > 0))
    avg_cost = float(np.mean(costs)) if len(costs) else 0.0

    # Build a simple trade log (flip-based)
    tcol = "open_time" if "open_time" in df_test.columns else None
    times = df_test[tcol].astype(str).tolist() if tcol else [str(i) for i in range(len(df_test))]

    trades = []
    current = 0
    entry_idx = None
    entry_time = None
    entry_price = None
    for i in range(len(pos)):
        if current == 0 and pos[i] != 0:
            current = int(pos[i])
            entry_idx = i
            entry_time = times[i]
            entry_price = float(df_test["close"].iloc[i]) if "close" in df_test.columns else np.nan
        elif current != 0 and pos[i] != current:
            # exit current trade at i (before flip)
            exit_time = times[i]
            exit_price = float(df_test["close"].iloc[i]) if "close" in df_test.columns else np.nan
            pnl = float(np.prod(1.0 + strat_ret[entry_idx:i]) - 1.0) if entry_idx is not None and i > entry_idx else 0.0
            trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": "LONG" if current == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl * 100.0,
                "bars_held": (i - entry_idx) if entry_idx is not None else 0,
            })
            # start new trade if new pos non-zero
            if pos[i] != 0:
                current = int(pos[i])
                entry_idx = i
                entry_time = times[i]
                entry_price = float(df_test["close"].iloc[i]) if "close" in df_test.columns else np.nan
            else:
                current = 0
                entry_idx = None
                entry_time = None
                entry_price = None

    trades_df = pd.DataFrame(trades)

    # Save artifacts
    reports_dir = os.path.dirname(report_prefix)
    os.makedirs(reports_dir, exist_ok=True)

    trades_csv = report_prefix + "_trades.csv"
    trades_df.to_csv(trades_csv, index=False)

    # Equity curve plot
    equity_png = report_prefix + "_equity.png"
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(equity, linewidth=1.5)
        ax.set_title(f"Equity Curve - {cfg.model_name} ({cfg.target})")
        ax.set_xlabel("Bars")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(equity_png, dpi=200)
        plt.close(fig)
    except Exception:
        equity_png = ""

    # Buy-and-hold baseline on the same test slice (no costs)
    bh_ret = realized
    bh_equity = _compute_equity(bh_ret)
    bh_total = float(bh_equity[-1] - 1.0) * 100.0
    bh_mdd = _max_drawdown(bh_equity) * 100.0

    summary = {
        "model": cfg.model_name,
        "target": cfg.target,
        "threshold": cfg.threshold,
        "threshold_auto": (cfg.threshold is None),
        "fee_bps": cfg.fee_bps,
        "slippage_bps": cfg.slippage_bps,
        "spread_bps": cfg.spread_bps,
        "allow_short": cfg.allow_short,
        "max_leverage": cfg.max_leverage,
        "vol_target_annual": cfg.vol_target_annual,
        "n_test_bars": int(len(df_test)),
        "turns": turns,
        "exposure": exposure,
        "total_return_pct": total_return * 100.0,
        "max_drawdown_pct": mdd * 100.0,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "avg_cost_per_bar_pct": avg_cost * 100.0,
        "buy_hold_total_return_pct": bh_total,
        "buy_hold_max_drawdown_pct": bh_mdd,
    }

    with open(report_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, indent=2)

    return {
        "summary": summary,
        "equity_png": equity_png,
        "trades_csv": trades_csv,
        "trades": trades_df,
        "equity": equity,
    }


def run_walk_forward_backtest(project_root: str, cfg: BacktestConfig, report_prefix: str) -> dict:
    """
    Walk-forward / rolling window backtest.

    Notes:
    - This uses *pre-trained* model artifacts for predictions (same as run_backtest).
    - It evaluates multiple sequential test windows to measure stability by period/regime.
    """
    feats = _load_feature_list(project_root)
    df = _prepare_backtest_frame(project_root, feats)

    predict_fn = _load_predict_fn(cfg.model_name)

    windows = []
    all_strat_ret = []
    all_equity = [1.0]

    start = 0
    while True:
        train_end = start + cfg.wf_train_bars
        test_end = train_end + cfg.wf_test_bars
        if test_end > len(df):
            break

        df_test = df.iloc[train_end:test_end].copy()
        X_test = df_test[feats].copy()

        pred = predict_fn(X_test, target=cfg.target)
        sig = _signals_from_prediction(pred, cfg)
        pos = _apply_position_sizing(df_test, sig, cfg)
        realized = df_test["future_returns"].to_numpy(dtype=float)
        strat_ret, costs = _compute_strategy_returns(realized, pos, cfg)

        equity = _compute_equity(strat_ret)
        total_return = float(equity[-1] - 1.0)
        avg = float(np.mean(strat_ret)) if len(strat_ret) else 0.0
        vol = float(np.std(strat_ret, ddof=1)) if len(strat_ret) > 1 else 0.0
        sharpe = float((avg / vol) * np.sqrt(cfg.periods_per_year)) if vol > 0 else 0.0
        mdd = _max_drawdown(equity)
        hit_rate = float(np.mean(strat_ret > 0)) if len(strat_ret) else 0.0
        exposure = float(np.mean(pos != 0)) if len(pos) else 0.0
        turns = int(np.sum(np.abs(np.diff(np.r_[0.0, pos])) > 0))
        avg_cost = float(np.mean(costs)) if len(costs) else 0.0

        windows.append({
            "window_idx": len(windows),
            "train_start": int(start),
            "train_end": int(train_end),
            "test_start": int(train_end),
            "test_end": int(test_end),
            "test_bars": int(len(df_test)),
            "total_return_pct": total_return * 100.0,
            "max_drawdown_pct": mdd * 100.0,
            "sharpe": sharpe,
            "hit_rate": hit_rate,
            "exposure": exposure,
            "turns": turns,
            "avg_cost_per_bar_pct": avg_cost * 100.0,
        })

        all_strat_ret.append(strat_ret)

        # stitch equity by compounding from prior
        last = all_equity[-1]
        for r in strat_ret:
            last = last * (1.0 + float(r))
            all_equity.append(last)

        start += cfg.wf_step_bars

    if not windows:
        raise ValueError("Walk-forward produced 0 windows. Reduce wf_* bars or increase dataset size.")

    all_strat_ret = np.concatenate(all_strat_ret) if all_strat_ret else np.array([], dtype=float)
    all_equity = np.asarray(all_equity, dtype=float)

    total_return = float(all_equity[-1] - 1.0)
    avg = float(np.mean(all_strat_ret)) if len(all_strat_ret) else 0.0
    vol = float(np.std(all_strat_ret, ddof=1)) if len(all_strat_ret) > 1 else 0.0
    sharpe = float((avg / vol) * np.sqrt(cfg.periods_per_year)) if vol > 0 else 0.0
    mdd = _max_drawdown(all_equity)

    windows_df = pd.DataFrame(windows)
    windows_csv = report_prefix + "_windows.csv"
    windows_df.to_csv(windows_csv, index=False)

    equity_png = report_prefix + "_equity.png"
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(all_equity, linewidth=1.5)
        ax.set_title(f"Walk-Forward Equity - {cfg.model_name} ({cfg.target})")
        ax.set_xlabel("Bars (stitched)")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(equity_png, dpi=200)
        plt.close(fig)
    except Exception:
        equity_png = ""

    # Buy-and-hold baseline over the stitched test periods (approx)
    # Use concatenated realized returns from windows (same as all_strat_ret but without position/cost).
    # We approximate by re-running over windows_df is too expensive; so store baseline as NaN for now.
    summary = {
        "mode": "walk_forward",
        "model": cfg.model_name,
        "target": cfg.target,
        "threshold": cfg.threshold,
        "threshold_auto": (cfg.threshold is None),
        "fee_bps": cfg.fee_bps,
        "slippage_bps": cfg.slippage_bps,
        "spread_bps": cfg.spread_bps,
        "allow_short": cfg.allow_short,
        "max_leverage": cfg.max_leverage,
        "vol_target_annual": cfg.vol_target_annual,
        "wf_train_bars": cfg.wf_train_bars,
        "wf_test_bars": cfg.wf_test_bars,
        "wf_step_bars": cfg.wf_step_bars,
        "windows": int(len(windows)),
        "total_return_pct": total_return * 100.0,
        "max_drawdown_pct": mdd * 100.0,
        "sharpe": sharpe,
        "window_return_pct_mean": float(windows_df["total_return_pct"].mean()),
        "window_return_pct_std": float(windows_df["total_return_pct"].std(ddof=1)) if len(windows_df) > 1 else 0.0,
    }

    with open(report_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "windows": windows}, f, indent=2)

    return {
        "summary": summary,
        "equity_png": equity_png,
        "windows_csv": windows_csv,
        "windows": windows_df,
        "equity": all_equity,
        "trades": pd.DataFrame(),  # walk-forward MVP: per-window stats first
        "trades_csv": "",
    }


def write_backtest_reports(result: dict, report_prefix: str):
    """
    Writes:
    - PDF: clean + readable
    - HTML: interactive trades table
    - TXT: quick summary
    """
    summary = result["summary"]
    trades_df: pd.DataFrame = result.get("trades", pd.DataFrame())
    windows_df: pd.DataFrame = result.get("windows", pd.DataFrame())
    equity_png = result.get("equity_png", "")

    # TXT
    txt_path = report_prefix + ".txt"
    lines = [
        "Backtest Report (MVP)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SUMMARY",
        f"Model: {summary['model']}",
        f"Target: {summary['target']}",
        f"Allow short: {summary['allow_short']}",
        f"Threshold: {summary['threshold']}",
        f"Fee bps: {summary.get('fee_bps')}",
        f"Slippage bps: {summary.get('slippage_bps')}",
        f"Spread bps: {summary.get('spread_bps', 0.0)}",
        f"Max leverage: {summary.get('max_leverage', 1.0)}",
        f"Vol target annual: {summary.get('vol_target_annual')}",
        "",
        f"Mode: {summary.get('mode', 'single_split')}",
        f"Test bars: {summary.get('n_test_bars', '')}",
        f"Windows: {summary.get('windows', '')}",
        f"Turns: {summary.get('turns', '')}",
        f"Exposure: {float(summary.get('exposure', 0.0)):.2%}" if 'exposure' in summary else "Exposure: ",
        f"Total return: {summary['total_return_pct']:.2f}%",
        f"Max drawdown: {summary['max_drawdown_pct']:.2f}%",
        f"Sharpe (approx): {summary['sharpe']:.3f}",
        f"Hit rate: {float(summary.get('hit_rate', 0.0)):.2%}" if 'hit_rate' in summary else "Hit rate: ",
        "",
        f"Trades saved to: {os.path.basename(report_prefix)}_trades.csv" if result.get("trades_csv") else "Trades saved to: (not generated)",
        f"Window stats saved to: {os.path.basename(report_prefix)}_windows.csv" if result.get("windows_csv") else "Window stats saved to: (not generated)",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # HTML (interactive trades table via DataTables CDN)
    html_path = report_prefix + ".html"
    html_rows = ""
    if not trades_df.empty:
        for _, r in trades_df.iterrows():
            html_rows += (
                "<tr>"
                f"<td>{r.get('entry_time','')}</td>"
                f"<td>{r.get('exit_time','')}</td>"
                f"<td>{r.get('side','')}</td>"
                f"<td>{r.get('entry_price','')}</td>"
                f"<td>{r.get('exit_price','')}</td>"
                f"<td>{float(r.get('pnl_pct',0.0)):.3f}</td>"
                f"<td>{int(r.get('bars_held',0))}</td>"
                "</tr>"
            )
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Backtest Report</title>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .muted {{ color: #666; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    h1 {{ margin-bottom: 6px; }}
  </style>
</head>
<body>
  <h1>Backtest Report</h1>
  <div class="muted">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

  <div class="grid" style="margin-top:16px;">
    <div class="card">
      <h2>Summary</h2>
      <ul>
        <li><b>Model</b>: {summary['model']}</li>
        <li><b>Target</b>: {summary['target']}</li>
        <li><b>Total return</b>: {summary['total_return_pct']:.2f}%</li>
        <li><b>Max drawdown</b>: {summary['max_drawdown_pct']:.2f}%</li>
        <li><b>Sharpe</b>: {summary['sharpe']:.3f}</li>
        <li><b>Exposure</b>: {float(summary.get('exposure', 0.0)):.2%}</li>
        <li><b>Turns</b>: {int(summary.get('turns', 0))}</li>
      </ul>
    </div>
    <div class="card">
      <h2>Equity Curve</h2>
      {f"<img src='{os.path.basename(equity_png)}' style='max-width:100%;' />" if equity_png else "<div class='muted'>No plot generated</div>"}
    </div>
  </div>

  <h2 style="margin-top:20px;">Trades</h2>
  <table id="trades" class="display" style="width:100%">
    <thead>
      <tr><th>Entry</th><th>Exit</th><th>Side</th><th>Entry Px</th><th>Exit Px</th><th>PnL %</th><th>Bars</th></tr>
    </thead>
    <tbody>
      {html_rows}
    </tbody>
  </table>

  <h2 style="margin-top:20px;">Walk-forward windows</h2>
  <table id="windows" class="display" style="width:100%">
    <thead>
      <tr><th>Window</th><th>Train (start-end)</th><th>Test (start-end)</th><th>Return %</th><th>Max DD %</th><th>Sharpe</th><th>Turns</th></tr>
    </thead>
    <tbody>
      {"" if windows_df.empty else "".join(
        f"<tr><td>{int(w.window_idx)}</td>"
        f"<td>{int(w.train_start)}-{int(w.train_end)}</td>"
        f"<td>{int(w.test_start)}-{int(w.test_end)}</td>"
        f"<td>{float(w.total_return_pct):.2f}</td>"
        f"<td>{float(w.max_drawdown_pct):.2f}</td>"
        f"<td>{float(w.sharpe):.3f}</td>"
        f"<td>{int(w.turns)}</td></tr>"
        for _, w in windows_df.iterrows()
      )}
    </tbody>
  </table>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script>
    $(document).ready(function() {{
      $('#trades').DataTable({{ pageLength: 25 }});
      $('#windows').DataTable({{ pageLength: 25 }});
    }});
  </script>
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    # PDF (matplotlib)
    pdf_path = report_prefix + ".pdf"
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        with PdfPages(pdf_path) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            ax.text(0.5, 0.8, "Backtest Report", ha="center", va="center", fontsize=22)
            ax.text(0.5, 0.75, f"Model: {summary['model']} ({summary['target']})", ha="center", va="center", fontsize=12)
            ax.text(0.5, 0.72, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha="center", va="center", fontsize=10, color="#555")
            ax.text(0.08, 0.62, f"Total return: {summary['total_return_pct']:.2f}%", fontsize=11)
            ax.text(0.08, 0.59, f"Max drawdown: {summary['max_drawdown_pct']:.2f}%", fontsize=11)
            ax.text(0.08, 0.56, f"Sharpe: {summary['sharpe']:.3f}", fontsize=11)
            ax.text(0.08, 0.53, f"Exposure: {float(summary.get('exposure', 0.0)):.2%}", fontsize=11)
            ax.text(0.08, 0.50, f"Turns: {int(summary.get('turns', 0))}", fontsize=11)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Equity curve
            equity = result["equity"]
            fig, ax = plt.subplots(figsize=(11.69, 4.0))
            ax.plot(equity, linewidth=1.5)
            ax.set_title("Equity Curve")
            ax.set_xlabel("Bars")
            ax.set_ylabel("Equity")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Trades table (first page)
            if not trades_df.empty:
                page = trades_df.head(35).copy()
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.axis("off")
                ax.set_title("Trades (first 35)", fontsize=14, pad=20)
                col_labels = ["entry_time", "exit_time", "side", "pnl_pct", "bars_held"]
                cell_text = [
                    [str(r.get("entry_time","")), str(r.get("exit_time","")), str(r.get("side","")),
                     f"{float(r.get('pnl_pct',0.0)):.3f}", str(int(r.get("bars_held",0)))]
                    for _, r in page.iterrows()
                ]
                tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                tbl.scale(1, 1.2)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    except Exception as e:
        with open(pdf_path + ".error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    return {"pdf": pdf_path, "html": html_path, "txt": txt_path}

