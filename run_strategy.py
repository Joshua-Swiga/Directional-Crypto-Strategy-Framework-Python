#!/usr/bin/env python3
"""
Central Entry Point — Trading Strategy Pipeline

Run the full analysis pipeline, train models, evaluate, and generate reports.

Usage:
  python run_strategy.py                    # Full pipeline (analysis + train + evaluate + report)
  python run_strategy.py --analysis         # Exploratory analysis & correlation only
  python run_strategy.py --train            # Train all models only
  python run_strategy.py --evaluate         # Evaluate all models only
  python run_strategy.py --report           # Generate evaluation report only
  python run_strategy.py --train --models linear_regression random_forest  # Train specific models
  python run_strategy.py --skip-analysis --train  # Skip analysis, train models (uses existing ml_data)
"""
import argparse
import os
import sys
from datetime import datetime
import shutil
import json

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Report output directory
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Charts output directory (from analysis)
CHARTS_DIR = os.path.join(PROJECT_ROOT, 'correlation_images')
os.makedirs(CHARTS_DIR, exist_ok=True)


def clean_run_outputs():
    """
    One-run policy:
    - Delete all prior reports in `reports/`
    - Delete all prior charts in `correlation_images/`

    This keeps the workspace focused on the single run you are currently dealing with.
    """
    for d in [REPORTS_DIR, CHARTS_DIR]:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            path = os.path.join(d, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
            except Exception:
                # Best-effort cleanup; don't fail the run for locked files.
                pass


def run_analysis():
    """Run exploratory analysis and correlation (creates ml_data)."""
    print("\n" + "=" * 80)
    print("STEP 1: EXPLORATORY ANALYSIS & CORRELATION")
    print("=" * 80)
    import Exploratory_Analysis_Correlation
    # The module runs on import
    print("OK: Analysis complete. ML data saved to ml_data/")


def run_data_pipeline():
    """Run data collection and feature derivation."""
    print("\n" + "=" * 80)
    print("STEP 0: DATA COLLECTION & FEATURE DERIVATION")
    print("=" * 80)
    from getting_data import get_data_from_binance
    from data_collection_and_cleaning import deriving_additional_features
    get_data_from_binance()
    deriving_additional_features()
    print("OK: Data pipeline complete.")


def print_final_trade_plan(model_name: str, threshold: float, allow_short: bool, risk_usdt: float, account_usdt: float,
                           sl_atr_mult: float, tp_atr_mult: float):
    """
    Final actionable output:
    based on the last timeframe -> enter with lot, time, SL/TP, and interval.
    """
    import pandas as pd
    import numpy as np
    from env import smoothed_data, INTERVAL

    # Load feature metadata (strategy reasoning)
    meta_path = os.path.join(PROJECT_ROOT, "ml_data", "feature_metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    feats_meta = meta.get("ml_features") or []
    strong = meta.get("strong_predictors_returns") or []
    val = meta.get("validation_metrics", {}).get("returns", {}).get("test", {})

    # Load the trained model to discover the *exact* feature order it expects
    # (prevents sklearn feature_names mismatch).
    feats = feats_meta
    try:
        from Models.utils import load_model
        models_root = os.path.join(PROJECT_ROOT, "Models") if os.path.exists(os.path.join(PROJECT_ROOT, "Models")) else os.path.join(PROJECT_ROOT, "models")
        model_dir = os.path.join(models_root, model_name)
        artifact_name = f"{model_name}_returns"
        model_obj, _ = load_model(model_dir, artifact_name)
        if hasattr(model_obj, "scaler") and hasattr(model_obj.scaler, "feature_names_in_"):
            feats = list(model_obj.scaler.feature_names_in_)
    except Exception:
        # Fallback to metadata ordering
        feats = feats_meta

    df = pd.read_csv(smoothed_data)
    for c in list(set(feats + ["close", "atr"])):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Take the latest row with non-null features
    if not feats:
        print("\nFINAL TRADE PLAN: no feature list available. Run --analysis and --train first.")
        return
    df_feat = df.dropna(subset=feats + ["close", "atr"]).copy()
    if df_feat.empty:
        print("\nFINAL TRADE PLAN: not enough feature history yet (rolling indicators need warm-up).")
        return
    last = df_feat.iloc[-1]

    # Predict next-bar return
    from backtesting.backtest import _load_predict_fn  # reuse mapping
    predict_fn = _load_predict_fn(model_name)
    # IMPORTANT: feature order must match training order exactly
    X_last = pd.DataFrame([[float(last[f]) for f in feats]], columns=feats)
    try:
        pred_ret = float(predict_fn(X_last, target="returns")[0])
    except FileNotFoundError:
        print("\nFINAL TRADE PLAN: model artifacts not found. Train the model first (e.g. --train --models linear_regression).")
        return
    except Exception as e:
        print(f"\nFINAL TRADE PLAN: could not generate prediction: {e}")
        return

    # Decide direction
    if allow_short:
        side = "LONG" if pred_ret > threshold else "SHORT" if pred_ret < -threshold else "FLAT"
    else:
        side = "LONG" if pred_ret > threshold else "FLAT"

    entry_time = str(last.get("open_time", ""))
    entry_price = float(last["close"])
    atr = float(last["atr"])

    # Risk-based sizing (simple): risk_usdt = account_usdt * risk_pct
    if risk_usdt <= 0:
        risk_usdt = account_usdt * 0.01

    if side == "LONG":
        stop_loss = entry_price - sl_atr_mult * atr
        take_profit = entry_price + tp_atr_mult * atr
        per_unit_risk = max(entry_price - stop_loss, 1e-9)
    elif side == "SHORT":
        stop_loss = entry_price + sl_atr_mult * atr
        take_profit = entry_price - tp_atr_mult * atr
        per_unit_risk = max(stop_loss - entry_price, 1e-9)
    else:
        stop_loss = float("nan")
        take_profit = float("nan")
        per_unit_risk = 1.0

    # BTC lot size approximation (USDT risk / price move)
    lot_btc = float(risk_usdt / per_unit_risk) if side != "FLAT" else 0.0

    print("\n" + "=" * 80)
    print("FINAL TRADE PLAN (MVP)")
    print("=" * 80)
    print(f"Timeframe (stick lifespan): {INTERVAL}")
    print(f"Based on last candle time: {entry_time}")
    print(f"Model used: {model_name}")
    print(f"Predicted next-bar return: {pred_ret:.6f} ({pred_ret*100:.4f}%)")
    print(f"Decision rule: LONG if pred > {threshold}, SHORT if pred < {-threshold} (allow_short={allow_short})")
    print("")
    print(f"Enter: {side}")
    print(f"Entry price (approx): {entry_price:.2f}")
    if side != "FLAT":
        print(f"Lot size (BTC, risk-based): {lot_btc:.6f}")
        print(f"Stop loss at: {stop_loss:.2f} (ATR x {sl_atr_mult})")
        print(f"Take profit at: {take_profit:.2f} (ATR x {tp_atr_mult})")
    else:
        print("No trade. Prediction does not exceed threshold.")

    print("\nWHY (high-signal reasoning):")
    if val:
        print(f"- Out-of-sample returns accuracy: R2={val.get('r2', 0):.4f}, RMSE={val.get('rmse', 0):.6f}")
    if strong:
        print(f"- Strong return predictors used/validated in this feature set: {', '.join(strong)}")
    print("- Features were selected by correlation screening + redundancy removal + out-of-sample validation.")
    print("- Close prediction R2 ~ 1.0 is mostly persistence; returns is the relevant target for trading entries.")
    print("")


def train_models(model_names=None):
    """Train specified models (or all if None)."""
    # Support both 'models' and 'Models' folder names
    mod_prefix = 'Models' if os.path.exists(os.path.join(PROJECT_ROOT, 'Models')) else 'models'
    models_config = {
        'linear_regression': (f'{mod_prefix}.linear_regression.train', 'train_model'),
        'random_forest': (f'{mod_prefix}.random_forest.train', 'train_model'),
        'xgboost': (f'{mod_prefix}.xgboost.train', 'train_model'),
        'svr': (f'{mod_prefix}.svr.train', 'train_model'),
        'neural_network': (f'{mod_prefix}.neural_network.train', 'train_model'),
        'arima': (f'{mod_prefix}.arima.train', 'train_model'),
        'ensemble': (f'{mod_prefix}.ensemble.train', 'train_ensemble'),
    }
    
    to_run = model_names or list(models_config.keys())
    results = {}
    
    for name in to_run:
        if name not in models_config:
            print(f"⚠ Unknown model: {name}. Skipping.")
            continue
        mod_path, func_name = models_config[name]
        print(f"\n--- Training {name} ---")
        try:
            mod = __import__(mod_path, fromlist=[func_name])
            func = getattr(mod, func_name)
            result = func()
            results[name] = {'status': 'ok', 'result': result}
            print(f"OK: {name} trained successfully")
        except Exception as e:
            results[name] = {'status': 'error', 'error': str(e)}
            print(f"ERROR: {name} failed: {e}")
    
    return results


def evaluate_models(model_names=None):
    """Evaluate specified models (or all if None)."""
    mod_prefix = 'Models' if os.path.exists(os.path.join(PROJECT_ROOT, 'Models')) else 'models'
    models_config = {
        'linear_regression': (f'{mod_prefix}.linear_regression.evaluate', 'evaluate'),
        'random_forest': (f'{mod_prefix}.random_forest.evaluate', 'evaluate'),
        'xgboost': (f'{mod_prefix}.xgboost.evaluate', 'evaluate'),
        'svr': (f'{mod_prefix}.svr.evaluate', 'evaluate'),
        'neural_network': (f'{mod_prefix}.neural_network.evaluate', 'evaluate'),
        'arima': (f'{mod_prefix}.arima.evaluate', 'evaluate'),
        'ensemble': (f'{mod_prefix}.ensemble.evaluate', 'evaluate'),
    }
    
    to_run = model_names or list(models_config.keys())
    all_metrics = {}
    
    for name in to_run:
        if name not in models_config:
            continue
        mod_path, func_name = models_config[name]
        try:
            mod = __import__(mod_path, fromlist=[func_name])
            func = getattr(mod, func_name)
            metrics = func()
            all_metrics[name] = metrics
        except Exception as e:
            all_metrics[name] = {'error': str(e)}
    
    return all_metrics


def generate_report(all_metrics, output_path=None):
    """
    Generate evaluation report in:
    - PDF (clean + readable)
    - HTML (interactive/sortable tables)
    - TXT + MD (plain artifacts)

    Notes:
    - PDFs are not truly interactive; interactivity is provided via HTML.
    """
    if output_path is None:
        # One-run policy: always overwrite "latest" artifacts.
        output_path = os.path.join(REPORTS_DIR, 'evaluation_report_latest')
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Normalize metrics shape and build a summary table
    rows = []
    errors = []
    for model_name, metrics in all_metrics.items():
        if isinstance(metrics, dict) and 'error' in metrics:
            errors.append((model_name, metrics['error']))
            continue
        if isinstance(metrics, dict) and 'rmse' in metrics and 'close' not in metrics:
            metrics = {'close': metrics}
        for target in ['close', 'returns']:
            m = (metrics or {}).get(target)
            if not isinstance(m, dict) or 'rmse' not in m:
                continue
            rows.append({
                'model': model_name,
                'target': target,
                'rmse': float(m.get('rmse', 0)),
                'mae': float(m.get('mae', 0)),
                'r2': float(m.get('r2', 0)),
                'mape_pct': float(m.get('mape', 0)),
            })

    # Text/markdown content (ASCII only)
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append("Trading Strategy - Model Evaluation Report")
    lines.append(f"Generated: {generated_at}")
    lines.append("")

    if rows:
        # Create a deterministic, readable summary section
        lines.append("SUMMARY")
        lines.append("Model | Target | RMSE | MAE | R2 | MAPE%")
        lines.append("----- | ------ | ---- | --- | -- | -----")
        for r in sorted(rows, key=lambda x: (x['model'], x['target'])):
            lines.append(
                f"{r['model']} | {r['target']} | {r['rmse']:.6f} | {r['mae']:.6f} | {r['r2']:.4f} | {r['mape_pct']:.2f}"
            )
        lines.append("")

    if errors:
        lines.append("ERRORS")
        for model_name, err in errors:
            lines.append(f"- {model_name}: {err}")
        lines.append("")

    report_text = "\n".join(lines)

    report_md = output_path + '.md'
    report_txt_path = output_path + '.txt'
    report_html = output_path + '.html'
    report_pdf = output_path + '.pdf'
    report_json = output_path + '.json'

    with open(report_md, 'w', encoding='utf-8') as f:
        f.write(report_text)
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump({'generated_at': generated_at, 'rows': rows, 'errors': errors}, f, indent=2)

    # HTML report (interactive tables via DataTables CDN)
    # If you want zero-external-deps, we can vendor JS/CSS locally later.
    html_rows = "".join(
        f"<tr><td>{r['model']}</td><td>{r['target']}</td>"
        f"<td>{r['rmse']:.6f}</td><td>{r['mae']:.6f}</td>"
        f"<td>{r['r2']:.4f}</td><td>{r['mape_pct']:.2f}</td></tr>"
        for r in sorted(rows, key=lambda x: (x['model'], x['target']))
    )
    html_errors = "".join(f"<li><b>{m}</b>: {e}</li>" for m, e in errors)
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Trading Strategy - Model Evaluation Report</title>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>Trading Strategy - Model Evaluation Report</h1>
  <div class="muted">Generated: {generated_at}</div>

  <h2>Summary</h2>
  <table id="summary" class="display" style="width:100%">
    <thead>
      <tr><th>Model</th><th>Target</th><th>RMSE</th><th>MAE</th><th>R2</th><th>MAPE %</th></tr>
    </thead>
    <tbody>
      {html_rows}
    </tbody>
  </table>

  <h2>Errors</h2>
  <ul>{html_errors or '<li>None</li>'}</ul>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script>
    $(document).ready(function() {{
      $('#summary').DataTable({{ pageLength: 25 }});
    }});
  </script>
</body>
</html>"""

    with open(report_html, 'w', encoding='utf-8') as f:
        f.write(html_doc)

    # PDF report (clean + printable)
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        def add_table_page(pdf, title, table_rows):
            fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape-ish
            ax.axis('off')
            ax.set_title(title, fontsize=16, pad=20)
            col_labels = ['Model', 'Target', 'RMSE', 'MAE', 'R2', 'MAPE %']
            cell_text = [[
                r['model'], r['target'],
                f"{r['rmse']:.6f}", f"{r['mae']:.6f}", f"{r['r2']:.4f}", f"{r['mape_pct']:.2f}"
            ] for r in table_rows]
            tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.2)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        with PdfPages(report_pdf) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
            ax.axis('off')
            ax.text(0.5, 0.8, 'Trading Strategy', ha='center', va='center', fontsize=24)
            ax.text(0.5, 0.75, 'Model Evaluation Report', ha='center', va='center', fontsize=18)
            ax.text(0.5, 0.7, f'Generated: {generated_at}', ha='center', va='center', fontsize=11, color='#555')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Summary table pages (paginate)
            sorted_rows = sorted(rows, key=lambda x: (x['model'], x['target']))
            page_size = 35
            for i in range(0, len(sorted_rows), page_size):
                chunk = sorted_rows[i:i + page_size]
                add_table_page(pdf, f"Summary (rows {i + 1}-{i + len(chunk)})", chunk)

            # Errors page
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.set_title('Errors', fontsize=16, pad=20)
            if errors:
                y = 0.9
                for m, e in errors[:40]:
                    ax.text(0.05, y, f"- {m}: {e}", fontsize=9, va='top')
                    y -= 0.02
            else:
                ax.text(0.05, 0.9, 'None', fontsize=11, va='top')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    except Exception as e:
        # If matplotlib isn't available for some reason, still keep txt/md/html/json.
        with open(report_pdf + '.error.txt', 'w', encoding='utf-8') as f:
            f.write(str(e))

    print(f"\nOK: Report saved to {report_pdf}")
    print(f"OK: Report saved to {report_html}")
    print(f"OK: Report saved to {report_md}")
    print(f"OK: Report saved to {report_txt_path}")
    return report_pdf


def main():
    parser = argparse.ArgumentParser(
        description='Trading Strategy — Central entry point for analysis, training, and evaluation.'
    )
    parser.add_argument(
        '--analysis',
        action='store_true',
        help='Run exploratory analysis & correlation only (creates ml_data)'
    )
    parser.add_argument(
        '--data',
        action='store_true',
        help='Run data collection and feature derivation first'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train all models (or specified via --models)'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate all models and collect metrics'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate evaluation report from last run'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['linear_regression', 'random_forest', 'xgboost', 'svr', 'neural_network', 'arima', 'ensemble'],
        help='Specific models to train/evaluate'
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip analysis step (use existing ml_data). Use with --train.'
    )
    parser.add_argument(
        '--keep-history',
        action='store_true',
        help='Do NOT delete old reports/charts at startup (default: delete for one-run policy).'
    )
    parser.add_argument(
        '--no-update-data',
        action='store_true',
        help='Do NOT refresh the last 10,000 candles before analysis (default: refresh on --analysis).'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run an MVP backtest using model predictions (writes backtest_latest.* to reports/).'
    )
    parser.add_argument(
        '--bt-model',
        default='linear_regression',
        choices=['linear_regression', 'random_forest', 'svr', 'neural_network', 'xgboost', 'ensemble'],
        help='Model to backtest (must be trained already).'
    )
    parser.add_argument(
        '--bt-threshold',
        type=float,
        default=None,
        help='Trade only when |predicted return| > threshold. If omitted, threshold is auto-set from costs.'
    )
    parser.add_argument(
        '--bt-fee-bps',
        type=float,
        default=5.0,
        help='Per-turn fee in basis points (bps).'
    )
    parser.add_argument(
        '--bt-slippage-bps',
        type=float,
        default=2.0,
        help='Per-turn slippage in basis points (bps).'
    )
    parser.add_argument(
        '--bt-spread-bps',
        type=float,
        default=0.0,
        help='Per-turn spread in basis points (bps).'
    )
    parser.add_argument(
        '--bt-allow-short',
        action='store_true',
        help='Enable shorting (default is long/flat only for MVP).'
    )
    parser.add_argument(
        '--bt-no-short',
        action='store_true',
        help='Force-disable shorting (override).'
    )
    parser.add_argument(
        '--bt-max-leverage',
        type=float,
        default=1.0,
        help='Max leverage / position cap (e.g. 1.0 = 100% notional).'
    )
    parser.add_argument(
        '--bt-vol-target',
        type=float,
        default=None,
        help='Annualized volatility target (e.g. 0.5 for 50%). Omit to disable.'
    )
    parser.add_argument(
        '--bt-walk-forward',
        action='store_true',
        help='Use walk-forward rolling window validation instead of single split.'
    )
    parser.add_argument(
        '--bt-single-split',
        action='store_true',
        help='Force single-split backtest (debug only). Default is walk-forward.'
    )
    parser.add_argument(
        '--bt-wf-train-bars',
        type=int,
        default=600,
        help='Walk-forward train window length (bars).'
    )
    parser.add_argument(
        '--bt-wf-test-bars',
        type=int,
        default=200,
        help='Walk-forward test window length (bars).'
    )
    parser.add_argument(
        '--bt-wf-step-bars',
        type=int,
        default=200,
        help='Walk-forward step size between windows (bars).'
    )
    
    args = parser.parse_args()

    # One-run policy: clean old reports/charts unless explicitly disabled.
    if not args.keep_history:
        clean_run_outputs()
    
    # If no flags, run full pipeline
    if not any([args.analysis, args.data, args.train, args.evaluate, args.report, args.backtest]):
        args.analysis = True
        args.train = True
        args.evaluate = True
        args.report = True
        args.backtest = True
        if not os.path.exists(os.path.join(PROJECT_ROOT, 'ml_data', 'X_train.csv')):
            args.data = True
        print("Running full pipeline (analysis + train + evaluate + report + backtest)")
    
    # Step 0: Data (optional)
    if args.data or (args.analysis and not args.no_update_data):
        run_data_pipeline()
    
    # Step 1: Analysis (creates ml_data)
    if args.analysis:
        run_analysis()
    elif args.train and not os.path.exists(os.path.join(PROJECT_ROOT, 'ml_data', 'X_train.csv')):
        print("\n⚠ ml_data not found. Run with --analysis first, or --data then --analysis.")
        sys.exit(1)
    
    # Step 2: Train models
    if args.train:
        print("\n" + "=" * 80)
        print("STEP 2: TRAINING MODELS")
        print("=" * 80)
        train_models(args.models)
    
    # Step 3: Evaluate
    if args.evaluate:
        print("\n" + "=" * 80)
        print("STEP 3: EVALUATING MODELS")
        print("=" * 80)
        all_metrics = evaluate_models(args.models)
        
        # Step 4: Report
        if args.report and all_metrics:
            print("\n" + "=" * 80)
            print("STEP 4: GENERATING REPORT")
            print("=" * 80)
            generate_report(all_metrics)

    elif args.report:
        # Report only: need to evaluate first to get metrics
        print("\nEvaluating models to generate report...")
        all_metrics = evaluate_models(args.models)
        if all_metrics:
            generate_report(all_metrics)
        else:
            print("No metrics available. Run --evaluate first.")

    # Step 5: Backtest (optional)
    if args.backtest:
        print("\n" + "=" * 80)
        print("STEP 5: BACKTESTING (MVP)")
        print("=" * 80)
        from backtesting.backtest import BacktestConfig, run_backtest, run_walk_forward_backtest, write_backtest_reports

        # Conservative defaults for MVP trading:
        # - walk-forward is default unless explicitly forced to single split
        # - long-only is default unless user explicitly allows shorts
        allow_short = bool(getattr(args, "bt_allow_short", False)) and (not bool(getattr(args, "bt_no_short", False)))
        cfg = BacktestConfig(
            model_name=args.bt_model,
            target="returns",
            threshold=args.bt_threshold,
            fee_bps=float(args.bt_fee_bps),
            slippage_bps=float(args.bt_slippage_bps),
            spread_bps=float(args.bt_spread_bps),
            allow_short=allow_short,
            max_leverage=float(args.bt_max_leverage),
            vol_target_annual=args.bt_vol_target,
            walk_forward=bool(args.bt_walk_forward) or (not bool(args.bt_single_split)),
            wf_train_bars=int(args.bt_wf_train_bars),
            wf_test_bars=int(args.bt_wf_test_bars),
            wf_step_bars=int(args.bt_wf_step_bars),
        )
        prefix = os.path.join(REPORTS_DIR, "backtest_latest")
        res = run_walk_forward_backtest(PROJECT_ROOT, cfg, prefix) if cfg.walk_forward else run_backtest(PROJECT_ROOT, cfg, prefix)
        write_backtest_reports(res, prefix)
        print("OK: Backtest reports saved to reports/backtest_latest.*")

        # Viability gate (MVP safety): warn loudly when strategy is not tradable.
        summ = res.get("summary", {})
        sharpe = float(summ.get("sharpe", 0.0))
        total = float(summ.get("total_return_pct", 0.0))
        mdd = float(summ.get("max_drawdown_pct", 0.0))
        bh = summ.get("buy_hold_total_return_pct", None)
        ok = True
        reasons = []
        if sharpe < 0.5:
            ok = False
            reasons.append(f"Sharpe too low ({sharpe:.3f} < 0.5)")
        if mdd < -20.0:
            ok = False
            reasons.append(f"Max drawdown too large ({mdd:.2f}% < -20%)")
        if bh is not None and total < float(bh):
            ok = False
            reasons.append(f"Underperforms buy-and-hold ({total:.2f}% < {float(bh):.2f}%)")

        print("\n" + "=" * 80)
        print("VIABILITY CHECK (MVP)")
        print("=" * 80)
        if ok:
            print("Status: PASS (paper-trade candidate)")
        else:
            print("Status: FAIL (DO NOT TRADE LIVE)")
            for r in reasons:
                print(f"- {r}")
        print("")

    # Final actionable printout (always last)
    _fee = float(getattr(args, "bt_fee_bps", 5.0))
    _slip = float(getattr(args, "bt_slippage_bps", 2.0))
    _spr = float(getattr(args, "bt_spread_bps", 0.0))
    _thr_arg = getattr(args, "bt_threshold", None)
    _thr = float(_thr_arg) if _thr_arg is not None else (2.0 * (_fee + _slip + _spr) / 10000.0)
    print_final_trade_plan(
        model_name=args.bt_model if hasattr(args, "bt_model") else "linear_regression",
        threshold=_thr,
        allow_short=(bool(getattr(args, "bt_allow_short", False)) and (not bool(getattr(args, "bt_no_short", False)))),
        risk_usdt=0.0,
        account_usdt=1000.0,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
    )
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
