"""Colorful, self-contained HTML dashboard for the pairs-trading screener.

Per pair it shows: how tightly the two coins move together, how fast (if at
all) their spread mean-reverts, where the spread sits right now vs its normal
range, the current trade signal, and an honest backtest. A traffic-light
verdict answers the one question that matters - "is this actually a tradeable
pair right now?" - because most of the time, for crypto majors, the honest
answer is no.

No external libraries/fonts/network - inline SVG + token-based CSS, theme-aware.
"""
from __future__ import annotations

import html
from typing import List

import numpy as np

from statarb import PairConfig, PairReport


def _verdict(report: PairReport, cfg: PairConfig):
    """Traffic-light read on tradeability. A pair is only interesting if its
    spread reverts reasonably fast AND the backtest cleared costs."""
    hl = report.stats.half_life_bars
    pf = report.backtest.profit_factor if report.backtest else 0.0
    ret = report.backtest.total_return_pct if report.backtest else 0.0
    if hl <= 40 and pf > 1.05 and ret > 0:
        return "good", "Looks tradeable", "Spread reverts fairly fast and the backtest cleared costs."
    if hl <= 120 and pf >= 0.95:
        return "marginal", "Borderline", "Some mean reversion, but the edge is thin once costs are paid."
    return "poor", "Not a good pair right now", (
        "The spread barely mean-reverts (very long half-life) — these coins mostly trend together. "
        "Fading it lost money after costs in the backtest."
    )


def _styles() -> str:
    return """<style>
  .sa {
    --bg:#f6f7f9; --surface:#ffffff; --ink:#14161b; --muted:#59616e; --border:#e3e6ec; --accent:#3f5bd9;
    --good:#2e9e6a; --marginal:#c48a1e; --poor:#d1483f;
    --long:#2e9e6a; --short:#d1483f; --flat:#7b8494;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    color:var(--ink); background:var(--bg); max-width:1080px; margin:0 auto; padding:30px 20px 48px; line-height:1.5;
  }
  @media (prefers-color-scheme: dark){ .sa{
    --bg:#0e1116; --surface:#161a20; --ink:#e9ebef; --muted:#97a1af; --border:#252b34; --accent:#7d92f5;
    --good:#4fca92; --marginal:#e2b24a; --poor:#ec6f66; --long:#4fca92; --short:#ec6f66; --flat:#8b94a2; } }
  :root[data-theme="light"] .sa{ --bg:#f6f7f9; --surface:#fff; --ink:#14161b; --muted:#59616e; --border:#e3e6ec; --accent:#3f5bd9; --good:#2e9e6a; --marginal:#c48a1e; --poor:#d1483f; --long:#2e9e6a; --short:#d1483f; --flat:#7b8494; }
  :root[data-theme="dark"] .sa{ --bg:#0e1116; --surface:#161a20; --ink:#e9ebef; --muted:#97a1af; --border:#252b34; --accent:#7d92f5; --good:#4fca92; --marginal:#e2b24a; --poor:#ec6f66; --long:#4fca92; --short:#ec6f66; --flat:#8b94a2; }
  .sa *{ box-sizing:border-box; }
  .sa .eyebrow{ font-size:11px; text-transform:uppercase; letter-spacing:.1em; color:var(--accent); font-weight:700; }
  .sa h1{ font-size:26px; font-weight:680; letter-spacing:-.015em; margin:4px 0 6px; }
  .sa .lede{ color:var(--muted); font-size:14px; margin:0 0 24px; max-width:72ch; }
  .sa section.pair{ background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:20px 22px; margin-bottom:20px; }
  .sa .head{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:14px; }
  .sa .head h2{ font-size:19px; font-weight:680; margin:0; }
  .sa .verdict{ display:inline-block; padding:4px 12px; border-radius:999px; font-size:12.5px; font-weight:700; color:#fff; }
  .sa .verdict.good{ background:var(--good); } .sa .verdict.marginal{ background:var(--marginal); } .sa .verdict.poor{ background:var(--poor); }
  .sa .verdict-note{ color:var(--muted); font-size:13px; flex:1 1 300px; }
  .sa .metrics{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin:6px 0 16px; }
  .sa .metric{ border:1px solid var(--border); border-radius:10px; padding:10px 12px; }
  .sa .metric .k{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
  .sa .metric .v{ font-size:17px; font-weight:680; font-variant-numeric:tabular-nums; margin-top:3px; }
  .sa .sig{ font-weight:700; }
  .sa .sig.long{ color:var(--long); } .sa .sig.short{ color:var(--short); } .sa .sig.flat{ color:var(--flat); }
  .sa .cap{ font-size:12px; color:var(--muted); margin-top:4px; }
  .sa .foot{ color:var(--muted); font-size:12px; margin-top:8px; border-top:1px solid var(--border); padding-top:14px; }
  .sa .foot b{ color:var(--ink); }
</style>"""


def _zscore_chart_svg(report: PairReport, cfg: PairConfig, width=1000, height=200) -> str:
    z = report.zscore.to_numpy()
    mask = ~np.isnan(z)
    if mask.sum() < 2:
        return "<p>not enough data to chart</p>"
    idx = np.where(mask)[0]
    zz = z[idx]
    n = len(zz)

    lim = max(cfg.stop_z + 0.5, float(np.nanmax(np.abs(zz))) * 1.05)
    pad_l, pad_r, pad_t, pad_b = 6, 6, 8, 8

    def x(k): return pad_l + (width - pad_l - pad_r) * k / (n - 1)
    def y(v): return pad_t + (height - pad_t - pad_b) * (1 - (v + lim) / (2 * lim))

    def band(lo, hi, color, op):
        return f'<rect x="{pad_l}" y="{y(hi):.1f}" width="{width-pad_l-pad_r}" height="{max(y(lo)-y(hi),0):.1f}" fill="{color}" opacity="{op}"/>'

    parts = [
        band(cfg.entry_z, lim, "var(--short)", 0.10),
        band(-lim, -cfg.entry_z, "var(--long)", 0.10),
        band(-cfg.exit_z, cfg.exit_z, "var(--flat)", 0.10),
    ]
    for zl in (cfg.entry_z, -cfg.entry_z, cfg.stop_z, -cfg.stop_z):
        dash = "" if abs(zl) == cfg.entry_z else 'stroke-dasharray="4 4"'
        parts.append(f'<line x1="{pad_l}" y1="{y(zl):.1f}" x2="{width-pad_r}" y2="{y(zl):.1f}" stroke="var(--muted)" stroke-width="0.8" {dash} opacity="0.6"/>')
    parts.append(f'<line x1="{pad_l}" y1="{y(0):.1f}" x2="{width-pad_r}" y2="{y(0):.1f}" stroke="var(--muted)" stroke-width="0.8" opacity="0.4"/>')
    pts = " ".join(f"{x(k):.1f},{y(float(zz[k])):.1f}" for k in range(n))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--ink)" stroke-width="1.4" vector-effect="non-scaling-stroke"/>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:100%;display:block" role="img" '
            f'aria-label="spread z-score">' + "".join(parts) + "</svg>")


def _sig_class(sig: str) -> str:
    return {"LONG_SPREAD": "long", "SHORT_SPREAD": "short", "FLAT": "flat"}.get(sig, "flat")


def _sig_text(report: PairReport) -> str:
    s = report.stats
    a = s.symbol_a.split("/")[0]
    b = s.symbol_b.split("/")[0]
    if s.current_signal == "LONG_SPREAD":
        return f"BUY {a} / SELL {b} (spread is unusually cheap)"
    if s.current_signal == "SHORT_SPREAD":
        return f"SELL {a} / BUY {b} (spread is unusually rich)"
    return "No trade — spread is within its normal range"


def _pair_section(report: PairReport, cfg: PairConfig) -> str:
    s = report.stats
    bt = report.backtest
    vclass, vlabel, vnote = _verdict(report, cfg)
    a, b = s.symbol_a.split("/")[0], s.symbol_b.split("/")[0]
    hl = "∞" if s.half_life_bars == float("inf") else f"{s.half_life_bars:.0f}"
    pf = "∞" if bt.profit_factor == float("inf") else f"{bt.profit_factor:.2f}"

    return f"""<section class="pair">
  <div class="head">
    <h2>{html.escape(a)} vs {html.escape(b)}</h2>
    <span class="verdict {vclass}">{html.escape(vlabel)}</span>
    <span class="verdict-note">{html.escape(vnote)}</span>
  </div>
  <div class="metrics">
    <div class="metric"><div class="k">move together</div><div class="v">{s.correlation:.2f}</div></div>
    <div class="metric"><div class="k">reverts in (½-life)</div><div class="v">{hl} bars</div></div>
    <div class="metric"><div class="k">spread now (z)</div><div class="v">{s.current_z:+.2f}</div></div>
    <div class="metric"><div class="k">backtest return</div><div class="v">{bt.total_return_pct:+.1f}%</div></div>
    <div class="metric"><div class="k">profit factor</div><div class="v">{pf}</div></div>
    <div class="metric"><div class="k">win rate</div><div class="v">{bt.win_rate:.0f}%</div></div>
  </div>
  <div class="sig {_sig_class(s.current_signal)}">Signal now: {html.escape(_sig_text(report))}</div>
  {_zscore_chart_svg(report, cfg)}
  <div class="cap">Spread z-score over time. Red band = spread rich (fade it), green band = spread cheap (buy it),
    grey = normal (no trade). Dashed lines are the stop level where the relationship is treated as broken.</div>
</section>"""


def render_statarb_html(reports: List[PairReport], cfg: PairConfig, timeframe: str) -> str:
    sections = "".join(_pair_section(r, cfg) for r in reports)
    body = (
        _styles()
        + '<div class="sa">'
        + '<div class="eyebrow">Statistical arbitrage · pairs screener</div>'
        + "<h1>Crypto pairs-trading screener</h1>"
        + f'<p class="lede">For each pair we fit a spread, measure how fast it mean-reverts, and backtest fading '
          f'its extremes (net of fees + slippage on all four legs per round trip, {timeframe} candles). '
          "The verdict tells you whether the pair is worth trading right now — for crypto majors the honest "
          "answer is usually <b>no</b>, because they trend together instead of reverting.</p>"
        + sections
        + '<div class="foot"><b>This is not pure arbitrage.</b> There is no guaranteed convergence; the edge is '
          "statistical and individual trades lose. Relationships break (a coin gets its own news), which is why "
          "there is a hard stop. Backtests are past performance, not a promise. Validate out-of-sample before "
          "risking money.</div>"
        + "</div>"
    )
    title = f"Crypto pairs-trading screener ({timeframe})"
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>body{{margin:0}}</style>\n</head>\n<body>\n"
        + body + "\n</body>\n</html>\n"
    )
