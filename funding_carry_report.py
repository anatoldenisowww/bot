"""Colorful, self-contained HTML dashboard for the funding-carry screener.

Per symbol: how persistently funding pays (the edge), the sustainable
annualized carry, the one-time cost to get in/out, a traffic-light verdict,
and a funding-history chart (green bars = you collect, red = you pay) with the
cumulative harvested-carry curve. Theme-aware, no external assets.
"""
from __future__ import annotations

import html
from typing import List

import numpy as np

from funding_carry import CarryConfig, CarryReport


def _styles() -> str:
    return """<style>
  .fc{ --bg:#f6f7f9; --surface:#fff; --ink:#14161b; --muted:#59616e; --border:#e3e6ec; --accent:#3f5bd9;
       --good:#2e9e6a; --marginal:#c48a1e; --poor:#d1483f; --pay:#d1483f; --collect:#2e9e6a;
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       color:var(--ink); background:var(--bg); max-width:1080px; margin:0 auto; padding:30px 20px 48px; line-height:1.5; }
  @media (prefers-color-scheme:dark){ .fc{ --bg:#0e1116; --surface:#161a20; --ink:#e9ebef; --muted:#97a1af; --border:#252b34;
       --accent:#7d92f5; --good:#4fca92; --marginal:#e2b24a; --poor:#ec6f66; --pay:#ec6f66; --collect:#4fca92; } }
  :root[data-theme="light"] .fc{ --bg:#f6f7f9; --surface:#fff; --ink:#14161b; --muted:#59616e; --border:#e3e6ec; --accent:#3f5bd9; --good:#2e9e6a; --marginal:#c48a1e; --poor:#d1483f; --pay:#d1483f; --collect:#2e9e6a; }
  :root[data-theme="dark"] .fc{ --bg:#0e1116; --surface:#161a20; --ink:#e9ebef; --muted:#97a1af; --border:#252b34; --accent:#7d92f5; --good:#4fca92; --marginal:#e2b24a; --poor:#ec6f66; --pay:#ec6f66; --collect:#4fca92; }
  .fc *{ box-sizing:border-box; }
  .fc .eyebrow{ font-size:11px; text-transform:uppercase; letter-spacing:.1em; color:var(--accent); font-weight:700; }
  .fc h1{ font-size:26px; font-weight:680; letter-spacing:-.015em; margin:4px 0 6px; }
  .fc .lede{ color:var(--muted); font-size:14px; margin:0 0 24px; max-width:74ch; }
  .fc section.sym{ background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:20px 22px; margin-bottom:18px; }
  .fc .head{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:12px; }
  .fc .head h2{ font-size:19px; font-weight:680; margin:0; }
  .fc .verdict{ display:inline-block; padding:4px 12px; border-radius:999px; font-size:12.5px; font-weight:700; color:#fff; }
  .fc .verdict.good{ background:var(--good); } .fc .verdict.marginal{ background:var(--marginal); } .fc .verdict.poor{ background:var(--poor); }
  .fc .metrics{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:4px 0 14px; }
  .fc .metric{ border:1px solid var(--border); border-radius:10px; padding:10px 12px; }
  .fc .metric .k{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
  .fc .metric .v{ font-size:17px; font-weight:680; font-variant-numeric:tabular-nums; margin-top:3px; }
  .fc .metric .v.pos{ color:var(--good); } .fc .metric .v.neg{ color:var(--poor); }
  .fc .cap{ font-size:12px; color:var(--muted); margin-top:4px; }
  .fc .foot{ color:var(--muted); font-size:12px; margin-top:8px; border-top:1px solid var(--border); padding-top:14px; }
  .fc .foot b{ color:var(--ink); }
</style>"""


def _chart(report: CarryReport, width=1000, height=170) -> str:
    rates = report.funding_rate.to_numpy()
    eq = np.array([e for _, e in report.equity_curve])
    n = len(rates)
    if n < 2:
        return ""
    mid = height * 0.62
    lim = max(float(np.abs(rates).max()), 1e-9)
    bw = (width - 12) / n
    bars = []
    for i, r in enumerate(rates):
        h = (abs(r) / lim) * (mid - 14)
        x = 6 + i * bw
        if r >= 0:
            bars.append(f'<rect x="{x:.1f}" y="{mid-h:.1f}" width="{max(bw-0.6,0.6):.1f}" height="{h:.1f}" fill="var(--collect)"/>')
        else:
            bars.append(f'<rect x="{x:.1f}" y="{mid:.1f}" width="{max(bw-0.6,0.6):.1f}" height="{h:.1f}" fill="var(--pay)"/>')
    bars.append(f'<line x1="6" y1="{mid:.1f}" x2="{width-6:.1f}" y2="{mid:.1f}" stroke="var(--muted)" stroke-width="0.8" opacity="0.5"/>')
    # cumulative harvested-carry curve in the lower strip
    lo, hi = float(eq.min()), float(eq.max())
    span = (hi - lo) or 1e-9
    y0, y1 = height - 6, mid + 10
    pts = " ".join(f"{6 + i*bw:.1f},{y0 - (eq[i]-lo)/span*(y0-y1):.1f}" for i in range(n))
    curve = f'<polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="1.6" vector-effect="non-scaling-stroke"/>'
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:100%;display:block" role="img" '
            f'aria-label="funding history and harvested carry">' + "".join(bars) + curve + "</svg>")


def _verdict(report: CarryReport, cfg: CarryConfig):
    s = report.stats
    if s.worth_it:
        return "good", "Worth harvesting"
    if s.persistence_pct >= 60 and s.net_annual_pct > 0:
        return "marginal", "Borderline"
    return "poor", "Not worth it"


def _section(report: CarryReport, cfg: CarryConfig) -> str:
    s = report.stats
    vc, vl = _verdict(report, cfg)
    base = s.symbol.split("/")[0]
    net_cls = "pos" if s.net_annual_pct > 0 else "neg"
    return f"""<section class="sym">
  <div class="head"><h2>{html.escape(base)}</h2><span class="verdict {vc}">{html.escape(vl)}</span></div>
  <div class="metrics">
    <div class="metric"><div class="k">funding pays (persistence)</div><div class="v">{s.persistence_pct:.0f}%</div></div>
    <div class="metric"><div class="k">sustainable carry / yr</div><div class="v {net_cls}">{s.net_annual_pct:+.1f}%</div></div>
    <div class="metric"><div class="k">one-time in/out cost</div><div class="v">{s.one_time_cost_pct:.2f}%</div></div>
    <div class="metric"><div class="k">sample</div><div class="v">{s.days_covered:.0f} days</div></div>
  </div>
  {_chart(report)}
  <div class="cap">Green bars = funding you'd collect (positive), red = funding you'd pay (negative). Blue line = cumulative
    harvested carry over the sample. You hold long spot + short perp; direction cancels, you keep the funding.</div>
</section>"""


def render_carry_html(reports: List[CarryReport], cfg: CarryConfig) -> str:
    from funding_carry import screen
    ordered = screen(reports)
    sections = "".join(_section(r, cfg) for r in ordered)
    body = (
        _styles()
        + '<div class="fc">'
        + '<div class="eyebrow">Funding-rate carry · cash-and-carry</div>'
        + "<h1>Funding carry screener</h1>"
        + '<p class="lede">The one structurally-grounded edge here: perpetual funding is <b>positive most of the '
          "time</b>, so holding <b>long spot + short perp</b> (delta-neutral) collects it without betting on "
          "direction. Returns are modest - this is rent, not riches - and the sample is only what Bitget serves "
          "(~33 days), so treat the annualized figures as indicative, not precise.</p>"
        + sections
        + '<div class="foot"><b>Honest limits.</b> Funding can turn negative (you\'d pay) - it\'s positive ~80% of '
          "the time, not always, and the negatives are already in these numbers. Basis risk (spot vs perp drift) is a "
          "real variance the central estimate treats as ~0. Needs both a spot and a perp leg; at 200 EUR the ~$5 leg "
          "minimums mean 1-2 positions only - it scales as the account grows. Not financial advice.</div>"
        + "</div>"
    )
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Funding carry screener</title>\n<style>body{margin:0}</style>\n</head>\n<body>\n"
        + body + "\n</body>\n</html>\n"
    )
