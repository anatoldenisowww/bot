"""Render an HMM RegimeReport as a self-contained HTML dashboard.

No external libraries, fonts, or network calls - the price chart is inline
SVG and all styling is a token-based inline stylesheet, so the file opens
anywhere, renders identically, and adapts to the viewer's light/dark theme.

Two entry points:
- render_regime_body(report): the <style> + content fragment (used when the
  markup is embedded in a page skeleton, e.g. a published artifact).
- render_regime_html(report): a complete standalone .html document wrapping
  that fragment - what `main.py regime --html` writes to disk.

Regime color is a semantic data encoding (red bearish -> amber chop -> green
bull), deliberately distinct from the indigo structural accent.
"""
from __future__ import annotations

import html
from typing import List

from regime_hmm import RegimeReport

# Semantic regime ramp by rank (index 0 = most bearish). [light, dark] pairs.
_REGIME_COLORS = {
    2: [("#d3574f", "#e0736b"), ("#3f9e73", "#57b98a")],
    3: [("#d3574f", "#e0736b"), ("#b8873a", "#d1a24e"), ("#3f9e73", "#57b98a")],
    4: [("#d3574f", "#e0736b"), ("#c9743a", "#dd9152"), ("#8a8f37", "#b4b957"), ("#3f9e73", "#57b98a")],
}


def _regime_var(state_index: int, n: int) -> str:
    return f"var(--regime-{state_index})"


def _regime_color_defs(n: int) -> str:
    pairs = _REGIME_COLORS.get(n, _REGIME_COLORS[3])
    light = "".join(f"  --regime-{i}: {lp};\n" for i, (lp, _dp) in enumerate(pairs))
    dark = "".join(f"  --regime-{i}: {dp};\n" for i, (_lp, dp) in enumerate(pairs))
    return light, dark


def _styles(n_states: int) -> str:
    light_regimes, dark_regimes = _regime_color_defs(n_states)
    return f"""<style>
  .rgm {{
    --bg: #f7f8fa; --surface: #ffffff; --surface-2: #fbfcfd;
    --ink: #16181d; --muted: #5b6470; --border: #e4e7ec;
    --accent: #3f5bd9;
{light_regimes}
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--ink); background: var(--bg);
    max-width: 1040px; margin: 0 auto; padding: 28px 20px 40px;
    line-height: 1.5; -webkit-font-smoothing: antialiased;
  }}
  @media (prefers-color-scheme: dark) {{
    .rgm {{
      --bg: #0f1216; --surface: #171b21; --surface-2: #13171c;
      --ink: #e8eaed; --muted: #98a2b0; --border: #262b33;
      --accent: #7d92f5;
{dark_regimes}    }}
  }}
  :root[data-theme="light"] .rgm {{
    --bg: #f7f8fa; --surface: #ffffff; --surface-2: #fbfcfd;
    --ink: #16181d; --muted: #5b6470; --border: #e4e7ec; --accent: #3f5bd9;
{light_regimes}  }}
  :root[data-theme="dark"] .rgm {{
    --bg: #0f1216; --surface: #171b21; --surface-2: #13171c;
    --ink: #e8eaed; --muted: #98a2b0; --border: #262b33; --accent: #7d92f5;
{dark_regimes}  }}
  .rgm * {{ box-sizing: border-box; }}
  .rgm h1 {{ font-size: 24px; font-weight: 650; letter-spacing: -0.01em; margin: 0 0 6px; text-wrap: balance; }}
  .rgm .lede {{ color: var(--muted); font-size: 14px; margin: 0 0 22px; max-width: 68ch; }}
  .rgm .eyebrow {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 600; }}
  .rgm .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-bottom: 18px; }}
  .rgm .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; }}
  .rgm .card .big {{ font-size: 21px; font-weight: 680; margin: 6px 0 3px; letter-spacing: -0.01em; }}
  .rgm .card .sub {{ font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }}
  .rgm .chart {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px 16px 10px; margin-bottom: 20px; }}
  .rgm .chart .cap {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .rgm .grid2 {{ display: grid; grid-template-columns: 1.7fr 1fr; gap: 20px; }}
  @media (max-width: 720px) {{ .rgm .grid2 {{ grid-template-columns: 1fr; }} }}
  .rgm h3 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 10px; font-weight: 650; }}
  .rgm table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
  .rgm .scroll {{ overflow-x: auto; }}
  .rgm th {{ text-align: left; font-weight: 600; color: var(--muted); font-size: 12px; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
  .rgm td {{ padding: 7px 8px; border-bottom: 1px solid var(--border); }}
  .rgm td.num, .rgm th.num {{ text-align: right; }}
  .rgm .swatch {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; vertical-align: -1px; margin-right: 6px; }}
  .rgm .foot {{ color: var(--muted); font-size: 12px; margin-top: 22px; border-top: 1px solid var(--border); padding-top: 12px; }}
  .rgm .disclaim {{ color: var(--accent); font-weight: 600; }}
</style>"""


def _price_chart_svg(report: RegimeReport, width: int = 980, height: int = 300) -> str:
    closes = report.closes.to_numpy()
    seq = report.state_sequence
    n = len(closes)
    if n < 2:
        return "<p>not enough data to chart</p>"

    pad_l, pad_r, pad_t, pad_b = 8, 8, 14, 14
    lo, hi = float(closes.min()), float(closes.max())
    span = (hi - lo) or 1.0

    def x(i: int) -> float:
        return pad_l + (width - pad_l - pad_r) * i / (n - 1)

    def y(v: float) -> float:
        return pad_t + (height - pad_t - pad_b) * (1 - (v - lo) / span)

    bands: List[str] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or seq[i] != seq[start]:
            state = int(seq[start])
            if state >= 0:
                x0, x1 = x(start), x(i - 1)
                bands.append(
                    f'<rect x="{x0:.1f}" y="{pad_t}" width="{max(x1 - x0, 1):.1f}" '
                    f'height="{height - pad_t - pad_b}" fill="{_regime_var(state, report.n_states)}" opacity="0.15"/>'
                )
            start = i

    pts = " ".join(f"{x(i):.1f},{y(float(closes[i])):.1f}" for i in range(n))
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;display:block" '
        f'role="img" aria-label="price with regime shading">'
        + "".join(bands)
        + f'<polyline points="{pts}" fill="none" stroke="var(--ink)" stroke-width="1.4" '
          'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>'
        + "</svg>"
    )


def render_regime_body(report: RegimeReport) -> str:
    cur = report.current

    state_rows = ""
    for s in report.states:
        dur = "&infin;" if s.expected_duration_bars == float("inf") else f"{s.expected_duration_bars:.1f}"
        state_rows += (
            "<tr>"
            f'<td><span class="swatch" style="background:{_regime_var(s.index, report.n_states)}"></span>{s.index}</td>'
            f"<td>{html.escape(s.label)}</td>"
            f'<td class="num">{s.mean_return_pct:+.3f}%</td>'
            f'<td class="num">{s.volatility_pct:.3f}%</td>'
            f'<td class="num">{s.frequency_pct:.0f}%</td>'
            f'<td class="num">{dur}</td>'
            "</tr>"
        )

    header = "".join(f'<th class="num">to&nbsp;{j}</th>' for j in range(report.n_states))
    trows = ""
    for i in range(report.n_states):
        cells = ""
        for j in range(report.n_states):
            p = report.transition_matrix[i][j]
            cells += f'<td class="num" style="background:color-mix(in srgb, var(--accent) {p*70:.0f}%, transparent)">{p:.2f}</td>'
        trows += f'<tr><th>from&nbsp;{i}</th>{cells}</tr>'

    probs = " · ".join(f"S{i} {p*100:.0f}%" for i, p in enumerate(report.current_state_probs))
    nxt = report.expected_next_state()
    cur_color = _regime_var(cur.index, report.n_states)

    return f"""{_styles(report.n_states)}
<div class="rgm">
  <div class="eyebrow">Hidden Markov Model · regime detection</div>
  <h1>{html.escape(report.symbol)} · {html.escape(report.timeframe)}</h1>
  <p class="lede">
    Fit on log-returns and rolling volatility over {len(report.closes)} candles. This describes the market's
    <em>current statistical regime and the historical dynamics between regimes</em> —
    <span class="disclaim">it is not a price forecast and not a profit guarantee.</span>
  </p>

  <div class="cards">
    <div class="card">
      <div class="eyebrow">Current regime</div>
      <div class="big" style="color:{cur_color}">{html.escape(cur.label)}</div>
      <div class="sub">mean {cur.mean_return_pct:+.3f}%/bar · vol {cur.volatility_pct:.3f}% · typically lasts {cur.expected_duration_bars:.1f} bars</div>
    </div>
    <div class="card">
      <div class="eyebrow">Confidence · last bar</div>
      <div class="big">{probs}</div>
      <div class="sub">Most likely next regime: {html.escape(nxt.label)}</div>
    </div>
  </div>

  <div class="chart">
    {_price_chart_svg(report)}
    <div class="cap">Price line with the background shaded by the inferred regime of each bar.</div>
  </div>

  <div class="grid2">
    <div>
      <h3>Regimes</h3>
      <div class="scroll">
        <table>
          <thead><tr>
            <th>#</th><th>character</th><th class="num">mean/bar</th>
            <th class="num">volatility</th><th class="num">time in</th><th class="num">avg dur</th>
          </tr></thead>
          <tbody>{state_rows}</tbody>
        </table>
      </div>
    </div>
    <div>
      <h3>Transition probabilities</h3>
      <div class="scroll">
        <table>
          <thead><tr><th></th>{header}</tr></thead>
          <tbody>{trows}</tbody>
        </table>
      </div>
      <div class="cap">P(next regime | current). A strong diagonal means regimes are sticky.</div>
    </div>
  </div>

  <div class="foot">
    Log-likelihood {report.log_likelihood:.1f}. HMM fitting is a non-convex optimization; a fixed seed is used each run
    for reproducibility. Regimes are labeled purely by their historical statistics, with no forward-looking claim.
  </div>
</div>"""


def render_regime_html(report: RegimeReport) -> str:
    """Complete standalone HTML document (what `main.py regime --html` writes)."""
    title = f"Market regime — {report.symbol} {report.timeframe}"
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        "<style>body{margin:0}</style>\n"
        "</head>\n<body>\n"
        + render_regime_body(report)
        + "\n</body>\n</html>\n"
    )
