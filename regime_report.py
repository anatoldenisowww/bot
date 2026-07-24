"""Render HMM RegimeReport(s) as a self-contained, colorful HTML dashboard.

No external libraries, fonts, or network calls - inline SVG chart + a
token-based inline stylesheet, so it renders identically anywhere and adapts
to the viewer's light/dark theme.

Entry points:
- render_regime_body(report): <style> + one symbol's section (single-symbol
  standalone / artifact fragment).
- render_multi_regime_html(reports): a complete .html document stacking
  several symbols (BTC / ETH / SOL) under one header - what
  `main.py regime --html` writes.

Two color systems, deliberately separate:
- Regime ramp (red bearish -> amber -> green bullish) shades the price chart.
- Lean pills (BUY green / HOLD amber / SELL red) are the plain-language call.
"""
from __future__ import annotations

import html
from typing import List

from regime_hmm import RegimeReport

# Semantic regime ramp by rank (index 0 = most bearish). [light, dark] pairs.
_REGIME_COLORS = {
    2: [("#d3574f", "#e0736b"), ("#3f9e73", "#57b98a")],
    3: [("#d3574f", "#e0736b"), ("#c99a34", "#e0bd5b"), ("#3f9e73", "#57b98a")],
    4: [("#d3574f", "#e0736b"), ("#d1863a", "#e5a45c"), ("#8a9a3c", "#b4c65a"), ("#3f9e73", "#57b98a")],
}

_LEAN_COLORS = {  # (light, dark)
    "BUY": ("#2e9e6a", "#4fca92"),
    "HOLD": ("#c48a1e", "#e2b24a"),
    "SELL": ("#d1483f", "#ec6f66"),
}


def _regime_var(state_index: int) -> str:
    return f"var(--regime-{state_index})"


def _regime_color_defs(n: int):
    pairs = _REGIME_COLORS.get(n, _REGIME_COLORS[3])
    light = "".join(f"  --regime-{i}: {lp};\n" for i, (lp, _dp) in enumerate(pairs))
    dark = "".join(f"  --regime-{i}: {dp};\n" for i, (_lp, dp) in enumerate(pairs))
    return light, dark


def _lean_pill(lean: str) -> str:
    l = lean.upper()
    return (f'<span class="pill pill-{l.lower()}">{html.escape(l)}</span>' if l in _LEAN_COLORS
            else html.escape(lean))


def _styles(max_states: int) -> str:
    light_regimes, dark_regimes = _regime_color_defs(max_states)
    lean_light = "".join(f"  --lean-{k.lower()}: {v[0]};\n" for k, v in _LEAN_COLORS.items())
    lean_dark = "".join(f"  --lean-{k.lower()}: {v[1]};\n" for k, v in _LEAN_COLORS.items())
    return f"""<style>
  .rgm {{
    --bg: #f6f7f9; --surface: #ffffff; --surface-2: #fbfcfd;
    --ink: #14161b; --muted: #59616e; --border: #e3e6ec; --accent: #3f5bd9;
{light_regimes}{lean_light}
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--ink); background: var(--bg);
    max-width: 1080px; margin: 0 auto; padding: 30px 20px 48px;
    line-height: 1.5; -webkit-font-smoothing: antialiased;
  }}
  @media (prefers-color-scheme: dark) {{
    .rgm {{
      --bg: #0e1116; --surface: #161a20; --surface-2: #12161b;
      --ink: #e9ebef; --muted: #97a1af; --border: #252b34; --accent: #7d92f5;
{dark_regimes}{lean_dark}    }}
  }}
  :root[data-theme="light"] .rgm {{
    --bg: #f6f7f9; --surface: #ffffff; --surface-2: #fbfcfd;
    --ink: #14161b; --muted: #59616e; --border: #e3e6ec; --accent: #3f5bd9;
{light_regimes}{lean_light}  }}
  :root[data-theme="dark"] .rgm {{
    --bg: #0e1116; --surface: #161a20; --surface-2: #12161b;
    --ink: #e9ebef; --muted: #97a1af; --border: #252b34; --accent: #7d92f5;
{dark_regimes}{lean_dark}  }}
  .rgm * {{ box-sizing: border-box; }}
  .rgm .hero-eyebrow {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); font-weight: 700; }}
  .rgm h1 {{ font-size: 26px; font-weight: 680; letter-spacing: -0.015em; margin: 4px 0 6px; text-wrap: balance; }}
  .rgm .lede {{ color: var(--muted); font-size: 14px; margin: 0 0 26px; max-width: 70ch; }}
  .rgm .legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 0 0 26px; font-size: 12.5px; }}
  .rgm .legend .pill {{ margin-right: 5px; }}
  .rgm section.sym {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 20px 22px; margin-bottom: 22px; }}
  .rgm .sym-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }}
  .rgm .sym-head h2 {{ font-size: 19px; font-weight: 680; margin: 0; letter-spacing: -0.01em; }}
  .rgm .sym-head .tf {{ color: var(--muted); font-size: 13px; font-weight: 500; }}
  .rgm .nowbar {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 14px 16px; border-radius: 12px;
                  border: 1px solid var(--border); margin-bottom: 16px; }}
  .rgm .nowbar .mood {{ font-size: 16px; font-weight: 680; }}
  .rgm .nowbar .expl {{ color: var(--muted); font-size: 13px; flex: 1 1 320px; }}
  .rgm .pill {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 12.5px; font-weight: 700;
               letter-spacing: 0.03em; color: #fff; }}
  .rgm .pill-buy {{ background: var(--lean-buy); }}
  .rgm .pill-hold {{ background: var(--lean-hold); }}
  .rgm .pill-sell {{ background: var(--lean-sell); }}
  .rgm .pill-lg {{ font-size: 15px; padding: 5px 15px; }}
  .rgm .conf {{ color: var(--muted); font-size: 12.5px; font-variant-numeric: tabular-nums; }}
  .rgm .chart {{ margin: 4px 0 8px; }}
  .rgm .cap {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .rgm .grid2 {{ display: grid; grid-template-columns: 1.7fr 1fr; gap: 22px; margin-top: 14px; }}
  @media (max-width: 760px) {{ .rgm .grid2 {{ grid-template-columns: 1fr; }} }}
  .rgm h3 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 10px; font-weight: 700; }}
  .rgm table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
  .rgm .scroll {{ overflow-x: auto; }}
  .rgm th {{ text-align: left; font-weight: 600; color: var(--muted); font-size: 11.5px; padding: 6px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  .rgm td {{ padding: 7px 8px; border-bottom: 1px solid var(--border); }}
  .rgm td.num, .rgm th.num {{ text-align: right; }}
  .rgm .swatch {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; vertical-align: -1px; margin-right: 6px; }}
  .rgm .foot {{ color: var(--muted); font-size: 12px; margin-top: 8px; border-top: 1px solid var(--border); padding-top: 14px; }}
  .rgm .foot b {{ color: var(--ink); }}
</style>"""


def _price_chart_svg(report: RegimeReport, width: int = 1000, height: int = 230) -> str:
    closes = report.closes.to_numpy()
    seq = report.state_sequence
    n = len(closes)
    if n < 2:
        return "<p>not enough data to chart</p>"

    pad_l, pad_r, pad_t, pad_b = 6, 6, 12, 12
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
                    f'height="{height - pad_t - pad_b}" fill="{_regime_var(state)}" opacity="0.22"/>'
                )
            start = i

    pts = " ".join(f"{x(i):.1f},{y(float(closes[i])):.1f}" for i in range(n))
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:100%;display:block" '
        f'role="img" aria-label="price with regime shading">'
        + "".join(bands)
        + f'<polyline points="{pts}" fill="none" stroke="var(--ink)" stroke-width="1.5" '
          'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>'
        + "</svg>"
    )


def _regime_section(report: RegimeReport) -> str:
    cur = report.current
    conf = report.current_state_probs[report.current_state] * 100

    state_rows = ""
    for s in report.states:
        dur = "&infin;" if s.expected_duration_bars == float("inf") else f"{s.expected_duration_bars:.0f}"
        state_rows += (
            "<tr>"
            f'<td><span class="swatch" style="background:{_regime_var(s.index)}"></span>{s.index}</td>'
            f"<td>{html.escape(s.label)}</td>"
            f"<td>{_lean_pill(s.lean)}</td>"
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
            cells += (f'<td class="num" style="background:color-mix(in srgb, var(--accent) {p*70:.0f}%, transparent)">'
                      f"{p:.2f}</td>")
        trows += f'<tr><th>from&nbsp;{i}</th>{cells}</tr>'

    nxt = report.expected_next_state()
    if nxt.index != cur.index:
        nxt_text = f"If the mood shifts, it most often moves to <b>{html.escape(nxt.label)}</b> ({_lean_pill(nxt.lean)})."
    else:
        nxt_text = "This mood is <b>sticky</b> — it usually continues rather than switching."

    return f"""<section class="sym">
  <div class="sym-head">
    <h2>{html.escape(report.symbol)}</h2>
    <span class="tf">{html.escape(report.timeframe)} · last {len(report.closes)} candles</span>
  </div>
  <div class="nowbar" style="border-left:4px solid {_regime_var(cur.index)}">
    <span class="pill pill-{cur.lean.lower()} pill-lg">{html.escape(cur.lean)}</span>
    <span class="mood">{html.escape(cur.label)}</span>
    <span class="conf">right now · {conf:.0f}% confident</span>
    <span class="expl">{html.escape(cur.plain)} {nxt_text}</span>
  </div>
  <div class="chart">{_price_chart_svg(report)}
    <div class="cap">Price line, background shaded by the mood of each candle (red = weak, amber = choppy, green = strong).</div>
  </div>
  <div class="grid2">
    <div>
      <h3>The moods this market has been in</h3>
      <div class="scroll"><table>
        <thead><tr>
          <th>#</th><th>mood</th><th>lean</th><th class="num">avg move/candle</th>
          <th class="num">swinginess</th><th class="num">how often</th><th class="num">lasts</th>
        </tr></thead>
        <tbody>{state_rows}</tbody>
      </table></div>
    </div>
    <div>
      <h3>How moods change</h3>
      <div class="scroll"><table>
        <thead><tr><th></th>{header}</tr></thead>
        <tbody>{trows}</tbody>
      </table></div>
      <div class="cap">Chance of tomorrow's mood given today's. Strong diagonal = moods persist.</div>
    </div>
  </div>
</section>"""


def _legend() -> str:
    return (
        '<div class="legend">'
        '<span><span class="pill pill-buy">BUY</span>prices historically rose in this mood</span>'
        '<span><span class="pill pill-hold">HOLD</span>historically choppy/sideways — wait</span>'
        '<span><span class="pill pill-sell">SELL</span>prices historically fell — reduce risk</span>'
        "</div>"
    )


def _disclaimer_foot() -> str:
    return (
        '<div class="foot">'
        "<b>Read this as weather, not a crystal ball.</b> The leans describe what price <i>did</i> "
        "in each mood in the past — they are <b>not</b> predictions and <b>not</b> financial advice. "
        "Moods flip without warning and detecting the current one has some lag. HMM fitting uses a fixed "
        "seed for reproducibility; regimes are labeled purely by their statistics."
        "</div>"
    )


def render_regime_body(report: RegimeReport) -> str:
    """Single-symbol fragment: <style> + one section + legend + disclaimer."""
    return (
        _styles(report.n_states)
        + '<div class="rgm">'
        + '<div class="hero-eyebrow">Hidden Markov Model · market moods</div>'
        + f"<h1>{html.escape(report.symbol)} regime read</h1>"
        + _legend()
        + _regime_section(report)
        + _disclaimer_foot()
        + "</div>"
    )


def render_multi_regime_html(reports: List[RegimeReport]) -> str:
    """Complete standalone document stacking several symbols under one header."""
    max_states = max((r.n_states for r in reports), default=3)
    symbols = ", ".join(r.symbol.split("/")[0] for r in reports)
    sections = "".join(_regime_section(r) for r in reports)
    body = (
        _styles(max_states)
        + '<div class="rgm">'
        + '<div class="hero-eyebrow">Hidden Markov Model · market moods</div>'
        + "<h1>Crypto regime read</h1>"
        + f'<p class="lede">What "mood" {html.escape(symbols)} are each in right now, and the plain-language '
          "buy / hold / sell lean that mood historically rewarded. Scroll for each coin.</p>"
        + _legend()
        + sections
        + _disclaimer_foot()
        + "</div>"
    )
    title = f"Crypto regime read — {symbols}"
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        "<style>body{margin:0}</style>\n"
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )


def render_regime_html(report: RegimeReport) -> str:
    """Complete standalone document for a single symbol (back-compat)."""
    return render_multi_regime_html([report])
