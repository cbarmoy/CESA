"""HTML report generator for CESA filter pipelines and audit logs.

Produces a self-contained HTML document that includes:

* Per-channel filter pipeline summaries
* Embedded SVG charts (signal overlay, frequency response, PSD, SNR)
* Full audit log table
* Adaptive suggestions accepted / proposed
* Channel annotations
* Metadata (CESA version, generation date, selected channels/period)

The report can be opened in any modern browser and printed to PDF via the
browser print dialog.  No external CSS or JS dependencies are required.

Usage (programmatic)::

    from CESA.report_generator import ReportGenerator
    gen = ReportGenerator(
        pipelines={"EEG Fp1": my_pipeline},
        audit_log=my_audit_log,
        annotations={"EEG Fp1": "Subject showed alpha bursts"},
    )
    gen.generate("report.html")

Usage (CLI)::

    python -m CESA.report_generator --audit audit.json --output report.html
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_CSS = """\
:root{--bg:#f8f9fa;--fg:#212529;--accent:#0d6efd;--card:#fff;
--border:#dee2e6;--warn:#ffc107;--ok:#198754;--code-bg:#f0f0f0;
--table-stripe:#f2f6fc;--header-bg:#0d6efd;--header-fg:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a2e;--fg:#e0e0e0;
--accent:#4dabf7;--card:#16213e;--border:#334;--warn:#ffe066;
--ok:#69db7c;--code-bg:#0f3460;--table-stripe:#1a2744;
--header-bg:#1a2744;--header-fg:#e0e0e0}}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
background:var(--bg);color:var(--fg);line-height:1.6;padding:2rem}
h1{color:var(--accent);margin-bottom:.5rem}
h2{border-bottom:2px solid var(--accent);padding-bottom:.3rem;margin:1.5rem 0 .8rem}
h3{margin:.8rem 0 .4rem}
.meta{font-size:.85rem;color:#6c757d;margin-bottom:1.5rem}
.card{background:var(--card);border:1px solid var(--border);
border-radius:8px;padding:1rem;margin-bottom:1rem;page-break-inside:avoid}
.card-title{font-weight:600;margin-bottom:.5rem}
.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem;
font-weight:600;margin-right:4px;color:#fff}
.tag-bp{background:#2196f3}.tag-hp{background:#4caf50}.tag-lp{background:#ff9800}
.tag-notch{background:#e91e63}.tag-smooth{background:#9c27b0}
table{width:100%;border-collapse:collapse;margin:.5rem 0}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border);
font-size:.85rem}
th{background:var(--header-bg);color:var(--header-fg)}
tr:nth-child(even){background:var(--table-stripe)}
.chart-container{text-align:center;margin:.8rem 0}
.chart-container img,.chart-container svg{max-width:100%;height:auto}
.warn{color:var(--warn);font-weight:600}
.ok{color:var(--ok);font-weight:600}
.snr-box{display:inline-block;padding:4px 12px;border-radius:4px;
font-weight:600;font-size:.9rem}
.annotation{background:var(--code-bg);border-left:4px solid var(--accent);
padding:.5rem .8rem;margin:.4rem 0;font-style:italic;font-size:.9rem}
.footer{margin-top:2rem;font-size:.8rem;color:#6c757d;text-align:center;
border-top:1px solid var(--border);padding-top:1rem}
@media print{body{padding:0}.card{break-inside:avoid}}
"""


def _svg_to_data_uri(svg_bytes: bytes) -> str:
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _render_signal_chart(
    raw_data: np.ndarray,
    filtered_data: np.ndarray,
    sfreq: float,
    title: str = "Signal",
) -> bytes:
    """Render raw vs filtered signal overlay as SVG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(raw_data), int(sfreq * 10))
    t = np.arange(n) / sfreq
    fig, ax = plt.subplots(figsize=(8, 2.2), dpi=96)
    ax.plot(t, raw_data[:n], color="#90caf9", linewidth=0.6, alpha=0.7, label="Raw")
    ax.plot(t, filtered_data[:n], color="#1565c0", linewidth=0.8, label="Filtered")
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_psd_chart(
    raw_data: np.ndarray,
    filtered_data: np.ndarray,
    sfreq: float,
) -> bytes:
    """Render raw vs filtered PSD as SVG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    nperseg = min(len(raw_data), int(sfreq * 4))
    if nperseg < 8:
        return b""
    f_r, p_r = welch(raw_data, fs=sfreq, nperseg=nperseg)
    f_f, p_f = welch(filtered_data, fs=sfreq, nperseg=nperseg)
    fig, ax = plt.subplots(figsize=(8, 2.2), dpi=96)
    ax.semilogy(f_r, p_r, color="#90caf9", linewidth=0.7, alpha=0.7, label="Raw PSD")
    ax.semilogy(f_f, p_f, color="#1565c0", linewidth=0.9, label="Filtered PSD")
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("PSD (V²/Hz)", fontsize=8)
    ax.set_title("Power Spectral Density", fontsize=9, fontweight="bold")
    ax.set_xlim(0, min(sfreq / 2, 80))
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_freq_response_chart(pipeline, sfreq: float) -> bytes:
    """Render combined frequency response of the pipeline as SVG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        result = pipeline.frequency_response(sfreq)
    except Exception:
        return b""
    if result is None:
        return b""

    freqs, gains_db = result
    if freqs is None or len(freqs) == 0:
        return b""

    fig, ax = plt.subplots(figsize=(8, 2.2), dpi=96)
    ax.plot(freqs, gains_db, color="#1565c0", linewidth=0.9, label="Combined")
    ax.axhline(-3, color="#e53935", linestyle="--", linewidth=0.5, label="-3 dB")
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Gain (dB)", fontsize=8)
    ax.set_title("Frequency Response", fontsize=9, fontweight="bold")
    ax.set_xlim(0, min(sfreq / 2, 150))
    ax.set_ylim(-60, 5)
    ax.legend(fontsize=7, loc="lower left")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _compute_snr(raw: np.ndarray, filtered: np.ndarray) -> float:
    sig_power = np.mean(filtered ** 2)
    noise = raw - filtered
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-30:
        return float("inf")
    return 10.0 * np.log10(sig_power / noise_power)


def _filter_tag_html(f) -> str:
    cls_map = {
        "BandpassFilter": ("BP", "tag-bp"),
        "HighpassFilter": ("HP", "tag-hp"),
        "LowpassFilter": ("LP", "tag-lp"),
        "NotchFilter": ("N", "tag-notch"),
        "SmoothingFilter": ("SM", "tag-smooth"),
    }
    name = type(f).__name__
    abbr, css = cls_map.get(name, (name[:2], "tag-bp"))
    return f'<span class="tag {css}">{abbr}</span>'


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@dataclass
class ReportConfig:
    """Options for report generation."""
    title: str = "CESA Filter Report"
    include_charts: bool = True
    include_audit: bool = True
    include_annotations: bool = True
    include_suggestions: bool = True
    channels: Optional[List[str]] = None
    time_range: Optional[Tuple[float, float]] = None


class ReportGenerator:
    """Generate a self-contained HTML report for filter pipelines.

    Parameters
    ----------
    pipelines : dict[str, FilterPipeline]
        Channel name -> FilterPipeline mapping.
    audit_log : FilterAuditLog | None
        Audit log instance to include in the report.
    annotations : dict[str, str]
        Channel name -> annotation text.
    suggestions : list[dict]
        Accepted/proposed adaptive filter suggestions.
    raw_data : dict[str, ndarray] | None
        Channel name -> raw signal data for chart generation.
    sfreq : float
        Sampling frequency (needed for charts).
    config : ReportConfig | None
        Report options.
    """

    def __init__(
        self,
        pipelines: Optional[Dict[str, Any]] = None,
        audit_log: Any = None,
        annotations: Optional[Dict[str, str]] = None,
        suggestions: Optional[List[Dict[str, Any]]] = None,
        raw_data: Optional[Dict[str, np.ndarray]] = None,
        sfreq: float = 256.0,
        config: Optional[ReportConfig] = None,
    ) -> None:
        self.pipelines = pipelines or {}
        self.audit_log = audit_log
        self.annotations = annotations or {}
        self.suggestions = suggestions or []
        self.raw_data = raw_data or {}
        self.sfreq = sfreq
        self.config = config or ReportConfig()

    def _channel_list(self) -> List[str]:
        if self.config.channels:
            return [c for c in self.config.channels if c in self.pipelines]
        return sorted(self.pipelines.keys())

    def _build_pipeline_section(self, channel: str) -> str:
        pipeline = self.pipelines[channel]
        lines: List[str] = []
        lines.append(f'<div class="card">')
        lines.append(f'<div class="card-title">{_escape(channel)}</div>')

        if hasattr(pipeline, "filters"):
            enabled = [f for f in pipeline.filters if getattr(f, "enabled", True)]
            disabled = [f for f in pipeline.filters if not getattr(f, "enabled", True)]
            if enabled:
                lines.append("<p>")
                for f in enabled:
                    lines.append(_filter_tag_html(f))
                lines.append("</p>")
            lines.append("<table><tr><th>Filter</th><th>Parameters</th><th>Status</th></tr>")
            for f in pipeline.filters:
                d = f.to_dict() if hasattr(f, "to_dict") else {}
                ftype = d.pop("type", type(f).__name__)
                en = d.pop("enabled", True)
                params = ", ".join(f"{k}={v}" for k, v in d.items())
                status_cls = "ok" if en else "warn"
                status_txt = "Enabled" if en else "Disabled"
                lines.append(
                    f'<tr><td>{_escape(ftype)}</td><td>{_escape(params)}</td>'
                    f'<td class="{status_cls}">{status_txt}</td></tr>'
                )
            lines.append("</table>")

            warnings = []
            if hasattr(pipeline, "physiological_warnings"):
                try:
                    ch_type = "eeg"
                    for t in ("eog", "emg", "ecg"):
                        if t in channel.lower():
                            ch_type = t
                            break
                    warnings = pipeline.physiological_warnings(ch_type, self.sfreq)
                except Exception:
                    pass
            if warnings:
                lines.append('<p class="warn">Warnings:</p><ul>')
                for w in warnings:
                    lines.append(f"<li>{_escape(w)}</li>")
                lines.append("</ul>")

        annotation = self.annotations.get(channel, "")
        if annotation and self.config.include_annotations:
            lines.append(f'<div class="annotation">{_escape(annotation)}</div>')

        if self.config.include_charts and channel in self.raw_data:
            raw = self.raw_data[channel]
            if hasattr(pipeline, "apply"):
                try:
                    filtered = pipeline.apply(raw, self.sfreq)
                except Exception:
                    filtered = raw.copy()
            else:
                filtered = raw.copy()

            sig_svg = _render_signal_chart(raw, filtered, self.sfreq, f"{channel} — Signal")
            if sig_svg:
                lines.append(f'<div class="chart-container"><img src="{_svg_to_data_uri(sig_svg)}" alt="Signal chart"></div>')

            psd_svg = _render_psd_chart(raw, filtered, self.sfreq)
            if psd_svg:
                lines.append(f'<div class="chart-container"><img src="{_svg_to_data_uri(psd_svg)}" alt="PSD chart"></div>')

            freq_svg = _render_freq_response_chart(pipeline, self.sfreq)
            if freq_svg:
                lines.append(f'<div class="chart-container"><img src="{_svg_to_data_uri(freq_svg)}" alt="Frequency response"></div>')

            snr = _compute_snr(raw, filtered)
            snr_cls = "ok" if snr > 5 else "warn"
            snr_text = f"{snr:.1f} dB" if np.isfinite(snr) else "∞"
            lines.append(f'<p>SNR: <span class="snr-box {snr_cls}">{snr_text}</span></p>')

        lines.append("</div>")
        return "\n".join(lines)

    def _build_audit_section(self) -> str:
        if not self.config.include_audit or self.audit_log is None:
            return ""
        entries = []
        if hasattr(self.audit_log, "to_list"):
            entries = self.audit_log.to_list()
        if not entries:
            return ""
        lines = ['<h2>Audit Log</h2>']
        lines.append("<table><tr><th>Timestamp</th><th>Channel</th><th>Action</th><th>Details</th></tr>")
        for e in entries:
            ts = _escape(str(e.get("timestamp", "")))
            ch = _escape(str(e.get("channel", "")))
            act = _escape(str(e.get("action", "")))
            det = _escape(json.dumps(e.get("details", {}), default=str))
            lines.append(f"<tr><td>{ts}</td><td>{ch}</td><td>{act}</td><td><code>{det}</code></td></tr>")
        lines.append("</table>")
        return "\n".join(lines)

    def _build_suggestions_section(self) -> str:
        if not self.config.include_suggestions or not self.suggestions:
            return ""
        lines = ['<h2>Adaptive Suggestions</h2>']
        lines.append("<table><tr><th>Preset</th><th>Channel</th><th>Reason</th><th>Confidence</th><th>Accepted</th></tr>")
        for s in self.suggestions:
            preset = _escape(str(s.get("preset_name", "")))
            ch = _escape(str(s.get("channel", "")))
            reason = _escape(str(s.get("reason", "")))
            conf = f"{s.get('confidence', 0):.0%}"
            accepted = "Yes" if s.get("accepted") else "No"
            lines.append(f"<tr><td>{preset}</td><td>{ch}</td><td>{reason}</td><td>{conf}</td><td>{accepted}</td></tr>")
        lines.append("</table>")
        return "\n".join(lines)

    def _build_summary_dashboard(self) -> str:
        channels = self._channel_list()
        n_total = len(channels)
        n_filtered = sum(
            1 for c in channels
            if hasattr(self.pipelines.get(c), "filters")
            and any(getattr(f, "enabled", True) for f in self.pipelines[c].filters)
        )
        n_warnings = 0
        for c in channels:
            p = self.pipelines.get(c)
            if p and hasattr(p, "physiological_warnings"):
                try:
                    ch_type = "eeg"
                    for t in ("eog", "emg", "ecg"):
                        if t in c.lower():
                            ch_type = t
                            break
                    n_warnings += len(p.physiological_warnings(ch_type, self.sfreq))
                except Exception:
                    pass
        n_annotations = sum(1 for c in channels if self.annotations.get(c))
        n_audit = 0
        if self.audit_log and hasattr(self.audit_log, "entries"):
            n_audit = len(self.audit_log.entries)

        lines = ['<h2>Dashboard</h2>', '<div class="card">', "<table>"]
        rows = [
            ("Channels", str(n_total)),
            ("Channels with active filters", str(n_filtered)),
            ("Physiological warnings", f'<span class="{"warn" if n_warnings else "ok"}">{n_warnings}</span>'),
            ("Channel annotations", str(n_annotations)),
            ("Audit log entries", str(n_audit)),
            ("Suggestions proposed", str(len(self.suggestions))),
        ]
        for label, val in rows:
            lines.append(f"<tr><td><strong>{label}</strong></td><td>{val}</td></tr>")
        lines.append("</table></div>")
        return "\n".join(lines)

    def render_html(self) -> str:
        """Build the full HTML string."""
        from CESA import __version__

        channels = self._channel_list()
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        parts: List[str] = []
        parts.append("<!DOCTYPE html><html lang='en'><head>")
        parts.append(f"<meta charset='utf-8'><title>{_escape(self.config.title)}</title>")
        parts.append(f"<style>{_CSS}</style></head><body>")
        parts.append(f"<h1>{_escape(self.config.title)}</h1>")
        parts.append(f'<div class="meta">CESA v{__version__} &mdash; Generated {now}</div>')

        parts.append(self._build_summary_dashboard())

        parts.append("<h2>Filter Pipelines</h2>")
        for ch in channels:
            parts.append(self._build_pipeline_section(ch))

        parts.append(self._build_audit_section())
        parts.append(self._build_suggestions_section())

        parts.append(f'<div class="footer">Generated by CESA v{__version__} &mdash; {now}</div>')
        parts.append("</body></html>")
        return "\n".join(parts)

    def generate(self, path: Union[str, Path]) -> Path:
        """Write the HTML report to *path* and return the resolved Path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.render_html()
        path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s (%d bytes)", path, len(html))
        return path


def generate_from_cli() -> None:
    """CLI entry point: ``python -m CESA.report_generator``."""
    import argparse

    parser = argparse.ArgumentParser(description="CESA filter report generator")
    parser.add_argument("--audit", type=str, default=None, help="Path to audit log JSON")
    parser.add_argument("--pipelines", type=str, default=None, help="Path to pipelines JSON")
    parser.add_argument("--output", type=str, default="filter_report.html", help="Output HTML path")
    parser.add_argument("--title", type=str, default="CESA Filter Report", help="Report title")
    args = parser.parse_args()

    from CESA.filter_engine import FilterAuditLog, FilterPipeline

    audit = FilterAuditLog()
    if args.audit:
        p = Path(args.audit)
        if p.exists():
            entries = json.loads(p.read_text(encoding="utf-8"))
            for e in entries:
                audit.record(e.get("channel", ""), e.get("action", ""), **e.get("details", {}))

    pipelines: Dict[str, FilterPipeline] = {}
    if args.pipelines:
        p = Path(args.pipelines)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for ch, pdict in data.items():
                pipelines[ch] = FilterPipeline.from_dict(pdict)

    gen = ReportGenerator(
        pipelines=pipelines,
        audit_log=audit,
        config=ReportConfig(title=args.title),
    )
    out = gen.generate(args.output)
    print(f"Report generated: {out}")


if __name__ == "__main__":
    generate_from_cli()
