"""Publication-ready export: PDF/HTML reports with hypnogram, EEG excerpts, metrics.

Generates structured reports suitable for clinical documentation or
research publications, including embedded figures, annotation tables,
and ML results.
"""

from __future__ import annotations

import datetime
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Builds an HTML report from viewer / pipeline data.

    The report includes:
    * Recording info
    * Hypnogram figure (as base64 PNG)
    * Sleep architecture metrics
    * Stage distribution
    * Annotation table
    * Filter audit
    * ML comparison table
    * Optional EEG excerpt figures
    """

    def __init__(self) -> None:
        self._title = "CESA - Rapport PSG"
        self._recording_info: Dict[str, str] = {}
        self._sleep_metrics: Dict[str, float] = {}
        self._stage_distribution: Dict[str, float] = {}
        self._annotations: List[Dict[str, Any]] = []
        self._filter_info: List[Dict[str, str]] = []
        self._ml_results: List[Dict[str, Any]] = []
        self._figures: List[Tuple[str, str]] = []  # (title, base64_png)
        self._warnings: List[str] = []
        self._hypnogram_labels: List[str] = []
        self._epoch_len: float = 30.0

    # ---- Builder methods -------------------------------------------

    def set_title(self, title: str) -> "ReportBuilder":
        self._title = title
        return self

    def set_recording_info(self, info: Dict[str, str]) -> "ReportBuilder":
        self._recording_info = info
        return self

    def set_sleep_metrics(self, metrics: Dict[str, float]) -> "ReportBuilder":
        self._sleep_metrics = metrics
        return self

    def set_stage_distribution(self, dist: Dict[str, float]) -> "ReportBuilder":
        self._stage_distribution = dist
        return self

    def set_annotations(self, annotations: List[Dict[str, Any]]) -> "ReportBuilder":
        self._annotations = annotations
        return self

    def set_filter_info(self, filters: List[Dict[str, str]]) -> "ReportBuilder":
        self._filter_info = filters
        return self

    def set_ml_results(self, results: List[Dict[str, Any]]) -> "ReportBuilder":
        self._ml_results = results
        return self

    def set_warnings(self, warnings: List[str]) -> "ReportBuilder":
        self._warnings = warnings
        return self

    def set_hypnogram(self, labels: List[str], epoch_len: float) -> "ReportBuilder":
        self._hypnogram_labels = labels
        self._epoch_len = epoch_len
        return self

    def add_figure(self, title: str, base64_png: str) -> "ReportBuilder":
        self._figures.append((title, base64_png))
        return self

    def add_figure_from_array(
        self, title: str, data: np.ndarray, sfreq: float,
        start_s: float = 0, duration_s: float = 30,
    ) -> "ReportBuilder":
        """Render a signal segment as a matplotlib figure and embed it."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import base64

            fig, ax = plt.subplots(figsize=(10, 2))
            i0 = int(start_s * sfreq)
            i1 = min(len(data), int((start_s + duration_s) * sfreq))
            segment = data[i0:i1]
            t = np.linspace(start_s, start_s + len(segment) / sfreq, len(segment))
            ax.plot(t, segment, linewidth=0.5, color="#89B4FA")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("uV")
            ax.set_title(title)
            ax.set_facecolor("#1E1E2E")
            fig.patch.set_facecolor("#1E1E2E")
            ax.tick_params(colors="#CDD6F4")
            ax.xaxis.label.set_color("#CDD6F4")
            ax.yaxis.label.set_color("#CDD6F4")
            ax.title.set_color("#CDD6F4")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode()
            self._figures.append((title, b64))
        except ImportError:
            logger.warning("matplotlib not available for figure export")
        return self

    def add_hypnogram_figure(self) -> "ReportBuilder":
        """Render hypnogram as embedded matplotlib figure."""
        if not self._hypnogram_labels:
            return self
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import base64

            stage_y = {"W": 5, "N1": 4, "N2": 3, "N3": 2, "R": 1, "REM": 1, "U": 0}
            stage_colors = {
                "W": "#EBA0AC", "N1": "#89B4FA", "N2": "#74C7EC",
                "N3": "#89DCEB", "R": "#CBA6F7", "REM": "#CBA6F7", "U": "#6C7086",
            }

            fig, ax = plt.subplots(figsize=(12, 2.5))
            for i, s in enumerate(self._hypnogram_labels):
                t0 = i * self._epoch_len / 3600
                t1 = (i + 1) * self._epoch_len / 3600
                y = stage_y.get(s.upper(), 0)
                c = stage_colors.get(s.upper(), "#6C7086")
                ax.barh(y, t1 - t0, left=t0, height=0.8, color=c, edgecolor="none")

            ax.set_yticks([0, 1, 2, 3, 4, 5])
            ax.set_yticklabels(["U", "REM", "N3", "N2", "N1", "W"])
            ax.set_xlabel("Temps (heures)")
            ax.set_title("Hypnogramme")
            ax.invert_yaxis()
            ax.set_facecolor("#1E1E2E")
            fig.patch.set_facecolor("#1E1E2E")
            ax.tick_params(colors="#CDD6F4")
            ax.xaxis.label.set_color("#CDD6F4")
            ax.title.set_color("#CDD6F4")
            for spine in ax.spines.values():
                spine.set_color("#45475A")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode()
            self._figures.insert(0, ("Hypnogramme", b64))
        except ImportError:
            logger.warning("matplotlib not available for hypnogram export")
        return self

    # ---- Rendering ------------------------------------------------

    def render_html(self) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        sections = []

        # Recording info
        if self._recording_info:
            rows = "".join(
                f"<tr><td class='key'>{k}</td><td>{v}</td></tr>"
                for k, v in self._recording_info.items()
            )
            sections.append(f"""
            <div class="section">
                <h2>Informations Enregistrement</h2>
                <table>{rows}</table>
            </div>""")

        # Figures
        for title, b64 in self._figures:
            sections.append(f"""
            <div class="section">
                <h2>{title}</h2>
                <img src="data:image/png;base64,{b64}" style="max-width:100%;"/>
            </div>""")

        # Sleep metrics
        if self._sleep_metrics:
            rows = "".join(
                f"<tr><td class='key'>{k}</td><td>{v:.2f}</td></tr>"
                for k, v in self._sleep_metrics.items()
            )
            sections.append(f"""
            <div class="section">
                <h2>Architecture du Sommeil</h2>
                <table>{rows}</table>
            </div>""")

        # Stage distribution
        if self._stage_distribution:
            bars = ""
            for stage, pct in self._stage_distribution.items():
                bars += f"""
                <div class="bar-row">
                    <span class="bar-label">{stage}</span>
                    <div class="bar-outer">
                        <div class="bar-inner" style="width:{pct:.0f}%"></div>
                    </div>
                    <span class="bar-value">{pct:.1f}%</span>
                </div>"""
            sections.append(f"""
            <div class="section">
                <h2>Distribution des Stades</h2>
                {bars}
            </div>""")

        # Annotations
        if self._annotations:
            rows = "".join(
                f"<tr><td>{a.get('onset', 0):.1f}s</td>"
                f"<td>{a.get('duration', 0):.1f}s</td>"
                f"<td>{a.get('type', '')}</td>"
                f"<td>{a.get('label', '')}</td></tr>"
                for a in self._annotations[:100]
            )
            sections.append(f"""
            <div class="section">
                <h2>Annotations ({len(self._annotations)})</h2>
                <table>
                    <tr><th>Debut</th><th>Duree</th><th>Type</th><th>Label</th></tr>
                    {rows}
                </table>
            </div>""")

        # ML results
        if self._ml_results:
            rows = "".join(
                f"<tr><td>{r.get('backend', '')}</td>"
                f"<td>{r.get('accuracy', 0):.2%}</td>"
                f"<td>{r.get('kappa', 0):.3f}</td>"
                f"<td>{r.get('macro_f1', 0):.3f}</td></tr>"
                for r in self._ml_results
            )
            sections.append(f"""
            <div class="section">
                <h2>Resultats ML/DL</h2>
                <table>
                    <tr><th>Backend</th><th>Accuracy</th><th>Kappa</th><th>Macro-F1</th></tr>
                    {rows}
                </table>
            </div>""")

        # Warnings
        if self._warnings:
            items = "".join(f"<li>{w}</li>" for w in self._warnings)
            sections.append(f"""
            <div class="section warnings">
                <h2>Alertes</h2>
                <ul>{items}</ul>
            </div>""")

        body = "\n".join(sections)

        return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<title>{self._title}</title>
<style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #1E1E2E; color: #CDD6F4;
           margin: 0; padding: 20px; }}
    h1 {{ color: #89B4FA; border-bottom: 2px solid #45475A; padding-bottom: 8px; }}
    h2 {{ color: #CBA6F7; margin-top: 20px; }}
    .section {{ background: #313244; border-radius: 8px; padding: 16px; margin: 12px 0; }}
    .warnings {{ border-left: 3px solid #F9E2AF; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #45475A; }}
    th {{ color: #89B4FA; }}
    .key {{ color: #94E2D5; font-weight: bold; width: 200px; }}
    .bar-row {{ display: flex; align-items: center; margin: 4px 0; }}
    .bar-label {{ width: 40px; font-weight: bold; }}
    .bar-outer {{ flex: 1; background: #45475A; border-radius: 4px; height: 16px; margin: 0 8px; }}
    .bar-inner {{ background: #89B4FA; height: 100%; border-radius: 4px; }}
    .bar-value {{ width: 50px; text-align: right; }}
    .footer {{ color: #6C7086; text-align: center; margin-top: 30px; font-size: 0.9em; }}
    img {{ border-radius: 4px; }}
</style>
</head>
<body>
<h1>{self._title}</h1>
<p style="color:#6C7086">Genere le {now}</p>
{body}
<div class="footer">CESA - Clinical EEG & Sleep Analysis</div>
</body>
</html>"""

    def save_html(self, path: str) -> Path:
        p = Path(path)
        p.write_text(self.render_html(), encoding="utf-8")
        logger.info("Report saved to %s", p)
        return p

    def save_pdf(self, path: str) -> Optional[Path]:
        """Save report as PDF (requires weasyprint or wkhtmltopdf)."""
        try:
            from weasyprint import HTML
            html_str = self.render_html()
            HTML(string=html_str).write_pdf(path)
            return Path(path)
        except ImportError:
            pass

        # Fallback: save HTML alongside with .pdf.html extension
        html_path = path + ".html"
        self.save_html(html_path)
        logger.warning(
            "weasyprint not available; saved HTML to %s", html_path
        )
        return Path(html_path)
