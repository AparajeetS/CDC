"""Minimal pure-Python SVG plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _nice_limits(values: np.ndarray):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        pad = 1.0 if vmin == 0.0 else 0.05 * abs(vmin)
        return vmin - pad, vmax + pad
    pad = 0.06 * (vmax - vmin)
    return vmin - pad, vmax + pad


def _polyline(points, color: str, width: float = 2.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}"'
        f'{dash_attr} points="{pts}" />'
    )


def write_three_panel_svg(output_path: Path, panels: list[dict]) -> None:
    width = 1440
    height = 470
    margin_left = 70
    margin_right = 30
    margin_top = 28
    margin_bottom = 58
    gap = 36
    panel_width = (width - margin_left - margin_right - 2 * gap) / 3.0
    panel_height = height - margin_top - margin_bottom

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white" />',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111} .small{font-size:13px} .label{font-size:15px} .title{font-size:16px;font-weight:bold}</style>',
    ]

    for index, panel in enumerate(panels):
        x0 = margin_left + index * (panel_width + gap)
        y0 = margin_top
        x1 = x0 + panel_width
        y1 = y0 + panel_height

        x_values = np.concatenate([np.asarray(line["x"], dtype=float) for line in panel["lines"]])
        y_values = np.concatenate([np.asarray(line["y"], dtype=float) for line in panel["lines"]])
        xmin, xmax = _nice_limits(x_values)
        ymin, ymax = _nice_limits(y_values)

        def map_x(x):
            return x0 + (x - xmin) / (xmax - xmin) * panel_width

        def map_y(y):
            return y1 - (y - ymin) / (ymax - ymin) * panel_height

        svg.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{panel_width:.1f}" height="{panel_height:.1f}" fill="none" stroke="#222" stroke-width="1.2" />')

        for frac, label in zip(np.linspace(0.0, 1.0, 5), np.linspace(xmin, xmax, 5)):
            xt = x0 + frac * panel_width
            svg.append(f'<line x1="{xt:.1f}" y1="{y1:.1f}" x2="{xt:.1f}" y2="{y1+6:.1f}" stroke="#222" stroke-width="1" />')
            svg.append(f'<text class="small" x="{xt:.1f}" y="{y1+22:.1f}" text-anchor="middle">{label:.2f}</text>')

        for frac, label in zip(np.linspace(0.0, 1.0, 5), np.linspace(ymin, ymax, 5)):
            yt = y1 - frac * panel_height
            svg.append(f'<line x1="{x0-6:.1f}" y1="{yt:.1f}" x2="{x0:.1f}" y2="{yt:.1f}" stroke="#222" stroke-width="1" />')
            svg.append(f'<text class="small" x="{x0-10:.1f}" y="{yt+4:.1f}" text-anchor="end">{label:.2f}</text>')

        for line in panel["lines"]:
            pts = [(map_x(x), map_y(y)) for x, y in zip(line["x"], line["y"])]
            svg.append(_polyline(pts, line.get("color", "#1f77b4"), dash=line.get("dash")))

        svg.append(f'<text class="title" x="{x0 + panel_width/2:.1f}" y="{y0-8:.1f}" text-anchor="middle">{panel["title"]}</text>')
        svg.append(f'<text class="label" x="{x0 + panel_width/2:.1f}" y="{height-16:.1f}" text-anchor="middle">{panel["xlabel"]}</text>')
        svg.append(
            f'<text class="label" x="{x0-52:.1f}" y="{y0 + panel_height/2:.1f}" text-anchor="middle" transform="rotate(-90 {x0-52:.1f},{y0 + panel_height/2:.1f})">{panel["ylabel"]}</text>'
        )

        legend_x = x0 + 12
        legend_y = y0 + 16
        for line_index, line in enumerate(panel["lines"]):
            yy = legend_y + line_index * 18
            dash = line.get("dash")
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(
                f'<line x1="{legend_x:.1f}" y1="{yy:.1f}" x2="{legend_x+24:.1f}" y2="{yy:.1f}" stroke="{line.get("color", "#1f77b4")}" stroke-width="2"{dash_attr} />'
            )
            svg.append(f'<text class="small" x="{legend_x+30:.1f}" y="{yy+4:.1f}">{line["label"]}</text>')

    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")
