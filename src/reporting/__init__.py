"""Reporting: GitHub Issue trade tickets and the static GitHub Pages dashboard."""

from .dashboard_builder import build_dashboard_html
from .issue_builder import build_issue_body

__all__ = ["build_dashboard_html", "build_issue_body"]
