"""Military-green look: Altair chart theme + a little CSS for the Streamlit page."""
import altair as alt

OLIVE_DARK = "#2F3A1E"
ARMY = "#4B5320"
MOSS = "#6B7D2E"
KHAKI = "#B8A65A"
SAND = "#D8CFA4"
INK = "#1F2A14"

POSITIVE = "#4F7A2B"
NEUTRAL = "#B9B58A"
NEGATIVE = "#A5402D"

SENTIMENT_SCALE = alt.Scale(domain=["positive", "neutral", "negative"], range=[POSITIVE, NEUTRAL, NEGATIVE])
# distinct but on-theme: dark green, moss, khaki, brown, slate; "Other" is the muted grey-green
COMPLAINT_COLORS = ["#2F4A1F", "#7A9A3A", "#C2AE5C", "#8C6D46", "#6F8596"]
OTHER_COLOR = "#A9AD9C"


def complaint_scale(order: list[str]) -> alt.Scale:
    colors = [OTHER_COLOR if name == "Other" else COMPLAINT_COLORS[i % len(COMPLAINT_COLORS)]
              for i, name in enumerate(order)]
    return alt.Scale(domain=order, range=colors)


def _altair_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "font": "sans-serif",
            "axis": {
                "labelColor": "#3B4A2A", "titleColor": "#3B4A2A", "gridColor": "#D9DEC6",
                "domainColor": "#A9B08A", "tickColor": "#A9B08A", "labelFontSize": 12, "titleFontSize": 12,
            },
            "legend": {"labelColor": "#3B4A2A", "titleColor": "#3B4A2A", "orient": "top", "direction": "horizontal"},
            "range": {"category": [ARMY, MOSS, KHAKI, SAND]},
        }
    }


def apply_altair_theme() -> None:
    alt.themes.register("military", _altair_theme)
    alt.themes.enable("military")


CSS = f"""
<style>
h1, h2, h3 {{ color: {OLIVE_DARK}; letter-spacing: -0.01em; }}
.rl-banner {{
  background: linear-gradient(120deg, {OLIVE_DARK} 0%, {ARMY} 60%, {MOSS} 100%);
  color: #F3F4EC; padding: 1.2rem 1.6rem; border-radius: 12px; margin-bottom: 1rem;
  border-left: 8px solid {KHAKI};
}}
.rl-banner h1 {{ color: #F3F4EC; margin: 0; padding: 0; font-size: 2rem; }}
.rl-banner p {{ margin: .2rem 0 0; color: #D8DDBF; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stMetric"]) {{
  background: #FBFBF6; border-color: #C9D0AE; border-radius: 10px;
  box-shadow: 0 1px 3px rgba(47, 58, 30, .12);
}}
[data-testid="stMetricValue"] {{ color: {OLIVE_DARK}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
.stTabs [aria-selected="true"] {{ color: {ARMY}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ARMY}; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background-color: {ARMY}; border-color: {ARMY}; color: #F3F4EC;
}}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {{
  background-color: {OLIVE_DARK}; border-color: {OLIVE_DARK};
}}
</style>
"""
