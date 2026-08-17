"""Centralized PrepWise AI design system (Streamlit CSS)."""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------
PRIMARY_NAVY = "#071A3D"
DARK_NAVY = "#04132F"
NAVY_BLUE = "#102B57"
PRIMARY_BLUE = "#315BFF"
BRIGHT_BLUE = "#4368FF"
ACTIVE_BLUE = "#2445B8"
LIGHT_BLUE = "#EEF3FF"
VERY_LIGHT_BLUE = "#F6F8FF"
WHITE = "#FFFFFF"
PAGE_BG = "#F8FAFD"
TEXT_PRIMARY = "#0B1833"
TEXT_SECONDARY = "#53627A"
TEXT_MUTED = "#7A879A"
SIDEBAR_MUTED = "#B9C5DA"
SIDEBAR_LABEL = "#9AA9C2"
BORDER = "#E1E7F0"
BORDER_DARK = "#CBD5E1"
SUCCESS = "#22A06B"
WARNING = "#F4B740"
ERROR = "#E05252"

# Back-compat aliases used by older shell imports
NAVY = PRIMARY_NAVY
BG = PAGE_BG
TEXT = TEXT_PRIMARY
ACCENT = PRIMARY_BLUE


def apply_global_styles(*, sidebar_collapsed: bool = False) -> None:
    """Inject global CSS once per run."""
    sidebar_width = "78px" if sidebar_collapsed else "262px"

    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, button, input, label {{
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }}

    .stApp {{
        background: {PAGE_BG};
        color: {TEXT_PRIMARY};
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        background: transparent;
        height: 0;
    }}

    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1.75rem !important;
        padding-right: 1.75rem !important;
        max-width: 1180px;
    }}

    /* ------------------------------------------------------------------ */
    /* Sidebar — dark navy                                                */
    /* ------------------------------------------------------------------ */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PRIMARY_NAVY} 0%, {DARK_NAVY} 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
        min-width: {sidebar_width} !important;
        max-width: {sidebar_width} !important;
        width: {sidebar_width} !important;
        transition: min-width 0.2s ease, max-width 0.2s ease, width 0.2s ease;
    }}
    section[data-testid="stSidebar"] > div {{
        background: transparent !important;
        padding-top: 0.85rem;
        padding-left: 0.7rem;
        padding-right: 0.7rem;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[kind="header"] {{
        display: none !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {WHITE} !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.1) !important;
        margin: 0.75rem 0 !important;
    }}

    /* Sidebar nav buttons — must beat global secondary rules */
    section[data-testid="stSidebar"] div.stButton > button,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"],
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"],
    section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]) {{
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100%;
        border-radius: 9px !important;
        border: none !important;
        background: transparent !important;
        color: {WHITE} !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.65rem 0.75rem !important;
        transition: background 0.15s ease !important;
        box-shadow: none !important;
        transform: none !important;
    }}
    section[data-testid="stSidebar"] div.stButton > button span,
    section[data-testid="stSidebar"] div.stButton > button p,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] span,
    section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]) span {{
        color: {WHITE} !important;
    }}
    section[data-testid="stSidebar"] div.stButton > button:hover,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]):hover {{
        background: rgba(255,255,255,0.06) !important;
        color: {WHITE} !important;
        border: none !important;
        transform: none !important;
        box-shadow: none !important;
    }}
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"],
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] {{
        background: linear-gradient(90deg, {ACTIVE_BLUE}, {PRIMARY_BLUE}) !important;
        color: {WHITE} !important;
    }}
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] span,
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] span {{
        color: {WHITE} !important;
    }}
    section[data-testid="stSidebar"] div.stButton > button:disabled {{
        opacity: 0.35 !important;
        color: {SIDEBAR_MUTED} !important;
    }}

    /* Collapsed: hide text chrome */
    {("section[data-testid='stSidebar'] .pw-hide-collapsed { display: none !important; }" if sidebar_collapsed else "")}
    {("section[data-testid='stSidebar'] .pw-brand-block { align-items: center !important; }" if sidebar_collapsed else "")}
    {("section[data-testid='stSidebar'] div.stButton > button { justify-content: center !important; padding: 0.65rem 0.35rem !important; }" if sidebar_collapsed else "")}

    /* ------------------------------------------------------------------ */
    /* Typography                                                         */
    /* ------------------------------------------------------------------ */
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}
    .stApp p, .stApp label, .stApp li,
    .stMarkdown p, .stMarkdown li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaption"], .stCaption, small {{
        color: {TEXT_PRIMARY};
    }}
    [data-testid="stCaption"], .stCaption, small {{
        color: {TEXT_SECONDARY} !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Metrics                                                            */
    /* ------------------------------------------------------------------ */
    [data-testid="stMetric"] {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 8px rgba(7,26,61,0.04);
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700 !important;
        font-size: 1.65rem !important;
    }}
    [data-testid="stMetricDelta"] {{
        color: {TEXT_SECONDARY} !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Buttons — primary / secondary / download                           */
    /* ------------------------------------------------------------------ */
    div.stButton > button,
    div.stDownloadButton > button,
    [data-testid="stDownloadButton"] button {{
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.1rem !important;
        transition: background 0.2s ease, border-color 0.2s ease,
                    box-shadow 0.2s ease, transform 0.2s ease !important;
    }}

    /* Secondary (light) */
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="baseButton-secondary"],
    div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {{
        background: {WHITE} !important;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {BORDER_DARK} !important;
        box-shadow: none !important;
    }}
    div.stButton > button[kind="secondary"] span,
    div.stButton > button[data-testid="baseButton-secondary"] span,
    div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) span {{
        color: {TEXT_PRIMARY} !important;
    }}
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="baseButton-secondary"]:hover,
    div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {{
        background: {LIGHT_BLUE} !important;
        border-color: {PRIMARY_BLUE} !important;
        color: {TEXT_PRIMARY} !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(49,91,255,0.10) !important;
    }}

    /* Primary + download (blue + white text) */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"],
    div.stDownloadButton > button,
    [data-testid="stDownloadButton"] button {{
        background: {PRIMARY_BLUE} !important;
        color: {WHITE} !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(49,91,255,0.18) !important;
    }}
    div.stButton > button[kind="primary"] span,
    div.stButton > button[kind="primary"] p,
    div.stButton > button[data-testid="baseButton-primary"] span,
    div.stDownloadButton > button span,
    div.stDownloadButton > button p,
    [data-testid="stDownloadButton"] button span,
    [data-testid="stDownloadButton"] button p {{
        color: {WHITE} !important;
    }}
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover,
    div.stDownloadButton > button:hover,
    [data-testid="stDownloadButton"] button:hover {{
        background: #2448D8 !important;
        color: {WHITE} !important;
        border: none !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(49,91,255,0.22) !important;
    }}
    div.stButton > button[kind="primary"]:hover span,
    div.stDownloadButton > button:hover span,
    [data-testid="stDownloadButton"] button:hover span {{
        color: {WHITE} !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Inputs / tables / expanders                                        */
    /* ------------------------------------------------------------------ */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 600 !important;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
        background: {WHITE};
        box-shadow: 0 2px 8px rgba(7,26,61,0.04);
    }}
    [data-testid="stExpander"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        background: {WHITE};
        box-shadow: 0 2px 8px rgba(7,26,61,0.04);
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stFileUploader"] {{
        background: {WHITE};
        border: 1px dashed {BORDER_DARK};
        border-radius: 12px;
        padding: 1rem;
    }}
    .stAlert {{
        border-radius: 10px !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Custom components                                                  */
    /* ------------------------------------------------------------------ */
    .pw-topbar {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 2px 8px rgba(7,26,61,0.04);
    }}
    .pw-topbar-brand {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }}
    .pw-topbar-title {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700;
        font-size: 1rem;
        margin: 0;
        line-height: 1.2;
    }}
    .pw-topbar-sub {{
        color: {TEXT_SECONDARY} !important;
        font-size: 0.78rem;
        margin: 0;
        line-height: 1.2;
    }}
    .pw-dataset-chip {{
        background: {VERY_LIGHT_BLUE};
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 0.45rem 0.85rem;
        font-size: 0.875rem;
        font-weight: 600;
        white-space: nowrap;
        display: inline-block;
    }}
    .pw-dataset-empty {{
        color: {TEXT_SECONDARY} !important;
        background: {WHITE};
    }}

    .pw-page-title {{
        color: {TEXT_PRIMARY} !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }}
    .pw-page-sub {{
        color: {TEXT_SECONDARY} !important;
        font-size: 0.95rem;
        margin: 0 0 1.25rem 0;
        line-height: 1.5;
    }}

    .pw-section {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(7,26,61,0.05);
        color: {TEXT_PRIMARY} !important;
    }}
    .pw-section-title {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700;
        font-size: 1rem;
        margin: 0 0 0.85rem 0;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }}

    /* Sidebar brand */
    .pw-brand-block {{
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        margin-bottom: 0.85rem;
        padding: 0.25rem 0.35rem;
    }}
    .pw-brand-collapsed {{
        margin-bottom: 0;
        padding: 0;
    }}
    .pw-logo-row {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }}
    .pw-logo-mark {{
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: linear-gradient(135deg, {PRIMARY_BLUE}, {BRIGHT_BLUE});
        display: flex;
        align-items: center;
        justify-content: center;
        color: {WHITE};
        font-weight: 800;
        font-size: 0.85rem;
        flex-shrink: 0;
    }}
    .pw-logo-title {{
        color: {WHITE} !important;
        font-weight: 700;
        font-size: 1.15rem;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }}
    .pw-logo-tag {{
        color: {SIDEBAR_MUTED} !important;
        font-size: 0.8rem;
        margin: 0.15rem 0 0 0;
        line-height: 1.35;
    }}
    .pw-nav-label {{
        color: {SIDEBAR_LABEL} !important;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0.9rem 0 0.4rem 0.35rem;
    }}
    .pw-sidebar-help {{
        margin-top: 1.25rem;
        padding: 0.85rem;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        color: {SIDEBAR_MUTED} !important;
        font-size: 0.78rem;
        line-height: 1.45;
    }}
    .pw-sidebar-help strong {{
        color: {WHITE} !important;
        display: block;
        margin-bottom: 0.25rem;
        font-size: 0.85rem;
    }}

    /* Vertical workflow stepper (sidebar) */
    .pw-stepper {{
        display: flex;
        flex-direction: column;
        gap: 0;
        margin: 0.25rem 0 0.5rem 0;
        padding-left: 0.35rem;
    }}
    .pw-stepper-item {{
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        position: relative;
        padding: 0.15rem 0;
        min-height: 2.1rem;
    }}
    .pw-stepper-rail {{
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 18px;
        flex-shrink: 0;
    }}
    .pw-stepper-dot {{
        width: 14px;
        height: 14px;
        border-radius: 50%;
        border: 2px solid rgba(255,255,255,0.35);
        background: transparent;
        z-index: 1;
        flex-shrink: 0;
    }}
    .pw-stepper-dot.done {{
        background: {PRIMARY_BLUE};
        border-color: {PRIMARY_BLUE};
    }}
    .pw-stepper-dot.current {{
        background: {BRIGHT_BLUE};
        border-color: {BRIGHT_BLUE};
        box-shadow: 0 0 0 3px rgba(67,104,255,0.25);
    }}
    .pw-stepper-line {{
        width: 2px;
        flex: 1;
        min-height: 14px;
        background: rgba(255,255,255,0.15);
        margin: 2px 0;
    }}
    .pw-stepper-label {{
        color: {SIDEBAR_MUTED} !important;
        font-size: 0.82rem;
        font-weight: 500;
        padding-top: 0;
        line-height: 1.2;
    }}
    .pw-stepper-item.current .pw-stepper-label {{
        color: {WHITE} !important;
        font-weight: 600;
    }}
    .pw-stepper-item.done .pw-stepper-label {{
        color: #D6E0F0 !important;
    }}

    /* Home hero — compact centered SaaS landing */
    .pw-hero.home-hero {{
        text-align: center;
        padding: 2.5rem 1rem 0;
        margin: 0 auto 0;
        position: relative;
        width: 100%;
        max-width: 920px;
    }}
    .pw-hero-inner {{
        position: relative;
        z-index: 1;
        max-width: 900px;
        margin: 0 auto;
        text-align: center;
    }}
    .pw-hero .pw-hero-title,
    .pw-hero h1.pw-hero-title {{
        font-size: clamp(2rem, 4.5vw, 3.25rem);
        font-weight: 700;
        line-height: 1.1;
        margin: 0 0 0.75rem 0;
        letter-spacing: -0.02em;
        text-align: center;
        color: #2453C5 !important;
    }}
    .pw-hero-sub {{
        color: #53657F;
        font-size: clamp(1.125rem, 2.2vw, 1.375rem);
        font-weight: 600;
        line-height: 1.4;
        margin: 0 0 1.375rem 0;
        text-align: center;
    }}
    .pw-hero-desc {{
        color: #596B84;
        font-size: clamp(0.9375rem, 1.6vw, 1.0625rem);
        font-weight: 400;
        line-height: 1.7;
        text-align: center;
        margin: 0 auto;
    }}
    .pw-hero-lead {{
        max-width: 700px;
        margin: 0 auto 0.875rem auto;
    }}
    .pw-hero-support {{
        color: #718096;
        font-size: clamp(0.875rem, 1.4vw, 0.9375rem);
        font-weight: 400;
        line-height: 1.6;
        max-width: 700px;
        margin: 0 auto;
        text-align: center;
    }}
    .pw-hero-cta-marker ~ div[data-testid="stHorizontalBlock"] {{
        max-width: 920px;
        margin: 1.875rem auto 2rem auto;
    }}
    .pw-hero-cta-marker ~ div[data-testid="stHorizontalBlock"] > div {{
        display: flex;
        justify-content: center;
    }}
    .pw-hero-cta-marker ~ div[data-testid="stHorizontalBlock"] div.stButton {{
        width: auto;
        margin: 0 auto;
    }}
    .pw-hero-cta-marker ~ div[data-testid="stHorizontalBlock"] div.stButton > button {{
        min-height: 48px;
        padding: 0 1.75rem;
        font-size: 0.9375rem;
        font-weight: 600;
        border-radius: 10px;
        background: #315BEA;
        color: {WHITE};
        border: none;
        box-shadow: none;
        transition: background 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .pw-hero-cta-marker ~ div[data-testid="stHorizontalBlock"] div.stButton > button span {{
        color: {WHITE};
    }}
    .pw-hero-cta-marker ~ div[data-testid="stHorizontalBlock"] div.stButton > button:hover {{
        background: #2448D8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(49, 91, 234, 0.22);
    }}
    @media (max-width: 768px) {{
        .pw-hero.home-hero {{
            padding: 2rem 0.75rem 0;
        }}
        .pw-hero .pw-hero-title,
        .pw-hero h1.pw-hero-title {{
            font-size: clamp(1.875rem, 7vw, 2.25rem);
        }}
        .pw-hero-sub {{
            font-size: clamp(1.0625rem, 4vw, 1.25rem);
        }}
    }}

    .pw-how-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.85rem 1.25rem;
    }}
    @media (max-width: 800px) {{
        .pw-how-grid {{ grid-template-columns: 1fr; }}
    }}
    .pw-how-item {{
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
    }}
    .pw-how-num {{
        color: {PRIMARY_BLUE};
        font-weight: 700;
        font-size: 0.8rem;
        min-width: 1.5rem;
    }}
    .pw-how-title {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700;
        font-size: 0.9rem;
        margin: 0 0 0.15rem 0;
    }}
    .pw-how-desc {{
        color: {TEXT_SECONDARY} !important;
        font-size: 0.82rem;
        margin: 0;
        line-height: 1.4;
    }}

    .pw-action-area {{
        margin-top: 1.25rem;
        padding-top: 1rem;
        border-top: 1px solid {BORDER};
    }}
    .pw-progress-strip {{
        padding: 0.9rem 1rem !important;
    }}

    .pw-home-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 4px 20px rgba(7,26,61,0.05);
    }}
    .pw-home-card-title {{
        color: {TEXT_PRIMARY} !important;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
    }}

    .pw-workflow-grid {{
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        gap: 0.65rem;
    }}
    @media (max-width: 1100px) {{
        .pw-workflow-grid {{ grid-template-columns: repeat(4, 1fr); }}
    }}
    @media (max-width: 700px) {{
        .pw-workflow-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .pw-wf-step {{
        text-align: center;
        padding: 0.85rem 0.4rem;
        border-radius: 10px;
        border: 1px solid {BORDER};
        background: {VERY_LIGHT_BLUE};
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .pw-wf-step:hover {{
        border-color: {PRIMARY_BLUE};
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(7,26,61,0.08);
    }}
    .pw-wf-num {{
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: {PRIMARY_NAVY};
        color: {WHITE};
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }}
    .pw-wf-step.current .pw-wf-num {{ background: {PRIMARY_BLUE}; }}
    .pw-wf-step.done .pw-wf-num {{ background: {SUCCESS}; }}
    .pw-wf-title {{
        color: {TEXT_PRIMARY} !important;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0 0 0.2rem 0;
    }}
    .pw-wf-desc {{
        color: {TEXT_MUTED} !important;
        font-size: 0.7rem;
        margin: 0;
        line-height: 1.3;
    }}

    .pw-feature-grid {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.75rem;
    }}
    @media (max-width: 1100px) {{
        .pw-feature-grid {{ grid-template-columns: repeat(3, 1fr); }}
    }}
    @media (max-width: 700px) {{
        .pw-feature-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    .pw-feature-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.1rem 1rem;
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
        min-height: 140px;
        display: flex;
        flex-direction: column;
    }}
    .pw-feature-card:hover {{
        border-color: {PRIMARY_BLUE};
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(7,26,61,0.08);
    }}
    .pw-feature-card:hover .pw-feature-arrow {{
        color: {PRIMARY_BLUE} !important;
    }}
    .pw-feature-icon {{
        width: 36px;
        height: 36px;
        border-radius: 9px;
        background: {LIGHT_BLUE};
        color: {PRIMARY_BLUE};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }}
    .pw-feature-title {{
        color: {TEXT_PRIMARY} !important;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
    }}
    .pw-feature-desc {{
        color: {TEXT_SECONDARY} !important;
        font-size: 0.78rem;
        line-height: 1.4;
        margin: 0;
        flex: 1;
    }}
    .pw-feature-arrow {{
        color: {TEXT_MUTED} !important;
        font-weight: 700;
        margin-top: 0.65rem;
        transition: color 0.2s ease;
    }}

    .pw-empty {{
        text-align: center;
        padding: 2rem 1rem;
        color: {TEXT_SECONDARY} !important;
    }}
    .pw-empty-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
        opacity: 0.7;
    }}
    .pw-empty-title {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700;
        font-size: 1rem;
        margin: 0 0 0.35rem 0;
    }}
    .pw-empty-desc {{
        color: {TEXT_SECONDARY} !important;
        font-size: 0.875rem;
        margin: 0;
    }}

    .pw-status-ok {{
        color: {SUCCESS} !important;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    .pw-badge {{
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid {BORDER};
    }}
    .pw-badge-keep {{ background: #E8F8F0; color: {SUCCESS}; border-color: #B7E4CE; }}
    .pw-badge-review {{ background: #FFF8E8; color: #B7791F; border-color: #F5D78E; }}
    .pw-badge-remove {{ background: #FDECEC; color: {ERROR}; border-color: #F5C2C2; }}

    .pw-export-flow {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        color: {TEXT_SECONDARY};
        font-size: 0.9rem;
        font-weight: 600;
    }}
    .pw-export-flow span.pw-pill {{
        background: {LIGHT_BLUE};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 0.35rem 0.7rem;
    }}
</style>
        """,
        unsafe_allow_html=True,
    )
