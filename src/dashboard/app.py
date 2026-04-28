from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Structured Products Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .metric-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 1rem;
        background: rgba(128,128,128,0.04);
    }
    .small-muted {color: #777; font-size: 0.9rem;}
    .warning-box {
        border-left: 4px solid #f39c12;
        padding: 0.75rem 1rem;
        background: rgba(243, 156, 18, 0.08);
        border-radius: 8px;
    }
    .ok-box {
        border-left: 4px solid #2ecc71;
        padding: 0.75rem 1rem;
        background: rgba(46, 204, 113, 0.08);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Paths and loading helpers
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_DIR = ROOT / "reports" / "dashboard_exports"

REQUIRED_TABLES = {
    "priced_lines": "priced_lines.csv",
    "risk_safe_totals": "risk_safe_totals.csv",
    "risk_by_product_class": "risk_by_product_class.csv",
    "risk_by_underlying": "risk_by_underlying.csv",
    "risk_by_maturity": "risk_by_maturity.csv",
    "risk_by_pillar": "risk_by_pillar.csv",
    "stress_summary": "stress_summary.csv",
    "stress_pnl_by_position": "stress_pnl_by_position.csv",
    "top_vega": "top_vega.csv",
    "economic_flags": "economic_flags.csv",
    "quality_summary": "quality_summary.csv",
}

NUMERIC_COLUMNS = {
    "price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "dv01",
    "gross_price",
    "line_count",
    "standard_error",
    "ci_low",
    "ci_high",
    "base_price",
    "scenario_price",
    "pnl",
    "pnl_vs_base",
    "value",
    "quantity",
    "notional",
    "booking_notional",
    "contract_multiplier",
}


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    for col in df.columns:
        if col in NUMERIC_COLUMNS or col.endswith("_price") or col.endswith("_pnl"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_all_tables(export_dir: str) -> dict[str, pd.DataFrame]:
    base = Path(export_dir)
    return {name: load_csv(str(base / filename)) for name, filename in REQUIRED_TABLES.items()}


def format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value) >= 1_000_000:
        return f"{value:,.{digits}f}"
    if abs(value) >= 1_000:
        return f"{value:,.{digits}f}"
    return f"{value:.{digits}f}"


def filter_df(
    df: pd.DataFrame,
    portfolios: list[str],
    currencies: list[str],
    product_classes: list[str],
    underlyings: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if portfolios and "portfolio" in out.columns:
        out = out[out["portfolio"].astype(str).isin(portfolios)]
    if currencies and "risk_currency" in out.columns:
        out = out[out["risk_currency"].astype(str).isin(currencies)]
    elif currencies and "currency" in out.columns:
        out = out[out["currency"].astype(str).isin(currencies)]
    if product_classes and "product_class" in out.columns:
        out = out[out["product_class"].astype(str).isin(product_classes)]
    if underlyings and "underlying" in out.columns:
        out = out[out["underlying"].astype(str).isin(underlyings)]
    return out


def available_values(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def sum_metric(df: pd.DataFrame, metric: str) -> float:
    if df.empty or metric not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[metric], errors="coerce").fillna(0.0).sum())


def show_dataframe(df: pd.DataFrame, *, height: int = 420) -> None:
    if df.empty:
        st.info("Aucune donnée disponible pour cette vue.")
        return
    st.dataframe(df, use_container_width=True, height=height)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str | None = None) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("Graphique indisponible : colonnes manquantes.")
        return
    chart_df = df[[x, y]].dropna().copy()
    if chart_df.empty:
        st.info("Graphique indisponible : données vides.")
        return
    if title:
        st.subheader(title)
    st.bar_chart(chart_df.set_index(x)[y])


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title("📊 Dashboard")
st.sidebar.caption("Structured Products Pricer")

export_dir_input = st.sidebar.text_input(
    "Dossier des exports CSV",
    value=str(DEFAULT_EXPORT_DIR),
    help="Chemin vers reports/dashboard_exports. Le dashboard ne relance pas le pricing.",
)

export_dir = Path(export_dir_input).expanduser().resolve()
tables = load_all_tables(str(export_dir))
priced_lines = tables["priced_lines"]

missing = [filename for name, filename in REQUIRED_TABLES.items() if tables[name].empty]
if missing:
    st.sidebar.warning(f"Fichiers absents ou vides : {len(missing)}")
else:
    st.sidebar.success("Tous les exports sont chargés")

portfolio_filter = st.sidebar.multiselect(
    "Portefeuille",
    available_values(priced_lines, "portfolio"),
)
currency_filter = st.sidebar.multiselect(
    "Devise de risque",
    available_values(priced_lines, "risk_currency"),
)
product_class_filter = st.sidebar.multiselect(
    "Classe produit",
    available_values(priced_lines, "product_class"),
)
underlying_filter = st.sidebar.multiselect(
    "Sous-jacent",
    available_values(priced_lines, "underlying"),
)

filtered_lines = filter_df(
    priced_lines,
    portfolio_filter,
    currency_filter,
    product_class_filter,
    underlying_filter,
)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.title("📈 Structured Products Pricer — Dashboard")
st.markdown(
    """
    Ce dashboard présente les résultats du pricer multi-produits : valorisation ligne à ligne,
    agrégations de risques, stress tests et contrôles de qualité. Les résultats sont lus depuis
    les exports CSV générés par le notebook final, afin d'éviter de relancer la calibration ou le
    Monte Carlo pendant la soutenance.
    """
)

if not export_dir.exists():
    st.error(f"Le dossier d'exports n'existe pas : {export_dir}")
    st.stop()

if priced_lines.empty:
    st.error(
        "Le fichier priced_lines.csv est manquant ou vide. Lance d'abord le notebook final "
        "pour générer les exports dans reports/dashboard_exports/."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab_overview, tab_lines, tab_risk, tab_stress, tab_quality, tab_doc = st.tabs(
    [
        "Vue d'ensemble",
        "Pricing lignes",
        "Risques agrégés",
        "Stress tests",
        "Qualité & limites",
        "Documentation",
    ]
)


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------

with tab_overview:
    st.header("Vue d'ensemble portefeuille")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes pricées", f"{len(filtered_lines):,}")
    c2.metric("Portefeuilles", f"{filtered_lines['portfolio'].nunique() if 'portfolio' in filtered_lines else 0}")
    c3.metric("Classes produit", f"{filtered_lines['product_class'].nunique() if 'product_class' in filtered_lines else 0}")
    c4.metric("Statuts pricing", ", ".join(available_values(filtered_lines, "status")) or "—")

    st.divider()

    by_ccy = pd.DataFrame()
    by_class = pd.DataFrame()

    if "risk_currency" in filtered_lines.columns and "price" in filtered_lines.columns:
        by_ccy = (
            filtered_lines.groupby("risk_currency", dropna=False)["price"]
            .sum()
            .reset_index()
            .sort_values("price", key=lambda s: s.abs(), ascending=False)
        )

    if "product_class" in filtered_lines.columns and "price" in filtered_lines.columns:
        by_class = (
            filtered_lines.groupby("product_class", dropna=False)
            .agg(price=("price", "sum"), line_count=("product_class", "size"))
            .reset_index()
            .sort_values("price", key=lambda s: s.abs(), ascending=False)
        )

    # Important: on affiche les graphiques dans une première ligne et les tableaux
    # dans un expander séparé. Cela évite l'effet de superposition / tassement visuel
    # observé quand st.dataframe et st.bar_chart sont empilés dans la même colonne.
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Prix par devise de risque")
        if not by_ccy.empty:
            st.bar_chart(by_ccy.set_index("risk_currency")["price"], height=320)
        else:
            st.info("Colonne risk_currency ou price absente.")

    with chart_right:
        st.subheader("Prix par classe produit")
        if not by_class.empty:
            st.bar_chart(by_class.set_index("product_class")["price"], height=320)
        else:
            st.info("Colonne product_class ou price absente.")

    with st.expander("Voir les tableaux détaillés de la vue d'ensemble", expanded=False):
        table_left, table_right = st.columns(2)
        with table_left:
            st.caption("Agrégation par devise de risque")
            show_dataframe(by_ccy, height=220)
        with table_right:
            st.caption("Agrégation par classe produit")
            show_dataframe(by_class, height=300)

    st.subheader("Interprétation financière")
    st.markdown(
        """
        - Les totaux sont affichés par **devise de risque**. Le dashboard ne convertit pas EUR et USD en une devise commune.
        - Les produits de taux portent principalement du **rho / DV01**.
        - Les produits optionnels portent du **delta, gamma et vega**.
        - Les produits autocallables et barrières peuvent utiliser des Greeks numériques selon la configuration du notebook.
        - Un prix élevé doit d'abord être rapproché du nominal ou de la quantité de position avant d'être interprété comme une anomalie de modèle.
        """
    )


# -----------------------------------------------------------------------------
# Line pricing
# -----------------------------------------------------------------------------

with tab_lines:
    st.header("Pricing ligne à ligne")

    cols = [
        "portfolio",
        "product_id",
        "product_type",
        "product_class",
        "underlying",
        "risk_currency",
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "dv01",
        "maturity_bucket",
        "strike_bucket",
        "status",
        "error_message",
    ]
    cols = [c for c in cols if c in filtered_lines.columns]

    search = st.text_input("Recherche produit / sous-jacent", value="")
    view = filtered_lines.copy()
    if search:
        mask = pd.Series(False, index=view.index)
        for col in ["product_id", "product_type", "product_class", "underlying", "portfolio"]:
            if col in view.columns:
                mask |= view[col].astype(str).str.contains(search, case=False, na=False)
        view = view[mask]

    show_dataframe(view[cols], height=520)

    with st.expander("Colonnes de taille économique"):
        size_cols = [
            "portfolio",
            "product_id",
            "product_class",
            "quantity",
            "notional",
            "booking_notional",
            "contract_multiplier",
            "price_unit",
            "position_sign",
        ]
        size_cols = [c for c in size_cols if c in filtered_lines.columns]
        show_dataframe(filtered_lines[size_cols], height=320)

    st.markdown(
        """
        **Lecture financière.** Une ligne correspond à une position de portefeuille. Le prix est signé selon le sens de position.
        Les colonnes `quantity`, `notional`, `booking_notional` et `contract_multiplier`, lorsqu'elles existent, servent à documenter
        la convention de taille utilisée dans l'inventaire. Si certains prix semblent élevés, la première vérification est le nominal
        ou la quantité de la position.
        """
    )


# -----------------------------------------------------------------------------
# Risk aggregation
# -----------------------------------------------------------------------------

with tab_risk:
    st.header("Risques agrégés")

    risk_table_name = st.selectbox(
        "Table d'agrégation",
        [
            "risk_safe_totals",
            "risk_by_product_class",
            "risk_by_underlying",
            "risk_by_maturity",
            "risk_by_pillar",
        ],
    )
    risk_df = tables[risk_table_name]
    risk_df = filter_df(risk_df, portfolio_filter, currency_filter, product_class_filter, underlying_filter)

    show_dataframe(risk_df, height=420)

    metric = st.selectbox("Métrique", [m for m in ["price", "delta", "gamma", "vega", "theta", "rho", "dv01"] if m in risk_df.columns])
    dimension_candidates = [c for c in ["portfolio", "product_class", "underlying", "risk_underlying", "maturity_bucket", "strike_bucket", "risk_currency"] if c in risk_df.columns]
    if dimension_candidates:
        dimension = st.selectbox("Dimension graphique", dimension_candidates)
        chart_df = risk_df.groupby(dimension, dropna=False)[metric].sum().reset_index()
        bar_chart(chart_df, dimension, metric, f"{metric} par {dimension}")

    st.markdown(
        """
        **Interprétation financière.** Les agrégations sont utiles pour identifier les concentrations de risque :
        exposition action par sous-jacent, exposition taux par courbe, maturité dominante, et zones de strike.
        Le `DV01` mesure l'impact d'un déplacement de 1 bp des taux. Le `vega` mesure l'impact d'un déplacement de volatilité.
        """
    )


# -----------------------------------------------------------------------------
# Stress tests
# -----------------------------------------------------------------------------

with tab_stress:
    st.header("Stress tests")

    stress_summary = tables["stress_summary"]
    stress_pnl = tables["stress_pnl_by_position"]

    if stress_summary.empty:
        st.info("stress_summary.csv absent ou vide.")
    else:
        stress_summary_view = filter_df(stress_summary, portfolio_filter, currency_filter, [], [])
        st.subheader("Résumé par scénario")
        show_dataframe(stress_summary_view, height=300)

        if {"scenario", "pnl_vs_base"}.issubset(stress_summary_view.columns):
            scenario_chart = stress_summary_view.groupby("scenario", dropna=False)["pnl_vs_base"].sum().reset_index()
            bar_chart(scenario_chart, "scenario", "pnl_vs_base", "P&L vs scénario base")

    st.subheader("P&L par position")
    if stress_pnl.empty:
        st.info("stress_pnl_by_position.csv absent ou vide.")
    else:
        stress_pnl_view = filter_df(stress_pnl, portfolio_filter, currency_filter, product_class_filter, underlying_filter)
        scenario_values = available_values(stress_pnl_view, "scenario")
        selected_scenarios = st.multiselect("Scénarios", scenario_values, default=scenario_values[:3])
        if selected_scenarios and "scenario" in stress_pnl_view.columns:
            stress_pnl_view = stress_pnl_view[stress_pnl_view["scenario"].astype(str).isin(selected_scenarios)]
        show_dataframe(stress_pnl_view, height=420)

    st.markdown(
        """
        **Interprétation financière.** Les stress tests permettent d'expliquer la sensibilité non linéaire du portefeuille :
        un choc spot impacte surtout les produits actions, un choc volatilité impacte les produits optionnels, et un choc taux
        impacte les swaps, obligations et notes à composante zéro-coupon.
        """
    )


# -----------------------------------------------------------------------------
# Quality and limits
# -----------------------------------------------------------------------------

with tab_quality:
    st.header("Qualité, contrôles et limites")

    quality = tables["quality_summary"]
    flags = tables["economic_flags"]

    col1, col2, col3 = st.columns(3)
    priced_count = int((priced_lines.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "priced").sum()) if "status" in priced_lines.columns else len(priced_lines)
    error_count = int((priced_lines.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "error").sum()) if "status" in priced_lines.columns else 0
    flag_count = len(flags) if not flags.empty else 0
    col1.metric("Lignes pricées", f"{priced_count:,}")
    col2.metric("Erreurs pricing", f"{error_count:,}")
    col3.metric("Flags économiques", f"{flag_count:,}")

    if error_count == 0:
        st.markdown('<div class="ok-box">Aucune erreur technique de pricing dans les lignes chargées.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">Certaines lignes sont en erreur. Vérifier la colonne error_message.</div>', unsafe_allow_html=True)

    st.subheader("Résumé qualité")
    show_dataframe(quality, height=240)

    st.subheader("Flags économiques")
    show_dataframe(flags, height=360)

    st.markdown(
        """
        ### Problèmes identifiés et interprétation

        Les éventuels flags ne signifient pas nécessairement que le moteur de pricing est faux. Ils indiquent surtout des points
        à documenter :

        - **Prix optionnels élevés** : souvent dus à la convention `quantity` / `notional` de l'inventaire.
        - **Autocall inférieur à 10** : peut indiquer un prix exprimé en fraction du nominal plutôt qu'en montant monétaire.
        - **Vega négatif** : possible pour certains profils optionnels, notamment les butterflies ou certaines barrières.
        - **Multi-devise** : les totaux EUR et USD ne sont pas agrégés sans couche FX.

        Dans la documentation finale, ces points doivent être présentés comme des limites de convention de booking et non comme
        des erreurs d'agrégation.
        """
    )


# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------

with tab_doc:
    st.header("Documentation fonctionnelle")

    st.markdown(
        """
        ## Objectif de l'application

        L'application permet de visualiser les résultats d'un pricer multi-produits :

        - produits de taux : swaps, obligations ou instruments actualisés ;
        - options vanilles et stratégies optionnelles ;
        - options à barrière ;
        - produits structurés à réplication statique ;
        - produits autocallables valorisés par Monte Carlo.

        ## Données utilisées

        Les inputs sont placés dans `data/raw` :

        - `inventory.xlsx` : inventaire de portefeuille ;
        - `options.csv` : panel d'options pour la volatilité implicite ;
        - `rate_curves.parquet` : courbes de taux.

        Le notebook final transforme ces inputs en exports dashboard dans `reports/dashboard_exports`.

        ## Conventions importantes

        - Les prix sont exprimés dans leur devise de risque.
        - Il n'y a pas de conversion FX automatique.
        - Les tailles de position dépendent des colonnes `quantity`, `notional`, `booking_notional` et `contract_multiplier`.
        - Les autocalls peuvent être exprimés en fraction de nominal si le nominal n'est pas harmonisé en amont.

        ## Limites assumées

        - Les produits sont mono-sous-jacent et non quanto.
        - Les surfaces de volatilité dépendent de la qualité du panel d'options disponible.
        - Les stress tests sont des chocs simples, pas des scénarios historiques complets.
        - Les Greeks numériques peuvent être approximatifs pour les produits path-dependent.
        """
    )

    st.subheader("Fichiers chargés")
    file_status = pd.DataFrame(
        [
            {"table": name, "file": filename, "rows": len(tables[name]), "loaded": not tables[name].empty}
            for name, filename in REQUIRED_TABLES.items()
        ]
    )
    show_dataframe(file_status, height=360)
