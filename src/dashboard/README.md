# Dashboard Streamlit — Structured Products Pricer

Ce dossier contient une application Streamlit prête à être placée dans `src/dashboard`.

## Lancement recommandé

Depuis la racine du projet :

```bash
streamlit run src/dashboard/app.py
```

## Données attendues

L'application lit en priorité les exports CSV produits par le notebook dans :

```text
reports/dashboard_exports/
```

Fichiers utilisés si disponibles :

- `priced_lines.csv`
- `risk_safe_totals.csv`
- `risk_by_product_class.csv`
- `risk_by_underlying.csv`
- `risk_by_maturity.csv`
- `risk_by_pillar.csv`
- `stress_summary.csv`
- `stress_pnl_by_position.csv`
- `top_vega.csv`
- `economic_flags.csv`
- `quality_summary.csv`

Si certains fichiers sont absents, l'application affiche un avertissement et continue avec les autres vues disponibles.

## Objectif

Le dashboard est volontairement robuste : il ne relance pas le pricing, la calibration SVI/SSVI ou le Monte Carlo. Il consomme les exports validés du notebook pour éviter les erreurs au moment de la soutenance.
