# Documentation du dashboard Streamlit

## Lancement

```bash
streamlit run src/dashboard/app.py
```

## Exports attendus

Le dashboard lit `reports/dashboard_exports/` :

- `priced_lines.csv`
- `risk_by_product_class.csv`
- `risk_by_underlying.csv`
- `risk_by_maturity.csv`
- `risk_by_pillar.csv`
- `risk_safe_totals.csv`
- `stress_summary.csv`
- `stress_pnl_by_position.csv`
- `top_vega.csv`
- `economic_flags.csv`
- `quality_summary.csv`

## Utilisation

La barre latérale permet de filtrer : portefeuille, devise de risque, classe produit et sous-jacent.

Onglets :

- Vue d'ensemble : synthèse globale.
- Lignes de pricing : détail ligne à ligne.
- Risques agrégés : analyse par classe, sous-jacent, maturité, pilier.
- Stress tests : P&L par scénario.
- Qualité et limites : flags économiques et conventions.
- Documentation : rappels fonctionnels.

## Choix de conception

Le dashboard ne relance pas le pricing. Il lit les exports validés par le notebook afin de garantir une restitution stable.
