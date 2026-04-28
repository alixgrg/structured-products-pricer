# Documentation fonctionnelle

## Objectif

L'application permet de valoriser et d'analyser un portefeuille multi-produits composé de produits de taux, d'options, de stratégies optionnelles, d'options à barrière, de notes structurées et d'autocalls.

## Fonctionnalités

- Pricing ligne à ligne.
- Calcul des Greeks et métriques de risque.
- Agrégation par portefeuille, devise, classe produit, sous-jacent, maturité et bucket de strike.
- Stress tests de marché.
- Lecture d'un inventaire Excel.
- Génération de quatre portefeuilles de démonstration.
- Visualisation des résultats dans Streamlit.

## Parcours utilisateur

1. Placer `inventory.xlsx`, `options.csv` et `rate_curves.parquet` dans `data/raw/`.
2. Exécuter le notebook final.
3. Vérifier les exports dans `reports/dashboard_exports/`.
4. Lancer `streamlit run src/dashboard/app.py`.
5. Utiliser les filtres : portefeuille, devise, classe produit, sous-jacent.
6. Lire les onglets : vue d'ensemble, lignes de pricing, risques, stress tests, qualité et documentation.

## Interprétation financière

- `price` : valeur modèle dans la devise de risque.
- `delta` : sensibilité au sous-jacent.
- `gamma` : convexité au sous-jacent.
- `vega` : sensibilité à la volatilité.
- `theta` : sensibilité au temps.
- `rho` : sensibilité au taux.
- `dv01` : variation de valeur pour +1 bp de taux.

Les devises EUR et USD ne sont pas converties. Les totaux doivent donc être lus par `risk_currency`.
