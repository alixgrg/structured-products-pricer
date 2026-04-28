# Documentation technique

## Architecture

```text
src/
  calibration/     Calibration volatilité et validation marché
  conventions/     Calendriers, day count, business days
  dashboard/       Application Streamlit
  factory/         Builders produits et router modèles
  market/          Chargement et contexte de marché
  models/          Modèles de pricing
  portfolio/       Inventaire, portefeuilles, moteur de pricing
  products/        Définition objet des produits financiers
  rates/           Courbes de taux et bootstrapping
  risk/            Agrégation, Greeks numériques, stress tests
```

## Flux de données

```text
data/raw -> normalisation -> calibration -> pricing -> agrégation risque -> exports -> dashboard
```

## Modules principaux

### `src/config.py`

Centralise les chemins du projet : raw, interim, processed, notebooks, reports, dashboard exports.

### `src/portfolio/inventory_loader.py`

Lit l'inventaire Excel, normalise les noms de colonnes, convertit les dates/nombres/pourcentages, puis construit une vue prête pour le pricing.

### `src/portfolio/demo_portfolios.py`

Crée quatre portefeuilles de démonstration à partir de l'inventaire source.

### `src/factory/builders.py`

Transforme une ligne d'inventaire en objet produit.

### `src/factory/pricing_router.py`

Route automatiquement chaque produit vers le modèle adapté.

### `src/portfolio/pricing_engine.py`

Orchestre le pricing ligne à ligne.

### `src/risk/aggregator.py`

Agrège les métriques de risque par portefeuille, devise, produit, sous-jacent, maturité et strike.

### `src/risk/stress_testing.py`

Reprice le portefeuille sous différents scénarios de marché.

## Extensibilité

Pour ajouter un produit :

1. Créer une classe dans `src/products/`.
2. Ajouter un builder dans `src/factory/builders.py`.
3. Ajouter le routage dans `src/factory/pricing_router.py`.
4. Ajouter ou réutiliser un modèle dans `src/models/`.
5. Ajouter un test unitaire.
