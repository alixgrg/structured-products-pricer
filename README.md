# Structured Products Pricer

## Objectif

Ce dépôt fournit une application Python modulaire pour le pricing de produits structurés mono-underlying, la calibration de surfaces de volatilité et de courbes de taux, la gestion d'inventaire et l'agrégation de risque.

## Résumé de l'architecture

- **src/** : code applicatif principal (modules de calibration, marché, produits, portefeuille, risque, utilitaires).
- **data/** : jeux de données organisés par étapes (raw, interim, processed).
- **notebooks/** : notebooks de démonstration et validation end-to-end.
- **scripts/** : utilitaires pour construire/transformer les jeux de données.
- **tests/** : suite pytest couvrant pricing, calibration, loaders et intégration.
- **legacy/** : anciennes étapes et scripts conservés pour traçabilité.

L'application suit une architecture modulaire claire :

- Market layer: ingestion et normalisation des données de marché.
- Calibration layer: bootstrap des courbes, calibration d'IV (SVI/SSVI).
- Models & Pricing: pricers pour produits vanilles et structurés.
- Factory / Router: sélection du pricer adapté par ligne produit.
- Portfolio & Risk: agrégation, calculs de grecs et stress tests.
- Dashboard: exports et tables prêtes pour visualisation.

## Structure du projet (sélection clé)

- [environment.yml](environment.yml) — environnement conda recommandé.
- [src/config.py](src/config.py) — configuration applicative centrale.
- [src/convention.py](src/convention.py) — conventions de date/tenor.
- [src/io_utils.py](src/io_utils.py) — helpers lecture/écriture.
- [src/calibration/](src/calibration) — calibration, SVI et vérifications.
- [src/market/](src/market) — loaders et normalisation des données de marché.
- [src/products/](src/products) — implémentations des produits supportés.
- [src/portfolio/](src/portfolio) — ingestion d'inventaire et moteur de pricing de portefeuille.
- [src/risk/](src/risk) — agrégateurs et routines de stress/test.
- [notebooks/](notebooks) — notebooks de validation et démonstration.
- [data/](data) — raw, interim et processed (pipeline de données).
- [tests/](tests) — tests automatisés pytest.

## Data flow

external sources -> data/raw -> data/interim -> data/processed

Les loaders privilégient les copies locales sous `data/raw/` puis retombent sur les chemins externes pour assurer reproductibilité des notebooks.

## Commandes rapides

- Créer l'environnement conda :

```bash
conda env create -f environment.yml
conda activate structured-products-pricer
```

- (optionnel) ou venv :

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

- Lancer la suite de tests :

```bash
pytest -q
```

- Exécuter le notebook de validation complet : ouvrir [notebooks/00_application_complete_demo_validation_v4.ipynb](notebooks/00_application_complete_demo_validation_v4.ipynb).

## Bonnes pratiques

- Les noms de modules/fonctions utilisent `snake_case`, les classes `PascalCase`.
- Les dates sont stockées en pandas Timestamp timezone-naive, normalisées au jour.
- Les colonnes tabulaires normalisées utilisent `snake_case`.

## Fichiers utiles à consulter

- Loader des courbes : [src/market/loaders.py](src/market) (voir `load_rate_curves`).
- Calibration IV / SVI : [src/calibration/](src/calibration).
- Moteur de pricing portefeuille : [src/portfolio/](src/portfolio).

## Étapes suivantes recommandées

- Vérifier les notebooks dans `notebooks/` pour une exécution end-to-end.
- Lancer `pytest` et corriger les éventuels échecs locaux.

---

