# Données et conventions

## Fichiers d'entrée

```text
data/raw/inventory.xlsx
data/raw/options.csv
data/raw/rate_curves.parquet
```

## Inventaire

Le loader convertit les colonnes françaises vers un schéma anglais canonique :

- `date_valorisation` -> `valuation_date`
- `devise` -> `currency`
- `quantite` -> `quantity`
- `sous_jacent` -> `underlying`
- `maturite` -> `maturity_date`

## Données options

Le fichier `options.csv` sert à récupérer les spots, extraire ou reconstruire des volatilités implicites, construire des surfaces et valider le repricing. Le séparateur peut être `;`.

## Courbes de taux

`rate_curves.parquet` contient les points de courbe utilisés pour construire la courbe d'actualisation.

## Conventions de taille

Les prix dépendent fortement de :

- `quantity` : quantité issue de l'inventaire ;
- `notional` : nominal économique du produit ;
- `booking_notional` : taille harmonisée avant pricing ;
- `contract_multiplier` : multiplicateur de contrat.

Dans la version de rendu, lorsque `notional` est disponible, il est la taille économique de référence.

## Devises

Le projet distingue :

- `currency` : devise du produit ;
- `risk_currency` : devise de reporting risque.

Aucun total EUR+USD n'est calculé sans conversion FX.
