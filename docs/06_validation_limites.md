# Validation, contrôles et limites

## Contrôles effectués

- Statut des lignes de pricing.
- Présence d'erreurs dans `error_message`.
- Cohérence entre lignes de pricing et agrégations.
- Contrôle des totaux par devise.
- Contrôle des stress tests.
- Identification des prix extrêmes.
- Identification des autocalls potentiellement exprimés en fraction du nominal.
- Analyse des top expositions vega.

## Problèmes identifiés et interprétation

### Prix optionnels élevés

Un prix optionnel élevé peut être dû à la taille de position. Ce n'est pas forcément un bug : il faut vérifier `quantity`, `notional` et `contract_multiplier`.

### Autocalls proches de 1

Un autocall autour de `0.86` indique souvent un prix exprimé en fraction du nominal. S'il doit être lu sur nominal 100, cela correspond à environ `86`.

### Vega négatif

Un vega négatif peut être normal pour des butterflies, options barrières ou positions optionnelles short.

### Multi-devise

Le portefeuille contient EUR et USD. Les totaux multi-devises ne sont pas additionnés sans FX.

## Limites de modélisation

- Produits mono-sous-jacent seulement.
- Pas de quanto.
- Autocalls simplifiés.
- Pas de corrélation multi-actifs.
- Surface de volatilité dépendante de la qualité du panel d'options.
- Stress tests simples, non historiques.
