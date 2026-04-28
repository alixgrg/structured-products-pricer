# Modèles, formules et métriques de risque

## Produits de taux

Zéro-coupon :

```text
PV = N * DF(0,T)
```

Obligation à coupons :

```text
PV = somme_i CF_i * DF(0,t_i)
```

Swap de taux :

```text
PV_swap = PV_fixed_leg - PV_floating_leg
```

## Options vanilles

Black-Scholes-Merton avec dividende continu :

```text
Call = S exp(-qT) N(d1) - K exp(-rT) N(d2)
Put  = K exp(-rT) N(-d2) - S exp(-qT) N(-d1)

d1 = [ln(S/K) + (r - q + 0.5 sigma^2)T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

## Stratégies optionnelles

Les stratégies sont valorisées par réplication statique :

```text
PV_strategy = somme_i quantity_i * PV(option_i)
```

## Options à barrière

Les options à barrière utilisent des formules fermées de type Reiner-Rubinstein/Merton pour des barrières simples sans rebate. Les knock-in peuvent être obtenus par parité :

```text
Vanilla = Knock-Out + Knock-In
```

## Autocalls

Les autocalls sont valorisés par Monte Carlo sous dynamique GBM risque-neutre :

```text
dS/S = (r - q) dt + sigma dW
```

## Volatilité implicite

Une surface de volatilité est construite par couple :

```text
(underlying, valuation_date)
```

Cela évite de mélanger les smiles de plusieurs sous-jacents.

## Convention DV01 / rho

```text
dv01 = V(r + 1bp) - V(r)
rho  ~= -dv01 / 1bp pour les produits de taux
```
