# Dashboard Streamlit — Structured Products Pricer

Ce dossier contient l'application Streamlit de restitution du projet **Structured Products Pricer**. Le dashboard est volontairement conçu en mode lecture d'exports CSV : il ne relance pas la calibration, le pricing Monte Carlo ou les traitements lourds. Cela permet d'avoir une application stable et rapide pour la soutenance.

## 1. Emplacement dans le projet

Le dashboard lit par défaut les exports depuis :

```text
structured-products-pricer/reports/dashboard_exports/
```

Ce chemin peut être modifié dans la barre latérale de l'application.

## 2. Pré-requis

Depuis la racine du projet, installer les dépendances si nécessaire :

```bash
pip install -r src/dashboard/requirements_dashboard.txt
```

Le fichier `requirements_dashboard.txt` contient les dépendances minimales pour lancer la partie restitution. Le projet principal peut bien sûr avoir d'autres dépendances pour le pricing, la calibration ou les notebooks.

## 3. Génération des exports CSV

Avant de lancer le dashboard, exécuter le notebook final de présentation. Il doit générer les fichiers suivants dans `reports/dashboard_exports/` :

```text
priced_lines.csv
risk_safe_totals.csv
risk_by_product_class.csv
risk_by_underlying.csv
risk_by_maturity.csv
risk_by_pillar.csv
stress_summary.csv
stress_pnl_by_position.csv
top_vega.csv
economic_flags.csv
quality_summary.csv
```

Le dashboard peut s'ouvrir même si certains fichiers secondaires sont absents, mais `priced_lines.csv` est indispensable. Si `priced_lines.csv` est manquant ou vide, l'application affiche un message d'erreur et demande de relancer le notebook.

## 4. Lancement de l'application

Depuis la racine du projet :

```bash
streamlit run src/dashboard/app.py
```

Streamlit ouvre ensuite une URL locale, généralement :

```text
http://localhost:8501
```

## 5. Utilisation de la barre latérale

La barre latérale sert à piloter toute l'application.

### Dossier des exports CSV

Champ texte permettant de modifier le dossier où se trouvent les exports. Par défaut :

```text
reports/dashboard_exports
```

Si tous les fichiers attendus sont disponibles, un message vert indique que les exports sont chargés.

### Filtres globaux

Les filtres s'appliquent aux vues principales :

- **Portefeuille** : sélection d'un ou plusieurs portefeuilles de démonstration.
- **Devise de risque** : filtre EUR, USD, etc.
- **Classe produit** : filtre par type de produit construit par le pricer, par exemple `InterestRateSwap`, `AutocallProduct`, `Butterfly`, `BarrierOption`.
- **Sous-jacent** : filtre par sous-jacent action ou par courbe de taux.

Si aucun élément n'est sélectionné dans un filtre, le dashboard conserve toutes les valeurs disponibles pour cette dimension.

## 6. Onglet “Vue d'ensemble”

Cet onglet donne une synthèse rapide du portefeuille filtré.

### Indicateurs affichés

- nombre de lignes pricées ;
- nombre de portefeuilles ;
- nombre de classes produit ;
- statuts de pricing présents dans les données.

### Graphiques

- **Prix par devise de risque** : somme des prix par devise. Les devises ne sont pas converties entre elles.
- **Prix par classe produit** : contribution de chaque classe produit au prix total filtré.

### Tableaux détaillés

Les tableaux de synthèse sont placés dans un expander afin d'éviter un effet de superposition visuelle avec les graphiques. Cliquer sur :

```text
Voir les tableaux détaillés de la vue d'ensemble
```

pour afficher les tables sous-jacentes.

### Interprétation financière

Cette vue permet d'identifier rapidement :

- la répartition EUR/USD ;
- les classes produit qui dominent la valeur ;
- les produits dont le prix peut être élevé à cause du nominal ou de la quantité.

Un prix élevé ne signifie pas nécessairement que le modèle est faux. La première vérification consiste à regarder les colonnes de taille : `quantity`, `notional`, `booking_notional`, `contract_multiplier`.

## 7. Onglet “Pricing lignes”

Cet onglet présente les résultats ligne par ligne.

Colonnes principales :

- `portfolio` : portefeuille de rattachement ;
- `product_id` : identifiant produit ;
- `product_type` : type produit issu de l'inventaire ;
- `product_class` : classe Python utilisée pour le pricing ;
- `underlying` : sous-jacent ou courbe de risque ;
- `risk_currency` : devise de risque ;
- `price` : prix signé ;
- `delta`, `gamma`, `vega`, `theta`, `rho`, `dv01` : métriques de risque ;
- `maturity_bucket`, `strike_bucket` : piliers de maturité et de strike ;
- `status`, `error_message` : statut technique du pricing.

Une barre de recherche permet de filtrer par produit, classe produit, portefeuille ou sous-jacent.

### Colonnes de taille économique

Un expander affiche les colonnes de taille :

- `quantity` ;
- `notional` ;
- `booking_notional` ;
- `contract_multiplier` ;
- `price_unit` ;
- `position_sign`.

Ces colonnes sont importantes pour expliquer les ordres de grandeur. Par exemple, une stratégie optionnelle peut avoir un prix élevé simplement parce que la position est notionnelle ou multipliée par une quantité importante.

## 8. Onglet “Risques agrégés”

Cet onglet permet d'explorer les risques agrégés.

Tables disponibles :

- `risk_safe_totals` : totaux par portefeuille et devise de risque ;
- `risk_by_product_class` : agrégation par classe produit ;
- `risk_by_underlying` : agrégation par sous-jacent ;
- `risk_by_maturity` : agrégation par pilier de maturité ;
- `risk_by_pillar` : agrégation plus détaillée par devise, produit, sous-jacent, maturité et strike.

Métriques disponibles selon les colonnes chargées :

- `price` ;
- `delta` ;
- `gamma` ;
- `vega` ;
- `theta` ;
- `rho` ;
- `dv01`.

### Lecture financière

- Le `delta` mesure la sensibilité au sous-jacent.
- Le `gamma` mesure la convexité au sous-jacent.
- Le `vega` mesure la sensibilité à la volatilité.
- Le `theta` mesure la sensibilité au passage du temps.
- Le `rho` mesure la sensibilité au taux.
- Le `dv01` mesure l'impact d'une hausse de 1 point de base des taux.

Les agrégations ne mélangent pas EUR et USD. C'est volontaire : sans couche FX, un total multi-devise serait économiquement ambigu.

## 9. Onglet “Stress tests”

Cet onglet présente les scénarios de stress générés par le notebook.

### Résumé par scénario

La table `stress_summary.csv` donne la valeur du portefeuille par scénario et le P&L par rapport au scénario de base.

### P&L par position

La table `stress_pnl_by_position.csv` permet d'identifier les positions qui expliquent le plus le P&L d'un scénario.

### Lecture financière

- Un choc spot impacte surtout les options, autocalls et produits structurés actions.
- Un choc volatilité impacte les produits optionnels et les produits exotiques.
- Un choc taux impacte les swaps, obligations et notes contenant une composante zéro-coupon.

## 10. Onglet “Qualité & limites”

Cet onglet affiche les contrôles de qualité produits par le notebook.

### Résumé qualité

`quality_summary.csv` résume les contrôles techniques : nombre de lignes, statuts, erreurs, cohérence d'agrégation.

### Flags économiques

`economic_flags.csv` liste les points à expliquer dans la soutenance. Les flags ne sont pas forcément des bugs. Ils peuvent indiquer :

- un prix élevé dû à une convention `quantity` / `notional` ;
- un autocall exprimé en fraction de nominal ;
- un vega négatif économiquement possible sur un butterfly ou une barrière ;
- une lecture multi-devise nécessitant de ne pas agréger EUR et USD directement.

## 11. Onglet “Documentation”

Cet onglet contient une documentation fonctionnelle intégrée :

- objectif de l'application ;
- inputs utilisés ;
- raison du mode CSV ;
- conventions importantes ;
- limites assumées ;
- statut de chargement de chaque fichier.

## 12. Correction de l'effet “tableaux superposés”

Si deux tableaux ou un tableau et un graphique semblent visuellement superposés dans la vue d'ensemble, la cause vient généralement de l'empilement de `st.dataframe` et `st.bar_chart` dans la même colonne, surtout avec un zoom navigateur élevé ou une fenêtre réduite.

La correction appliquée dans `app.py` est la suivante :

1. afficher les graphiques dans deux colonnes ;
2. placer les tableaux détaillés dans un expander séparé ;
3. fixer une hauteur raisonnable pour les graphiques ;
4. éviter d'empiler un tableau interactif et un graphique dans la même colonne principale.

En pratique, dans l'onglet “Vue d'ensemble”, les graphiques apparaissent directement et les tableaux sont accessibles via :

```text
Voir les tableaux détaillés de la vue d'ensemble
```

## 13. Problèmes fréquents

### Le dashboard affiche “priced_lines.csv manquant ou vide”

Relancer le notebook final et vérifier que `priced_lines.csv` existe dans `reports/dashboard_exports/`.

### Les filtres sont vides

Vérifier que `priced_lines.csv` contient les colonnes `portfolio`, `risk_currency`, `product_class` et `underlying`.

### Certains fichiers sont absents mais l'application s'ouvre

C'est normal pour les fichiers secondaires. Le dashboard affiche simplement des messages d'information dans les onglets concernés.

### Les prix semblent trop élevés

Vérifier les colonnes de taille économique dans l'onglet “Pricing lignes”. Les écarts viennent souvent des conventions de quantité, nominal ou multiplicateur.

### Les totaux EUR et USD ne sont pas additionnés

C'est volontaire. Sans conversion FX, l'agrégation multi-devise n'a pas de sens financier robuste.

## 14. Commandes utiles

Relancer le dashboard :

```bash
streamlit run src/dashboard/app.py
```

Forcer Streamlit à utiliser un autre port :

```bash
streamlit run src/dashboard/app.py --server.port 8502
```

Nettoyer le cache Streamlit depuis l'interface :

```text
Menu ⋮ > Clear cache
```

Puis recharger la page.

## 15. Message de soutenance conseillé

Le dashboard est une couche de restitution. Les calculs lourds sont réalisés dans le notebook final, puis exportés en CSV. Ce choix permet d'avoir une application stable, rapide et reproductible. Les résultats sont analysés par devise de risque, classe produit, sous-jacent, maturité et scénario de stress. Les limites principales portent sur les conventions de booking, notamment la distinction entre quantité, nominal et multiplicateur de contrat.
