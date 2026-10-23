# Polymarket BTC Up/Down bot (v2)

Bot de paper trading sur les marchés horaires « Bitcoin Up or Down » de Polymarket. Le token YES paie 1 $ si le BTC clôture l'heure au-dessus de son prix d'ouverture : c'est une option digitale. Le bot compare son prix à une probabilité calculée par un modèle et prend position quand l'écart est suffisant et robuste.

## Lancer

```bash
pip install -r requirements.txt
python main.py            # paper trading
python -m pytest tests    # 21 tests unitaires
```

Tous les paramètres sont dans `config.yaml`. Une clé manquante fait échouer le démarrage : il n'y a aucune valeur par défaut cachée dans le code.

## Structure

```
bot/
  config.py              chargement typé de config.yaml
  data/
    binance.py           prix spot (1 s), klines 1 min clôturées (60 s), open horaire = strike
    polymarket.py        découverte du marché horaire (API Gamma), carnets d'ordres (CLOB)
    types.py             carnet d'ordres : meilleur prix, profondeur, fill VWAP
  model/
    volatility.py        EWMA sur returns 1 min, sauts exclus et mesurés à part
    pricing.py           probabilité en formule fermée (Merton), scénarios de vol
  strategy/
    signal.py            edge = probabilité prudente - ask
    sizing.py            Kelly fractionné pour binaires
  execution/
    broker.py            exécution papier : latence, re-vérification au fill, VWAP
    portfolio.py         positions valorisées au bid, sorties, journal des trades
  engine.py              boucles de données et tick de stratégie
tests/test_bot.py
```

## 1. Données

Binance : prix spot en REST chaque seconde ; klines 1 minute interrogées chaque minute, seules les bougies clôturées et non encore vues sont consommées. 2 880 klines (2 jours, récupérées par pages de 1 000) servent à la calibration au démarrage. Le strike est l'open de la bougie 1 h Binance, qui est la référence de règlement du marché.

Polymarket : le slug du marché horaire est reconstruit en heure de New York avec gestion été/hiver (`zoneinfo`), puis les carnets YES et NO sont lus toutes les 2 secondes sur le CLOB.

Aucun trade si le prix Binance date de plus de 10 secondes. Chaque tick est journalisé dans `signals.csv` (spot, strike, vol, probabilités, carnets, signal) : ce fichier constitue le jeu de données pour un backtest honnête.

## 2. Signal

Volatilité : EWMA sur les log-returns 1 minute, σ²(t) = λ × σ²(t−1) + (1 − λ) × r²(t). La variance est la variable mise à jour à chaque minute ; λ fixe la mémoire du modèle (demi-vie = ln 0,5 / ln λ).

λ n'est pas fixé à 0,94 : cette valeur RiskMetrics a été calibrée sur des returns journaliers. Il est estimé par maximum de vraisemblance sur la fenêtre de calibration, c'est-à-dire la valeur dont les prévisions de variance à une minute expliquent le mieux les returns observés. Validation sur données simulées : l'estimateur retrouve le vrai λ avec un écart-type de 0,004 sur 2 880 returns (0,016 sur 500, d'où la fenêtre de 2 jours). La valeur estimée, sa demi-vie et le gain de vraisemblance par rapport à 0,94 sont journalisés au démarrage.

Sauts : un return supérieur à 3 sigmas locaux est classé comme saut, exclu de la variance diffusive et de la vraisemblance, et utilisé pour estimer la fréquence et la taille des sauts. La détection dépendant de λ, la calibration se fait en trois passes : détection avec λ initial, estimation de λ sur les returns restants, reconstruction de la variance avec le λ estimé.

Prix théorique : P(S_T ≥ K) en jump-diffusion de Merton, calculée en formule fermée (somme de lois normales pondérée par une loi de Poisson). Résultat exact et sans bruit, validé contre un Monte Carlo à 400 000 trajectoires dans les tests.

Robustesse : la probabilité est recalculée avec une volatilité à −20 % et +20 %. Pour chaque côté, on garde la valeur la moins favorable : la probabilité prudente.

Entrée : edge = probabilité prudente − ask. Signal si l'edge est compris entre 3 et 30 points, ask entre 5 et 95 cents, plus de 5 minutes avant l'expiration, 15 secondes depuis le dernier signal. YES et NO sont évalués, seul le meilleur est retenu.

## 3. Sizing

Kelly fractionné pour binaires : `f = (q × b − (1 − q)) / b`, avec q la probabilité prudente et b le gain net par dollar engagé, `b = (1 − ask) × (1 − frais) / ask`.

On mise un quart de Kelly, plafonné à 2,5 % du capital courant : les mises baissent donc automatiquement en drawdown. Ce montant est une cible par issue (marché × côté) : on ne complète que l'écart avec l'exposition déjà ouverte. La taille est ensuite limitée par le budget d'exposition totale restant (10 %) et par 50 % de la profondeur disponible à moins de 2 cents du meilleur ask. En dessous de 1 $, l'ordre n'est pas envoyé.

| Ask | Probabilité prudente | Kelly plein | Mise (100 $) |
|---|---|---|---|
| 0,45 | 0,48 | 4,6 % | 1,15 $ |
| 0,42 | 0,46 | 6,1 % | 1,52 $ |
| 0,58 | 0,66 | 18,1 % | 2,50 $ (plafond) |

## 4. Prise et gestion de position

L'ordre est exécuté 2 secondes après la décision, sur le carnet à cet instant : l'edge est revérifié sur le nouvel ask (ordre annulé s'il est passé sous 3 points), la taille est recalculée, et le fill parcourt les niveaux du carnet (prix moyen pondéré). On n'ouvre jamais le côté opposé d'un marché déjà en position.

Les positions sont valorisées au bid. Sorties, par ordre de priorité :

1. moins de 5 minutes avant l'expiration : clôture forcée ;
2. perte de 80 % ou plus : stop ;
3. bid ≥ prix théorique courant − 1 cent : convergence. Couvre à la fois la prise de profit (le marché a rejoint le modèle) et la disparition de l'edge (le modèle a bougé contre nous).

Des frais de 2 % sont prélevés sur les gains. Chaque position clôturée est écrite dans `trades.csv`.

## Changements par rapport à la v1

| v1 | v2 | Pourquoi |
|---|---|---|
| λ = 0,94 fixé (RiskMetrics, données journalières) | λ estimé par maximum de vraisemblance sur 2 jours de returns 1 min | Mémoire du modèle adaptée à l'horizon minute |
| Monte Carlo 10 000 trajectoires | Formule fermée | Supprime ~0,5 point de bruit qui déclenchait des sorties |
| Sauts calibrés dans un espace normalisé | Sauts en unités de log-return | Paramètres interprétables et cohérents avec le modèle |
| Z-score du mispricing sur 100 ticks | Edge sur probabilité prudente | Le z-score ignorait les edges persistants et reposait sur des données autocorrélées |
| Mispricing sur le mid moins demi-spread | Edge mesuré sur l'ask | Prix réellement payé |
| Sizing linéaire 2 % à 2,5 % | Kelly fractionné, cible par issue | Taille proportionnelle à l'edge, pas d'empilement |
| Capital initial | Capital courant | Réduction automatique en drawdown |
| Plafond sur la liquidité totale du marché | Plafond sur la profondeur au meilleur prix | Slippage maîtrisé |
| Sortie sur prix théorique d'entrée | Sortie sur prix théorique courant | La cible d'entrée devient fausse quand le BTC bouge |
| Fuseau New York codé en UTC−5 | `zoneinfo` America/New_York | Slug faux une partie de l'année |
| Paramètres lus dans les mauvaises sections | Config typée, sans défaut caché | Les réglages du fichier sont réellement utilisés |

## Limites connues

Exécution papier uniquement. Le modèle suppose une dérive nulle sur l'heure. La calibration du modèle (une probabilité annoncée de 60 % se réalise-t-elle 60 % du temps ?) reste à mesurer sur les données de `signals.csv` avant tout passage en réel.
