# Botifarra - Reinforcement Learning

Aquest projecte conté un entorn bàsic que permet jugar a la botifarra per agents de RL.
EL projecte no conté cap eina per entrenar els agents només les regles de la botifarra amb la codificació d'observacions, accions i recompenses proposats per l'autor.

## Instal·lació
Per instal·lar el projecte cal clonar el repositori i instal·lar les dependències:

```bash
git clone 
cd botifarra_rl
pip install -e .
```

## Ús
Existeixen dos exemples per veure com funcionen les regles de la botifarra i com es poden utilitzar en agents de RL.

- `tests/test_aitomatic.py`: Exemple de com s'han implemntat les regles de la Botifarra. Conté agents que juguen de forma aleatòria però seguint les regles.
- `tests/test_env.py`: Exemple bàsic de com utilitzar l'entor de RL, basat en Gymnasium Farama. Es pot veure com s'ha codificat l'espai d'observacions, l'acció i les recompenses.

## About
Aquest projecte ha estat desenvolupat per Narcís Palomeras i està llicenciat sota la llicència MIT.
