### Setup

* Créer un venv, se placer dedans et installer les requirements avec :
```
python3.6 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Scraping

* Scraper les liens vers chaque discours et les stocker dans un JSON
```
scrapy crawl links -o links.json
```

* Scraper les discours en entier via leurs liens
```
scrapy crawl speech -o speeches.json
```

### Modèle

* Lancer le training (les arguments sont optionnels)
```
python network.py [-m <modele_depart.hdf5>] [--gpu]
```

* Générer un texte à partir d'un modèle (obligatoire)
```
python prediction.py <modele.hdf5> [-t <temp_sampling>]
```

* Changer l'architecture du modèle en modifiant network.py
* Changer les paramètres d'apprentissage et de prédiction en modifiant params.py
