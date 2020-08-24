# Installation des packages
```
pip3 -r install requirements.txt
```
Si cela ne fonctionne pas, recommencer avec la distribution Anaconda d'installer : https://www.anaconda.com/products/individual
# Tester la détection sur une image
```
python face_detector --image ./Examples/<image>
```

# Tester la détection sur une vidéo/webcam
Si l'argument vidéo n'est pas fourni, la détection se lance automatiquement sur la caméra
```
python face_detector_video [--video./Examples/<video>] 
```

# Démarrer le serveur flask
## Sur Windows
Dans le répertoire flask_serv
```
set FLASK_APP=main.py
flask run
```
## Sur Linux
```
export FLASK_APP=main.py
flask run
```

L'exécution du jupyter notebook nécessite l'utilisation d'une carte graphique dédiée. Le modèle déjà entraîné est fourni avec le fichier face_mask.model.