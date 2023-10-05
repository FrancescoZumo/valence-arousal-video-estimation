# Valence-Arousal video estimation: model training and evaluation

Code written for my master's thesis: Procedural music generation for videogames conditioned through video emotion recognition

Complementary repositories, part of the same project:
-  valence-arousal-video-estimation (current repository)
-  [midi-emotion-primer-continuation](https://github.com/FrancescoZumo/midi-emotion-primer-continuation)
-  [videogame-procedural-music-experimental-setup](https://github.com/FrancescoZumo/videogame-procedural-music-experimental-setup)

More details can be found in the [thesis manuscript](https://www.politesi.polimi.it/handle/10589/210809
)

## Description

This folder contains the code used to train and evaluate a 3D-CNN that predicts valence and arousal continuous values from a sequence of input frames. 

The dataset used for training our proposed model is the discrete LIRIS-ACCEDE dataset: https://liris-accede.ec-lyon.fr/

For evaluating our method we also used the continuous LIRIS-ACCEDE dataset (see previous link) and the AGAIN dataset: https://again.institutedigitalgames.com/

## Usage

For training a model, modify all settings and parameters inside `main.py` according to your needs and run the script. The code can be expanded by adding more networks inside `models.py`
