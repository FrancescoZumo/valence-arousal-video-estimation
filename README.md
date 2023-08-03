# valence-arousal video estimation

Code written for my master's thesis: Procedural music generation forvideogames conditioned through video emotion recognition

This folder contains the code used to train and evaluate a 3D-CNN that predicts valence and arousal continuous values from a sequence of input frames

The dataset used for training our proposed model is the discrete LIRIS-ACCEDE dataset: https://liris-accede.ec-lyon.fr/

For evaluating our method we also used the continuous LIRIS-ACCEDE dataset and the AGAIN dataset: https://again.institutedigitalgames.com/


for running the code, adapt all settings and parameters inside `main.py` for your needs and run the code. Several models were implemented and the code can be expanded by adding more networks inside `models.py`
