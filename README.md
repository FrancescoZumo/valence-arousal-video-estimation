# valence-arousal video estimation

Code written for my master's thesis: Procedural music generation forvideogames conditioned through video emotion recognition

This folder contains the code used to train and evaluate a 3D-CNN that predicts valence and arousal continuous values from a sequence of input frames

The dataset used for training our proposed model is the discrete LIRIS-ACCEDE dataset: https://liris-accede.ec-lyon.fr/

For evaluating our method we also used the continuous LIRIS-ACCEDE dataset and the AGAIN dataset: https://again.institutedigitalgames.com/


For training a model, modify all settings and parameters inside `main.py` according to your needs and run the script. The code can be expanded by adding more networks inside `models.py`
