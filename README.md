## TextMatch
Implement some deep model for Text Matching

[DIIN](./textmatch/model/diin.py)
<br>
[Natural Language Inference over Interaction Space](https://arxiv.org/abs/1709.04348)
<br>
ICLR2018


[ESIM](./textmatch/model/esim.py)
<br>
[Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038)
<br>
ACL2017

[MatchPyramid](./textmatch/model/matchPyramid.py)
<br>
[Text Matching as Image Recognition](https://arxiv.org/abs/1602.06359)
<br>
AAAI16

### About the code
Easy to train. You just need to edit the code in [preprocessor.py](./textmatch/preprocessor.py) to load your data
and just run the command `python -m textmatch.train --classfier=DIIN` to train a diin model.

Easy to extend. if you want to explore a new model. just to create a new file in `textmatch/model`
and define a class in this file, which the class extends the `TextModel` in the current dir. you just to override  method `get_model`.
last just register in the `textmatch/model/__init__.py`