## Multimodal Transformer Networks on SIMMC 2.0 dataset
<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This is the PyTorch implementation of the paper:
**[Multimodal Transformer Networks for End-to-End Video-Grounded Dialogue Systems](<https://arxiv.org/abs/1907.01166>)**. [**Hung Le**](https://github.com/henryhungle), [Doyen Sahoo](http://www.doyensahoo.com/), [Nancy F. Chen](https://sites.google.com/site/nancyfchen/home), [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/). ***[ACL 2019](<https://www.aclweb.org/anthology/P19-1564/>)***. 

**This branch contains the code to run on the Situated and [Interactive Multimodal Conversations (SIMMC 2.0) dataset](https://github.com/facebookresearch/simmc2).
You can refer to main branch which is for the AVSD benchmark [here](https://github.com/henryhungle/MTN).** 

To understand the architecture changes, please read the Multimodal Transformer Network (MTN) adaptation to Visual Dialog (Section 4.4 from MTN paper).

This code has been written using PyTorch 1.0.1.
You can also use the `mtn_environment.yml` conda environment file.
If you use the source code in this repo in your work, please cite the following papers.

**SIMMC 2.0: A Task-oriented Dialog Dataset for Immersive Multimodal Conversations**
<pre>
@article{DBLP:journals/corr/abs-2104-08667,
  author    = {Satwik Kottur and
               Seungwhan Moon and
               Alborz Geramifard and
               Babak Damavandi},
  title     = {{SIMMC} 2.0: {A} Task-oriented Dialog Dataset for Immersive Multimodal
               Conversations},
  journal   = {CoRR},
  volume    = {abs/2104.08667},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.08667},
  archivePrefix = {arXiv},
  eprint    = {2104.08667},
  timestamp = {Mon, 26 Apr 2021 17:25:10 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2104-08667.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
</pre>

**Multimodal Transformer Networks**
<pre>
@inproceedings{le-etal-2019-multimodal,
    title = "Multimodal Transformer Networks for End-to-End Video-Grounded Dialogue Systems",
    author = "Le, Hung  and
      Sahoo, Doyen  and
      Chen, Nancy  and
      Hoi, Steven",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = July,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1564",
    doi = "10.18653/v1/P19-1564",
    pages = "5612--5623",
}
</pre>

## Dataset

Download [dataset](https://github.com/facebookresearch/simmc2) of the SIMMC 2.0 benchmark, including the training, validation, and test dialogs.

All the data should be saved into folder `data` in the repository root folder.

## Scripts 

During train time, the model is trained in a generative setting using the ground-truth answer. During test time, at each dialog turn, the model uses beamsearch to generate either assistant response or belief states (Refer to the paper Section 4.4 for more details).

We created `run.sh` to prepare evaluation code, train models, generate_responses, and evaluating the generated responses with automatic metrics. You can run:

```console
run.sh [execution_stage] [image_features] [image_features] [numb_epochs] [warmup_steps] [dropout_rate]
```

For example, to train the model:
```
run.sh 0 resnet resnet 10 1000 0
```

The parameters are: 

| Parameter           | Description                                                  | Values                                                       |
| :------------------ | :----------------------------------------------------------- | ------------------------------------------------------------ |
| execution_state     | Stage of execution e.g. preparing, training, generating, evaluating |  <br /><=2: training the models<br /><=3: select output responses using log likelihood |
| image_features      | Image features extracted from pretrained models              |  <br /> e.g. rcnn|
| image\_feature\_names | Names of features for saving output                    | any value corresponding to the image_features input |
| num_epochs          | Number of training epochs                                    | e.g. 20                                                      |
| warmup_steps        | Number of warmup steps                                       | e.g. 9660                                                    |
| dropout_rate        | Dropout rate during training                                 | e.g. 0.2                                                     |

While training, the model with the best validation is saved. The model is evaluated by using loss per token. The model output, parameters, vocabulary, and training and validation logs will be save into folder `exps`.
Other parameters, including data-related options, model parameters,  training and generating settings, are defined in the `run.sh` file.

**NOTE**: Use the `--predict_belief_states` flag to train the model to learn predicting the belief state instead of assistant generation.

## Performance on SIMMC 2.0 Dataset

This model has been used to benchmark the assistant response generation (Task 4) 
and multimodal dialog state tracking (Task 4 - Generation) of the SIMMC 2.0 dataset 
[here](https://github.com/facebookresearch/simmc2).

**Multimodal Dialog State Tracking (Task 3)**

| Model           | Joint Accuracy  | Dialog Act F1 | Slot F1 | Request Slot F1 |
| :-------------: | :-------------: | :-----------: | :-----: | :-------------: |
| MTN-SIMMC2      | 0.283           | 0.934         | 0.748   | 0.854           |

**Assistant Response Generation (Task 4 - Generation)**

| Model           |  BLEU  | 
| :-------------: | :----: |
| MTN-SIMMC2      | 0.2174 |