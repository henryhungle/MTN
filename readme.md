## Multimodal Transformer Networks for End-to-End Video-Grounded Dialogue Systems
<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This is the PyTorch implementation of the paper:
**[Multimodal Transformer Networks for End-to-End Video-Grounded Dialogue Systems](<https://arxiv.org/abs/1907.01166>)**. [**Hung Le**](https://github.com/henryhungle), [Doyen Sahoo](http://www.doyensahoo.com/), [Nancy F. Chen](https://sites.google.com/site/nancyfchen/home), [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/). ***[ACL 2019](<https://www.aclweb.org/anthology/P19-1564/>)***. 


This code has been written using PyTorch 1.0.1. If you use the source code in this repo in your work, please cite the following paper. The bibtex is:
<pre>
@inproceedings{le-etal-2019-multimodal,
    title = "Multimodal Transformer Networks for End-to-End Video-Grounded Dialogue Systems",
    author = "Le, Hung  and
      Sahoo, Doyen  and
      Chen, Nancy  and
      Hoi, Steven",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1564",
    doi = "10.18653/v1/P19-1564",
    pages = "5612--5623",
    abstract = "Developing Video-Grounded Dialogue Systems (VGDS), where a dialogue is conducted based on visual and audio aspects of a given video, is significantly more challenging than traditional image or text-grounded dialogue systems because (1) feature space of videos span across multiple picture frames, making it difficult to obtain semantic information; and (2) a dialogue agent must perceive and process information from different modalities (audio, video, caption, etc.) to obtain a comprehensive understanding. Most existing work is based on RNNs and sequence-to-sequence architectures, which are not very effective for capturing complex long-term dependencies (like in videos). To overcome this, we propose Multimodal Transformer Networks (MTN) to encode videos and incorporate information from different modalities. We also propose query-aware attention through an auto-encoder to extract query-aware features from non-text modalities. We develop a training procedure to simulate token-level decoding to improve the quality of generated responses during inference. We get state of the art performance on Dialogue System Technology Challenge 7 (DSTC7). Our model also generalizes to another multimodal visual-grounded dialogue task, and obtains promising performance.",
}
</pre>


## Abstract
Developing Video-Grounded Dialogue Systems (VGDS), where a dialogue is conducted based on visual and audio aspects of a given video, is significantly more challenging than traditional image or text-grounded dialogue systems because (1) feature space of videos span across multiple picture frames, making it difficult to obtain semantic information; and (2) a dialogue agent must perceive and process information from different modalities (audio, video, caption, etc.) to obtain a comprehensive understanding.  Most existing work is based on RNNs and sequence-to-sequence architectures, which are not very effective for capturing complex long-term dependencies (like in videos).  To overcome this, we propose Multimodal Transformer Networks (MTN) to encode videos and incorporate information from different modalities. We also propose query-aware attention through an auto-encoder to extract query-aware features from non-text modalities. We develop a training procedure to simulate token-level decoding to improve the quality of generated responses during inference. We get state of the art performance on Dialogue System Technology Challenge 7 (DSTC7). Our model also generalizes to another multimodal visual-grounded dialogue task, and obtains promising performance.

<p align="center">
<img src="img/example_dstc7.png" width="50%" />
 <br>
 A sample dialogue from the DSTC7 Video Scene-aware Dialogue training set with 4 example video scenes.
</p>

## Model Architecture

<p align="center">
<img src="img/model_v4.1.png" width="100%" />
 Our MTN architecture includes 3 major components: (i) encoder layers encode text sequences and video features; (ii) decoder layers (D) project target sequence and attend on multiple inputs; and (iii) Query-Aware Auto-Encoder layers (QAE) attend on non-text modalities from query features. For simplicity, Feed Forward, Residual Connection and Layer Normalization layers are not presented. 
<img src="img/encoders.png" width="100%" />
2 types of encoders are used: text-sequence encoders (left) and video encoders (right). Text-sequence encoders are used on text input, i.e. dialogue history, video caption, query, and output sequence. Video encoders are used on visual and audio features of input video.
</p>

## Dataset

Download [dataset](<https://drive.google.com/drive/folders/1SlZTySJAk_2tiMG5F8ivxCfOl_OWwd_Q>) of the DSTC7, Video-Scene Aware Dialogues Track, including the training, validation, and test dialogues and the features of Charades videos extracted using VGGish and I3D models. 

All the data should be saved into folder `data` in the repo root folder.

## Scripts 

We created `run.sh` to prepare evaluation code, train models, generate_responses, and evaluating the generated responses with automatic metrics. You can run:

```console
❱❱❱ run.sh [execution_stage] [video_features] [video_feature_names] [numb_epochs] [warmup_steps] [dropout_rate] 
```
The parameters are: 

| Parameter           | Description                                                  | Values                                                       |
| :------------------ | :----------------------------------------------------------- | ------------------------------------------------------------ |
| execution_state     | Stage of execution e.g. preparing, training, generating, evaluating | <=1: preparing evaluation code by downloading the COCO caption evaluation tool <br /><=2: training the models<br /><=3: generating responses using beam search (default)<br /><=4: evaluating the generated responses |
| video_features      | Video features extracted from pretrained models              | vggish: audio features extracted using VGGish<br />i3d_flow: visual features extracted from I3D model <br />i3d_rgb: visual features extracted from I3D model <br />Features can also be combined, separated by single space e.g. "vggish i3d_flow" |
| video_feature_names | Names of video features for saving output                    | any value corresponding to the video_features input e.g. vggish+i3d_flow |
| num_epochs          | Number of training epochs                                    | e.g. 20                                                      |
| warmup_steps        | Number of warmup steps                                       | e.g. 9660                                                    |
| dropout_rate        | Dropout rate during training                                 | e.g. 0.2                                                     |

While training, the model with the best validation is saved. The model is evaluated by using loss per token. The model output, parameters, vocabulary, and training and validation logs will be save into folder `exps`.  

Other parameters, including data-related options, model parameters,  training and generating settings, are defined in the `run.sh` file.

## Sample Dialogues
<p align="center">
<img src="img/samples.png" width="100%" />
 Example test dialogue responses extracted from the ground-truth and generated by MTN and the baseline. For simplicity, the dialogue history is not presented and only parts of the video caption C are shown. Our model provides answers that are more accurate than the baseline, capturing single human action or a series of actions in the videos.
</p>

## Visdial 
MTN can also be adapted to run on the VisDial benchmark. Please switch the repo branch to 'visdial' [here](https://github.com/henryhungle/MTN/tree/visdial) for the code. Main changes include data loading in data handler. 

During train time, the model is trained in a generative setting using the ground-truth answer. During test time, at each dialogue turn, the model selects the best answer candidate based on the log likelihood among the answer options (Refer to the paper Section 4.4 for more details). 

Note that in the current code, we use a small version of dialogue files in the `data` folder and dummy features for images during data loading/batching. Please replace the code with the paths to your downloaded data files and available features. 
