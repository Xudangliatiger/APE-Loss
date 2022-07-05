# Revisiting AP loss for Dense Object Detection: Adaptive Ranking Pair Selection

The official implementation of APE Loss. Our implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [aLRP loss](https://arxiv.org/abs/2009.13592)

> [**Revisiting AP loss for Dense Object Detection: Adaptive Ranking Pair Selection**](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Revisiting_AP_Loss_for_Dense_Object_Detection_Adaptive_Ranking_Pair_CVPR_2022_paper.html).            
> Dongli Xu, Jinghong Deng , Wen Li.
> *CVPR 2022*


## Summary

In this paper, we revisit the AP loss from a pairwise ranking perspective for dense object detection.
In the process, we reveal an essential fact that proper ranking pair selection plays an important role in producing accurate detection results compared with the distance function and balance constant.
Therefore, we propose a novel strategy, Adaptive Ranking Pair Selection (ARPS), by providing more complete and accurate ranking pairs.
We first exploit the localization information into extra rank pairs with the Adaptive Pairwise Error, which can be also considered as a more accurate form of AP loss.
We then use normalized ranking scores and localization scores to split the positive and negative samples.
The proposed method is very simple and achieves performance comparable to existing classification and ranking methods.


## Specification of Dependencies and Preparation

- Please see requirements.txt and requirements folder for the rest of the dependencies.
- Please refer to [install.md](docs/install.md) for installation instructions of MMDetection.
- Please see [getting_started.md](docs/getting_started.md) for dataset preparation and the basic usage of MMDetection.


## Training Code
The configuration files of all models listed above can be found in the `configs/ape_loss` folder. You can follow [getting_started.md](docs/getting_started.md) for training code. As an example, to train APE Loss (PAA* 800) on 4 GPUs as we did, use the following command:

```
./tools/dist_train.sh
```


## Test Code
The configuration files of all models listed above can be found in the `configs/ape_loss` folder.

```
./tools/dist_test.sh
```

## License
Following MMDetection, this project is released under the [Apache 2.0 license](LICENSE).


## How to Cite

Please cite the paper if you benefit from our paper or repository:
```
@inproceedings{xu_apeloss_2022,
       title     = {Revisiting AP loss for Dense Object Detection: Adaptive Ranking Pair Selection},
       author    = {Xu, Dongli and Deng, Jinghong and Li, Wen},
       booktitle = {Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)},
       pages     = {14187-14196},
       year      = {2022}
}
```