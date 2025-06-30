# PVTree-palm-vein-generationion
*PVTree: Realistin and Controllable Palm Vein Generation for Recognition Tasks.*
| [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32726)

## Prerequisites
- Python 3
- NVIDIA GPU + CUDA CuDNN

# Getting Started

## Installation
- Clone this repo:
```bash
git clone https://github.com/Sunniva-Shang/PVTree-palm-vein-generation
``` 
- Install dependencies:
`
pip install -r requirements.txt
`

## Palm Vein Images Generation
### Bezier creases Generation
- `python get_bezier.py`

### Palm Vascular Tree Generation
- `python main.py`
- The output are saved in `./pv_pattern_results`


### Images Generation
- Download [pcemodel-checkpoints](https://drive.google.com/file/d/1aN7Bi-dDXWC_EpwmMiK_iqXCw2hyyK8h/view?usp=share_link), unzip it and place it in `./pcemodel/checkpoints`.
- `bash ./pcemodel/run_gen.sh`
- The generated images are saved in `./pcemodel/results`.


## Train the PCE Model
- `bash ./pcemodel/train.sh`
- More details see paper [PCE-Palm](https://ojs.aaai.org/index.php/AAAI/article/view/28039).


### Citation
If you find this useful for your research, please use the following.

```
@inproceedings{jin2024pce,
  title={PCE-Palm: Palm Crease Energy Based Two-Stage Realistic Pseudo-Palmprint Generation},
  author={Jin, Jianlong and Shen, Lei and Zhang, Ruixin and Zhao, Chenglong and Jin, Ge and Zhang, Jingyun and Ding, Shouhong and Zhao, Yang and Jia, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={3},
  pages={2616--2624},
  year={2024}
}
```

If you have any questions or encounter any issues with the this code, please feel free to contact me (email: shengshang@mail.hfut.edu.cn). 
I would be more than happy to assist you in any way I can.

### Acknowledgements

This code borrows heavily from the [PCE-Palm](https://github.com/Ukuer/PCE-Palm).
