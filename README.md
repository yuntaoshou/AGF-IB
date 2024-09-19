[stars-img]: https://img.shields.io/github/stars/yuntaoshou/AGF-IB?color=yellow
[stars-url]: https://github.com/yuntaoshou/AGF-IB/stargazers
[fork-img]: https://img.shields.io/github/forks/yuntaoshou/AGF-IB?color=lightblue&label=fork
[fork-url]: https://github.com/yuntaoshou/AGF-IB/network/members
[AKGR-url]: https://github.com/yuntaoshou/AGF-IB

# Adversarial alignment and graph fusion via information bottleneck for multimodal emotion recognition in conversations
By Shou, Yuntao and Meng, Tao and Ai, Wei and Zhang, Fuchen and Yin, Nan and Li, Keqin. [[paper link]](https://www.sciencedirect.com/science/article/pii/S1566253524003683)

[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]

This is an official implementation of 'Adversarial alignment and graph fusion via information bottleneck for multimodal emotion recognition in conversations' :fire:. Any problems, please contact shouyuntao@stu.xjtu.edu.cn. If you find this repository useful to your research or work, it is really appreciated to star this repository :heart:.

<div  align="center"> 
  <img src="https://github.com/yuntaoshou/DSAGCN/blob/main/fig/DSAGCN.png" width=100% />
</div>



## 🚀 Installation

```bash
Python 3.8.5
torch 1.7.1
CUDA 11.3
torch-geometric 1.7.2
```

## Training
```bash
python train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type='hyper' --epochs=0 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP'
```

If our work is helpful to you, please cite:
```bash
@article{shou2024adversarial,
  title={Adversarial alignment and graph fusion via information bottleneck for multimodal emotion recognition in conversations},
  author={Shou, Yuntao and Meng, Tao and Ai, Wei and Zhang, Fuchen and Yin, Nan and Li, Keqin},
  journal={Information Fusion},
  volume={112},
  pages={102590},
  year={2024},
  publisher={Elsevier}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yuntaoshou/AGF-IB&type=Date)](https://star-history.com/#yuntaoshou/AGF-IB&Date)
