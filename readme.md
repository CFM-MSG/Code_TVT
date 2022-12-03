# TVT: Three-Way Vision Transformer through Multi-Modal Hypersphere Learning for Zero-Shot Sketch-Based Image Retrieval

This repository is the Pytorch implementation of the proposed work "TVT: Three-way Vision Transformer through Multi-modal Hypersphere Learning for Zero-shot Sketch-based Image Retrieval"



## Installation and Requirements

- ```
  cudatoolkit=11.1.1
  ```

- ```
  numpy=1.20.3
  ```

- ```
  python=3.9.6
  ```

- ```
  pytorch=1.8.0
  ```

- ```
  timm=0.4.12
  ```

- ```
  torchvision=0.9.0
  ```



## Training

### Data Download

Sketchy and TU-Berlin: [by Dutta et al.](https://github.com/AnjanDutta/sem-pcyc) or [by Liu et al.](https://github.com/qliu24/SAKE)

QuickDraw: [by Dey et al.](https://github.com/sounakdey/doodle2search)

Note that these datasets are publicly available and have been widely used in the community.



### Running

#### Codes

**sketch_dino.py** and **eval_image_retrieval.py** are the scripts used to train and evaluate sketch ViT, respectively.

**main_myvit.py** and **eval_myvit.py** are the scripts used to train and evaluate TVT, respectively.

**vision_transformer.py** is the script that contains the implementation of models.



#### Training Sketch ViT

We use the [DINO-S/8](https://github.com/facebookresearch/dino) as the starting checkpoint of sketch ViT. The training instructions are listed as follows:

- TU-Berlin 
  ``` bash
  CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 sketch_dino.py --batch_size_per_gpu 512 --epochs 50 --lr 0.0001 --local_crops_number 10 --gradient_accumulation_steps 64  --dataset tuberlin --resume_pretrain 1 --split random --disable_dropout 0
  ```

- Sketchy-NO 

  ```bash
  CUDA_VISIBILE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 sketch_dino.py --batch_size_per_gpu 512 --epochs 50 --lr 0.0001 --local_crops_number 10 --gradient_accumulation_steps 64  --dataset sketchy --resume_pretrain 1 --split zeroshot --disable_dropout 0
  ```


- Sketchy

  ```bash
  CUDA_VISIBILE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 sketch_dino.py --batch_size_per_gpu 512 --epochs 50 --lr 0.0001 --local_crops_number 10 --gradient_accumulation_steps 64  --dataset sketchy --resume_pretrain 1 --split random --disable_dropout 0
  ```

- QuickDraw

  ```bash
  CUDA_VISIBILE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 sketch_dino.py --batch_size_per_gpu 512 --epochs 20 --lr 0.0001 --local_crops_number 10 --gradient_accumulation_steps 64 --dataset quickdraw --resume_pretrain 1 --split zeroshot --disable_dropout 0
  ```



#### Evaluation of Sketch ViT

We evaluate the pre-trained sketch ViT using image-to-image and sketch-to-sketch retrieval tasks. We observe that the pre-training model suffers from severe performance degradation after a few epochs, requiring more data and longer time to recover its performance. For the sake of efficiency, we use the checkpoint before the performance degradation as the sketch ViT model.

- TU-Berlin

  ```bash
  CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port=65000 --use_env --nproc_per_node=1 eval_image_retrieval.py --imsize 224 --multiscale 0 --data_path ../dataset --dataset tuberlin --arch vit_small --patch_size 8 --preck 100  --use_train 0 --intra_modal 1 --check_sketch_dino True --use_cuda True --split random
  ```

- Sketchy-NO

  ```bash
  CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=65000 --use_env --nproc_per_node=1 eval_image_retrieval.py --imsize 224 --multiscale 0 --data_path ../dataset --dataset sketchy --arch vit_small --patch_size 8 --preck 200  --use_train 0 --intra_modal 1 --check_sketch_dino True --use_cuda True --split zeroshot
  ```

- Sketchy

  ```bash
  CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=65000 --use_env --nproc_per_node=1 eval_image_retrieval.py --imsize 224 --multiscale 0 --data_path ../dataset --dataset sketchy --arch vit_small --patch_size 8 --preck 100  --use_train 0 --intra_modal 1 --check_sketch_dino True --use_cuda True --split random
  ```

- QuickDraw

  ```bash
  CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=65000 --use_env --nproc_per_node=1 eval_image_retrieval.py --imsize 224 --multiscale 0 --data_path ../dataset --dataset quickdraw --arch vit_small --patch_size 8 --preck 100  --use_train 0 --intra_modal 1 --check_sketch_dino True --use_cuda True --split zeroshot
  ```



#### Training TVT

The image ViT is used as the starting checkpoint of the fusion ViT. The training instructions are listed as follows:

- TU-Berlin

  ```bash
  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 main_myvit.py --batch_size_per_gpu 128 --epochs 50 --lr 0.0005 5 --global_crops_scale 0.14 1.0 --global_crops_number 1 --local_crops_number 0 --gradient_accumulation_steps 8  --dataset tuberlin --resume_pretrain 1 --split random --disable_dropout 1 --skt_factor 4 --token_num 2 --use_align_uniform 1 
  ```

- Sketchy-NO

  ```bash
  CUDA_VISIBILE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_myvit.py --batch_size_per_gpu 256 --epochs 50 --lr 0.0005 --global_crops_scale 0.14 1.0 --global_crops_number 1 --local_crops_number 0 --gradient_accumulation_steps 16  --dataset sketchy --resume_pretrain 1 --split zeroshot --disable_dropout 1 --skt_factor 2 --token_num 2 --use_align_uniform 1 
  ```

- Sketchy

  ```bash
  CUDA_VISIBILE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_myvit.py --batch_size_per_gpu 256 --epochs 50 --lr 0.0005 --global_crops_scale 0.14 1.0 --global_crops_number 1 --local_crops_number 0 --gradient_accumulation_steps 16  --dataset sketchy --resume_pretrain 1 --split random --disable_dropout 1 --skt_factor 2 --token_num 2 --use_align_uniform 1 
  ```

- QuickDraw

  ```bash
  CUDA_VISIBILE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_myvit.py --batch_size_per_gpu 256 --epochs 50 --lr 0.0005 --global_crops_scale 0.14 1.0 --global_crops_number 1 --local_crops_number 0 --gradient_accumulation_steps 16  --dataset quickdraw --resume_pretrain 1 --split zeroshot --disable_dropout 1 --skt_factor 2 --token_num 2 --use_align_uniform 1 
  ```



#### Evaluation of TVT

- TU-Berlin

  ```bash
  CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port=65000 --use_env --nproc_per_node=1 eval_myvit.py --imsize 224 --multiscale 0 --data_path ../dataset --dataset tuberlin --arch vit_small_fd --patch_size 8 --preck 100  --use_train 0 --intra_modal 0 --use_cuda True --split random2 --token_num 2 --num_classes 220 --return_idx 0 --pretrained_weight ../output_fd/checkpoint-0050.pth
  ```

- Sketchy-NO

  ```bash
  CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=65000 --use_env --nproc_per_node=1 eval_myvit.py --imsize 224 --multiscale 0 --data_path ../dataset --dataset sketchy --arch vit_small_fd --patch_size 8 --preck 200 --mapk 200  --use_train 0 --intra_modal 0 --use_cuda True --split zeroshot --token_num 2 --num_classes 104 --return_idx 0 --pretrained_weight ../output_fd/checkpoint-0050.pth
  ```

- Sketchy

  ```bash
  CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=50000 --use_env --nproc_per_node=1 eval_myvit.py --imsize 224 --multiscale 0 --data_path ../dataset --dataset sketchy --arch vit_small_fd --patch_size 8 --preck 100  --use_train 0 --intra_modal 0 --use_cuda True --split random1 --token_num 2 --num_classes 100 --return_idx 0 --pretrained_weight ../output_fd/checkpoint-0050.pth
  ```

- QuickDraw

  ```bash
  CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=65000 --use_env --nproc_per_node=1 eval_myvit.py --imsize 224 --multiscale 0 --data_path ../dataset --dataset quickdraw --arch vit_small_fd --patch_size 8 --preck 200 --mapk 200  --use_train 0 --intra_modal 0 --use_cuda True --split zeroshot --token_num 2 --num_classes 110 --return_idx 0 --pretrained_weight ../output_fd/checkpoint-0050.pth
  ```


### Main Idea

In this paper, we study the zero-shot sketch-based image retrieval (ZS-SBIR) task, which retrieves natural images related to sketch queries from unseen categories. In the literature, convolutional neural networks (CNNs) have become the de-facto standard and they are either trained end-to-end or used to extract pretrained features for images and sketches. However, CNNs are limited in modeling the global structural information of objects due to the intrinsic locality of convolution operations. To this end, we propose a Transformer-based approach called Three-Way Vision Transformer (TVT) to leverage the ability of Vision Transformer (ViT) to model global contexts due to the global self-attention mechanism. Going beyond simply applying ViT to this task, we propose a token-based strategy of adding fusion and distillation tokens and making them complementary to each other. Specifically, we integrate three ViTs, which are pre-trained on data of each modality, into a three-way pipeline through the processes of distillation and multimodal hypersphere learning. The distillation process is proposed to supervise fusion ViT (ViT with an extra fusion token) with soft targets from modality-specific ViTs, which prevent fusion ViT from catastrophic forgetting. Furthermore, our method learns a multi-modal hypersphere by performing inter- and intra-modal alignment without loss of uniformity, which aims to bridge the modal gap between modalities of sketch and image and avoid the collapse in dimensions. Extensive experiments on three benchmark datasets, i.e., Sketchy, TU-Berlin, and QuickDraw, demonstrate the superiority of our TVT method over the state-of-the-art ZS-SBIR methods.

![image-20221203145156932](C:\Users\jiatelin\AppData\Roaming\Typora\typora-user-images\image-20221203145156932.png)

### Overall Results

![image-20221203145331343](C:\Users\jiatelin\AppData\Roaming\Typora\typora-user-images\image-20221203145331343.png)

![image-20221203145345379](C:\Users\jiatelin\AppData\Roaming\Typora\typora-user-images\image-20221203145345379.png)

### Visualization

![image-20221203145445381](C:\Users\jiatelin\AppData\Roaming\Typora\typora-user-images\image-20221203145445381.png)

![image-20221203145458810](C:\Users\jiatelin\AppData\Roaming\Typora\typora-user-images\image-20221203145458810.png)

### Cite

```
@article{Tian_Xu_Shen_Yang_Shen_2022, title={TVT: Three-Way Vision Transformer through Multi-Modal Hypersphere Learning for Zero-Shot Sketch-Based Image Retrieval}, volume={36}, url={https://ojs.aaai.org/index.php/AAAI/article/view/20136}, DOI={10.1609/aaai.v36i2.20136}, number={2}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Tian, Jialin and Xu, Xing and Shen, Fumin and Yang, Yang and Shen, Heng Tao}, year={2022}, month={Jun.}, pages={2370-2378} }
```

