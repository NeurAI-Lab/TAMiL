# TAMiL
Official repository for "[Task-Aware Information Routing from Common Representation Space in Lifelong Learning](https://openreview.net/forum?id=-M0TNnyWFT5)", ICLR 2023


## How to run?
+ python main.py  --seed 10  --dataset seq-tinyimg  --model tam --buffer_size 200   --load_best_args \
  --tensorboard --pretext_task mse --notes 'experiment 1'
        
## Setup

+ Use `./main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.
+ Use `--evaluate` to load and evaluate the model 
## Datasets

**Class-Il / Task-IL settings**

+ Sequential CIFAR-10
+ Sequential CIFAR-100
+ Sequential Tiny-ImageNet
+ Sequential Core50

## Cite Our Work

If you find the code useful in your research, please consider citing our paper:


    @inproceedings{
      bhat2023taskaware,
      title={Task-Aware Information Routing from Common Representation Space in Lifelong Learning},
      author={Prashant Shivaram Bhat and Bahram Zonooz and Elahe Arani},
      booktitle={The Eleventh International Conference on Learning Representations },
      year={2023},
      url={https://openreview.net/forum?id=-M0TNnyWFT5}
    }
