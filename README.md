# Code for CLDL

Code for "Continuous Label Distribution Learning" in Pattern Recognition 2023.

If you use the code in this repo for your work, please cite the following bib entries:

```
@article{DBLP:journals/pr/ZhaoAXG23,
  author    = {Xingyu Zhao and
               Yuexuan An and
               Ning Xu and
               Xin Geng},
  title     = {Continuous label distribution learning},
  journal   = {Pattern Recognit.},
  volume    = {133},
  pages     = {109056},
  year      = {2023},
  url       = {https://doi.org/10.1016/j.patcog.2022.109056},
  doi       = {10.1016/j.patcog.2022.109056},
}
```

## Requirements

- Python == 3.6
- PyTorch == 1.5
- GPyTorch == 1.1
- NumPy >= 1.13.3
- Scikit-learn >= 0.20
- Matlab == R2018b

## Running the scripts

To preprocess the label vectors, please run "get_encoding.m" in Matlab.

To train and test the CLDL model in the terminal, use:

```bash
$ python run_CLDL.py --dataset sample_data --outDim 10 --max_iter 10 --lr 0.05 --kernel_type linear --device cuda:0 --neighbor_num 10
```

## Acknowledgment

Our project references the codes and datasets in the following repo and papers.

[Label Distribution Learning](http://palm.seu.edu.cn/xgeng/LDL/index.htm)

Prateek Jain, Raghu Meka, Inderjit S. Dhillon. Guaranteed Rank Minimization via Singular Value Projection. NIPS 2010: 937-945.

Kush Bhatia, Himanshu Jain, Purushottam Kar, Manik Varma, Prateek Jain. Sparse Local Embeddings for Extreme Multi-label Classification. NIPS 2015: 730-738.
