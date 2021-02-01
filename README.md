# POTION
### Code to replicate experimental results in the paper.
1. Use `conda` to create a virtual environment named `POTION`:
```
conda create -n POTION python=3.7
```

2. Activate the virtual environment:
```
conda activate POTION
```

3. Install necessary packages (PyTorch version is 1.4.0):
```
conda install numpy scipy pandas matplotlib seaborn pandas networkx=2.4 jupyter
conda install pytorch torchvision -c pytorch
pip install EoN
```

4. Create a folder at the root directory to save results:
```
mkdir result/
```


5. Check out the ipython notebook `src/exp.ipynb` for an example to running the code.


### Reference
```
@article{yu2020optimizing,
  title={Optimizing Graph Structure for Targeted Diffusion},
  author={Yu, Sixie and Torres, Leonardo and Alfeld, Scott and Eliassi-Rad, Tina and Vorobeychik, Yevgeniy},
  journal={arXiv preprint arXiv:2008.05589},
  year={2020}
}
```
