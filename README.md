This repository provides the code for finite-length scaling laws for SC-LDPC codes and their Monte-Carlo simulations.
It provides implementations of density evolution, mean evolution, peeling decoding, full BP decoding with and without a limit on the number of iterations, and sliding window decoding with a limit on the number of iterations.

The step-by-step guide to using the scaling laws can be found in the Jupyter notebook `sc_ldpc_scaling_bec_4_8.ipynb`.

The code in this repository is used in several papers.
Please consider citing them if you are using the code.

For the scaling laws for full BP and sliding window decoding with unlimited iterations, please cite
```
R. Sokolovskii, A. Graell i Amat, and F. Brännström,
"Finite-length scaling of spatially coupled LDPC codes under window decoding over the BEC"
IEEE Trans. Commun., vol. 68, no. 10, pp. 5988–5998, Oct. 2020.
```

For the scaling laws for full BP and sliding window decoding with limited iterations, please cite
```
Roman Sokolovskii, Alexandre Graell i Amat, Fredrik Brännström,
"Finite-Length Scaling of SC-LDPC Codes With a Limited Number of Decoding Iterations"
https://arxiv.org/abs/2203.08880
```

For the scaling laws for doped ensembles, please cite
```
R. Sokolovskii, A. Graell i Amat, and F. Brännström,
"On doped SC- LDPC codes for streaming"
IEEE Commun. Lett., vol. 25, no. 7, pp. 2123–2127, Jul. 2021.
```


Some files are larger than 100 MB. GitHub requires using `git lfs` in this case.
To install `git lfs`, run
```bash
$ brew install git-lfs
```

To fetch the file (after cloning the repository), run
```bash
$ git lfs fetch --all
```

See [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/collaboration-with-git-large-file-storage) for details on using `git lfs`.
