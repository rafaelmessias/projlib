### Installation

The `numpy` and `cython` modules must be manually installed before the rest of the dependencies.

```
pip install numpy cython
```

The `requirements.txt` file will handle most of the dependencies, but some of them are a bit trickier.

* FIt-SNE: requires FFTW (http://www.fftw.org/) to be installed. 
    * For OSX you can use `brew install fftw`; for Debian-based distros you can use `sudo apt install libfftw3-dev`.

* OptSNE: will be downloaded and installed from the git repo (https://github.com/rafaelmessias/Multicore-opt-SNE) automatically by pip, but first you must check the extra dependencies below.
    * CMake: For OSX you can use `brew install cmake`; for Debian-based distros you can use `sudo apt install cmake`.

* CUDA t-SNE: so far only worked when installed with Conda, as described in https://github.com/CannyLab/tsne-cuda/wiki/Installation. Setting up the actual CUDA libraries is beyond the scope of this document.

You can install everything else with:

```
pip install -r requirements.txt
```

Finally, make sure that `projlib` is in the path to be imported by python.

```
cd projlib
export PYTHONPATH=$PYTHONPATH:$(dirname $(pwd))
```
