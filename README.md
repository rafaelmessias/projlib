## Installation

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

## Usage

The main features that are currently available are listed below. Click on any of them to see the respective README and learn more about how to use them. Keep in mind that this library is very much a work in progress and will change/grow a lot in the near future!

<<<<<<< HEAD
* [Quality measures](quality/)
* [Input/Output formats](io/)
=======
* [Quality measures](quality/README.md)
* [Input/Output formats](io/README.md)
>>>>>>> 16e881572459a7159ce60b0e82ecb20f7bce4b08
