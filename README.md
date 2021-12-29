![Norfair by Tryolabs logo](docs/logo.png)

![Build status](https://github.com/tryolabs/norfair/workflows/CI/badge.svg?branch=master) [![DOI](https://zenodo.org/badge/276473370.svg)](https://zenodo.org/badge/latestdoi/276473370)

Norfair is a customizable lightweight Python library for real-time 2D object tracking.

Using Norfair, you can add tracking capabilities to any detector with just a few lines of code.

The original Norfair is built, used and maintained by [Tryolabs](https://tryolabs.com).

---

This fork was maintained by [Techainer](https://techainer.com/). It assigns a track id to each object instead of returning a list of new objects after tracking.

This is optimized for the use case when there is 1 representative point per detection (i.e. the center of detection box) and uses a fixed distance function that is the Euclidean distance between the tracker's estimate and that point. Making this up to 10 times faster than the original Norfair implementation.

In doing so, we also dropped the use of `past_detections_length`, `distance_function`, and the concept of "detection scores" since we are not using them anyway. Thus make the dependencies only include `numpy` and `numba`.


## Installation

Norfair currently supports Python 3.6+. To install it, simply run:
```bash
pip install techainer_norfair
```
## Documentation

You can find the documentation for Norfair's API [here](docs/README.md).

## Citing Norfair

For citations in academic publications, please export your desired citation format (BibTeX or other) from [Zenodo](https://doi.org/10.5281/zenodo.5146253).

## License

Copyright © 2021, [Tryolabs](https://tryolabs.com) and [Techainer](https://techainer.com). Released under the [BSD 3-Clause](LICENSE).
