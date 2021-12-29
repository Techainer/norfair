![Norfair by Tryolabs logo](docs/logo.png)

![Build status](https://github.com/tryolabs/norfair/workflows/CI/badge.svg?branch=master) [![DOI](https://zenodo.org/badge/276473370.svg)](https://zenodo.org/badge/latestdoi/276473370)

Norfair is a customizable lightweight Python library for real-time 2D object tracking.

Using Norfair, you can add tracking capabilities to any detector with just a few lines of code.

---

This fork was maintained by [Techainer](https://techainer.com/). It's attempt to assign a track id to each object instead of return a list of new object after tracking.

This is optimized for the usecase when there is 1 representative point per detection (i.e. the center of detection box), and use a fixed distance function that is the Euclidean distance between tracker's estimate and that point. Making this up to 10 times faster than the original Norfair implementation.

In doing so, we also dropped the use of `past_detections_length`, `distance_function` and the concept of "detection scores" since we are not using them anyway. Thus make the dependencies only include numpy and numba.

## Features

- Any detector expressing its detections as a series of `(x, y)` coordinates can be used with Norfair. This includes detectors performing object detection, pose estimation, and instance segmentation.

- The function used to calculate the distance between tracked objects and detections is defined by the user, making the tracker extremely customizable. This function can make use of any extra information, such as appearance embeddings, which can heavily improve tracking performance.

- Modular. It can easily be inserted into complex video processing pipelines to add tracking to existing projects. At the same time it is possible to build a video inference loop from scratch using just Norfair and a detector.

- Fast. The only thing bounding inference speed will be the detection network feeding detections to Norfair.

The original Norfair is built, used and maintained by [Tryolabs](https://tryolabs.com).

## Installation

Norfair currently supports Python 3.7+. To install it, simply run:
```bash
pip install norfair
```
## How it works

Norfair works by estimating the future position of each point based on its past positions. It then tries to match these estimated positions with newly detected points provided by the detector. For this matching to occur, Norfair can rely on any distance function specified by the user of the library. Therefore, each object tracker can be made as simple or as complex as needed.

The following is an example of a particularly simple distance function calculating the Euclidean distance between tracked objects and detections. This is possibly the simplest distance function you could use in Norfair, as it uses just one single point per detection/object.

```python
 def euclidean_distance(detection, tracked_object):
     return np.linalg.norm(detection.points - tracked_object.estimate)
```

As an example we use [Detectron2](https://github.com/facebookresearch/detectron2) to get the single point detections to use with this distance function. We just use the centroids of the bounding boxes it produces around cars as our detections, and get the following results.

## Documentation

You can find the documentation for Norfair's API [here](docs/README.md).

## Citing Norfair

For citations in academic publications, please export your desired citation format (BibTeX or other) from [Zenodo](https://doi.org/10.5281/zenodo.5146253).

## License

Copyright © 2021, [Tryolabs](https://tryolabs.com). Released under the [BSD 3-Clause](LICENSE).
