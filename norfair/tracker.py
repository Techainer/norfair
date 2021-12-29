import math
from typing import Callable, List, Optional, Sequence
import numpy as np

from .filter import FilterSetup
from .utils import validate_points

H = np.array(
    [[1., 0., 0., 0.],
     [0., 1., 0., 0.]],
    dtype=np.float64
)

class Tracker:
    def __init__(
        self,
        distance_threshold: float,
        hit_inertia_min: int = 10,
        hit_inertia_max: int = 25,
        initialization_delay: Optional[int] = None,
        initial_hit_count: Optional[int] = None,
        detection_threshold: float = 0,
        point_transience: int = 4,
        filter_setup: "FilterSetup" = FilterSetup(),
    ):
        self.tracked_objects: Sequence["TrackedObject"] = []
        self.hit_inertia_min = hit_inertia_min
        self.hit_inertia_max = hit_inertia_max
        self.filter_setup = filter_setup
        self.ID = 0

        if initialization_delay is None:
            self.initialization_delay = int(
                (self.hit_inertia_max - self.hit_inertia_min) / 2
            )
        elif (
            initialization_delay < 0
            or initialization_delay > self.hit_inertia_max - self.hit_inertia_min
        ):
            raise ValueError(
                f"Argument 'initialization_delay' for 'Tracker' class should be an int between 0 and (hit_inertia_max - hit_inertia_min = {hit_inertia_max - hit_inertia_min}). The selected value is {initialization_delay}.\n"
            )
        else:
            self.initialization_delay = initialization_delay

        if initial_hit_count is None:
            self.initial_hit_count = self.hit_inertia_max//2
        else:
            self.initial_hit_count = initial_hit_count

        self.distance_threshold = distance_threshold
        self.detection_threshold = detection_threshold
        self.point_transience = point_transience

    def update(self, detections: Optional[List["Detection"]] = None, period: int = 1):
        self.period = period

        if detections:
            for idx, det in enumerate(detections):
                det._id = idx

        # Remove stale trackers and make candidate object real if it has hit inertia
        self.tracked_objects = [o for o in self.tracked_objects if o.has_inertia]

        # Update tracker
        for obj in self.tracked_objects:
            obj.tracker_step()

        # Update initialized tracked objects with detections
        initialized_objs = []
        initializing_objs = []
        for o in self.tracked_objects:
            if o.is_initializing:
                initializing_objs.append(o)
            else:
                initialized_objs.append(o)
        unmatched_detections, det_obj_pairs_1 = self.update_objects_in_place(
            initialized_objs, detections
        )

        # Update not yet initialized tracked objects with yet unmatched detections
        unmatched_detections, det_obj_pairs_2 = self.update_objects_in_place(
            initializing_objs, unmatched_detections
        )
        
        det_obj_pairs = det_obj_pairs_1 + det_obj_pairs_2
        
        det_obj_pairs.sort(key=lambda x: x[0])
        map_result = []
        
        current_idx = 0
        for expected_id in range(len(detections)):
            if current_idx >= len(det_obj_pairs) or det_obj_pairs[current_idx][0] != expected_id:
                map_result.append(None)
            else:
                map_result.append(det_obj_pairs[current_idx][1])
                current_idx += 1
        
        # Create new tracked objects from remaining unmatched detections
        for detection in unmatched_detections:
            self.tracked_objects.append(
                TrackedObject(
                    detection,
                    self.hit_inertia_min,
                    self.hit_inertia_max,
                    self.initialization_delay,
                    self.initial_hit_count,
                    self.detection_threshold,
                    self.period,
                    self.point_transience,
                    self.filter_setup
                )
            )

        # Finish initialization of new tracked objects
        for p in self.tracked_objects:
            if not p.is_initializing and p.id == None:
                p.id = self.ID
                self.ID += 1

        return map_result

    def update_objects_in_place(
        self,
        objects: Sequence["TrackedObject"],
        detections: Optional[List["Detection"]],
    ):
        if detections:
            if objects:
                # NOTE: I use a fixed distance function that is the Euclidean distance
                # between the representation point (i.e., box center) of the detections
                # and tracker's estimate of tracked objects
                dets = np.ascontiguousarray(np.vstack([det.points for det in detections]), dtype=np.float64)
                objs = np.ascontiguousarray(np.vstack([obj.estimate for obj in objects]), dtype=np.float64)
                distance_matrix = np.linalg.norm(dets[:, None, :] - objs[None, :, :], axis=-1)

                matched_det_indices, matched_obj_indices = self.match_dets_and_objs(
                    distance_matrix
                )

                det_obj_pairs = []

                if len(matched_det_indices) > 0:
                    unmatched_detections = [
                        d for i, d in enumerate(detections) if i not in matched_det_indices
                    ]

                    # Handle matched people/detections
                    for (match_det_idx, match_obj_idx) in zip(
                        matched_det_indices, matched_obj_indices
                    ):
                        match_distance = distance_matrix[match_det_idx, match_obj_idx]
                        matched_detection = detections[match_det_idx]
                        matched_object = objects[match_obj_idx]
                        if match_distance < self.distance_threshold:
                            matched_object.hit(matched_detection, period=self.period)
                            matched_object.last_distance = match_distance
                            det_obj_pairs.append((matched_detection._id, matched_object.id))
                        else:
                            unmatched_detections.append(matched_detection)
                else:
                    unmatched_detections = detections
            else:
                unmatched_detections = detections
                det_obj_pairs = []
        else:
            unmatched_detections = []
            det_obj_pairs = []

        return unmatched_detections, det_obj_pairs

    def match_dets_and_objs(self, distance_matrix: np.array):
        """Matches detections with tracked_objects from a distance matrix
        I used to match by minimizing the global distances, but found several
        cases in which this was not optimal. So now I just match by starting
        with the global minimum distance and matching the det-obj corresponding
        to that distance, then taking the second minimum, and so on until we
        reach the distance_threshold.
        This avoids the the algorithm getting cute with us and matching things
        that shouldn't be matching just for the sake of minimizing the global
        distance, which is what used to happen
        """
        # NOTE: This implementation is terribly inefficient, will revisit later
        distance_matrix = distance_matrix.copy()
        det_idxs = []
        obj_idxs = []
        current_min = distance_matrix.min()

        while current_min < self.distance_threshold:
            flattened_arg_min = distance_matrix.argmin()
            det_idx = flattened_arg_min // distance_matrix.shape[1]
            obj_idx = flattened_arg_min % distance_matrix.shape[1]
            det_idxs.append(det_idx)
            obj_idxs.append(obj_idx)
            distance_matrix[det_idx, :] = self.distance_threshold + 1
            distance_matrix[:, obj_idx] = self.distance_threshold + 1
            current_min = distance_matrix.min()

        return det_idxs, obj_idxs


class TrackedObject:
    def __init__(
        self,
        initial_detection: "Detection",
        hit_inertia_min: int,
        hit_inertia_max: int,
        initialization_delay: int,
        initial_hit_count: int,
        detection_threshold: float,
        period: int,
        point_transience: int,
        filter_setup: "FilterSetup",
    ):
        try:
            initial_detection_points = validate_points(initial_detection.points)
        except AttributeError:
            print(
                f"\n[red]ERROR[/red]: The detection list fed into `tracker.update()` should be composed of {Detection} objects not {type(initial_detection)}.\n"
            )
            exit()
        self.hit_inertia_min: int = hit_inertia_min
        self.hit_inertia_max: int = hit_inertia_max
        self.initialization_delay = initialization_delay
        self.initial_hit_count = initial_hit_count
        self.point_hit_inertia_min: int = math.floor(hit_inertia_min / point_transience)
        self.point_hit_inertia_max: int = math.ceil(hit_inertia_max / point_transience)
        if (self.point_hit_inertia_max - self.point_hit_inertia_min) < period:
            self.point_hit_inertia_max = self.point_hit_inertia_min + period
        self.detection_threshold: float = detection_threshold
        self.initial_period: int = period
        self.hit_counter: int = hit_inertia_min + period
        self.point_hit_counter = self.point_hit_inertia_min
        self.last_distance: Optional[float] = None
        self.current_min_distance: Optional[float] = None
        self.last_detection: "Detection" = initial_detection
        self.age: int = 0
        self.is_initializing_flag: bool = True
        self.id: Optional[int] = None
        self.detected_at_least_once_points = False

        # Create Kalman Filter
        self.filter = filter_setup.create_filter(initial_detection_points)
        self.dim_z = 2

    def tracker_step(self):
        self.hit_counter -= 1
        self.point_hit_counter -= 1
        self.age += 1
        # Advances the tracker's state
        self.filter.predict()

    @property
    def is_initializing(self):
        if (
            self.is_initializing_flag
            and self.hit_counter > self.hit_inertia_min + self.initialization_delay
        ):
            self.is_initializing_flag = False
            ## NOTE: Think about using this or not later
            self.hit_counter = self.initial_hit_count
            
        return self.is_initializing_flag

    @property
    def has_inertia(self):
        return self.hit_counter >= self.hit_inertia_min

    @property
    def estimate(self):
        positions = self.filter.x.T.flatten()[: self.dim_z].reshape(-1, 2)
        # velocities = self.filter.x.T.flatten()[self.dim_z :].reshape(-1, 2)
        return positions

    @property
    def live_points(self):
        return self.point_hit_counter > self.point_hit_inertia_min

    def hit(self, detection: "Detection", period: int = 1):
        points = detection.points

        self.last_detection = detection
        if self.hit_counter < self.hit_inertia_max:
            self.hit_counter += 2 * period
        
        # We use a kalman filter in which we consider each coordinate on each point as a sensor.
        # This is a hacky way to update only certain sensors (only x, y coordinates for
        # points which were detected).
        # TODO: Use keypoint confidence information to change R on each sensor instead?
        self.point_hit_counter += 2 * period
        self.point_hit_counter = max(0, self.point_hit_counter)
        self.point_hit_counter = min(self.point_hit_counter, self.point_hit_inertia_max)
        self.filter.update(points.T, H)

        # Force points being detected for the first time to have velocity = 0
        # This is needed because some detectors (like OpenPose) set points with
        # low confidence to coordinates (0, 0). And when they then get their first
        # real detection this creates a huge velocity vector in our KalmanFilter
        # and causes the tracker to start with wildly inaccurate estimations which
        # eventually coverge to the real detections.
        if not self.detected_at_least_once_points:
            self.filter.x[self.dim_z :][:] = 0
            self.detected_at_least_once_points = True
            

    def __repr__(self):
        if self.last_distance is None:
            placeholder_text = "\033[1mObject_{}\033[0m(age: {}, hit_counter: {}, last_distance: {}, init_id: {})"
        else:
            placeholder_text = "\033[1mObject_{}\033[0m(age: {}, hit_counter: {}, last_distance: {:.2f}, init_id: {})"
        return placeholder_text.format(
            self.id,
            self.age,
            self.hit_counter,
            self.last_distance,
            self.initializing_id,
        )


class Detection:
    def __init__(self, points: np.array, data=None):
        self.points = points
        self.data = data