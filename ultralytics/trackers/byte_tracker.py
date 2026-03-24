# # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# from __future__ import annotations

# from typing import Any
# from .utils.gmc import GMC
# import numpy as np
# from ..utils import LOGGER
# from ..utils.ops import xywh2ltwh
# from .basetrack import BaseTrack, TrackState
# from .utils import matching
# from .utils.kalman_filter import KalmanFilterXYAH


# class STrack(BaseTrack):
#     """Single object tracking representation that uses Kalman filtering for state estimation.

#     This class is responsible for storing all the information regarding individual tracklets and performs state updates
#     and predictions based on Kalman filter.

#     Attributes:
#         shared_kalman (KalmanFilterXYAH): Shared Kalman filter used across all STrack instances for prediction.
#         _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
#         kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
#         mean (np.ndarray): Mean state estimate vector.
#         covariance (np.ndarray): Covariance of state estimate.
#         is_activated (bool): Boolean flag indicating if the track has been activated.
#         score (float): Confidence score of the track.
#         tracklet_len (int): Length of the tracklet.
#         cls (Any): Class label for the object.
#         idx (int): Index or identifier for the object.
#         frame_id (int): Current frame ID.
#         start_frame (int): Frame where the object was first detected.
#         angle (float | None): Optional angle information for oriented bounding boxes.

#     Methods:
#         predict: Predict the next state of the object using Kalman filter.
#         multi_predict: Predict the next states for multiple tracks.
#         multi_gmc: Update multiple track states using a homography matrix.
#         activate: Activate a new tracklet.
#         re_activate: Reactivate a previously lost tracklet.
#         update: Update the state of a matched track.
#         convert_coords: Convert bounding box to x-y-aspect-height format.
#         tlwh_to_xyah: Convert tlwh bounding box to xyah format.

#     Examples:
#         Initialize and activate a new track
#         >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
#         >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
#     """

#     shared_kalman = KalmanFilterXYAH()

#     def __init__(self, xywh: list[float], score: float, cls: Any):
#         """Initialize a new STrack instance.

#         Args:
#             xywh (list[float]): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y)
#                 is the center, (w, h) are width and height, and `idx` is the detection index.
#             score (float): Confidence score of the detection.
#             cls (Any): Class label for the detected object.
#         """
#         super().__init__()
#         # xywh+idx or xywha+idx
#         assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
#         self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
#         self.kalman_filter = None
#         self.mean, self.covariance = None, None
#         self.is_activated = False

#         self.score = score
#         self.tracklet_len = 0
#         self.cls = cls
#         self.idx = xywh[-1]
#         self.angle = xywh[4] if len(xywh) == 6 else None

#     def predict(self):
#         """Predict the next state (mean and covariance) of the object using the Kalman filter."""
#         mean_state = self.mean.copy()
#         if self.state != TrackState.Tracked:
#             mean_state[7] = 0
#         self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

#     @staticmethod
#     def multi_predict(stracks: list[STrack]):
#         """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
#         if len(stracks) <= 0:
#             return
#         multi_mean = np.asarray([st.mean.copy() for st in stracks])
#         multi_covariance = np.asarray([st.covariance for st in stracks])
#         for i, st in enumerate(stracks):
#             if st.state != TrackState.Tracked:
#                 multi_mean[i][7] = 0
#         multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
#         for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#             stracks[i].mean = mean
#             stracks[i].covariance = cov

#     @staticmethod
#     def multi_gmc(stracks: list[STrack], H: np.ndarray = np.eye(2, 3)):
#         """Update state tracks positions and covariances using a homography matrix for multiple tracks."""
#         if stracks:
#             multi_mean = np.asarray([st.mean.copy() for st in stracks])
#             multi_covariance = np.asarray([st.covariance for st in stracks])

#             R = H[:2, :2]
#             R8x8 = np.kron(np.eye(4, dtype=float), R)
#             t = H[:2, 2]

#             for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#                 mean = R8x8.dot(mean)
#                 mean[:2] += t
#                 cov = R8x8.dot(cov).dot(R8x8.transpose())

#                 stracks[i].mean = mean
#                 stracks[i].covariance = cov

#     def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
#         """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
#         self.kalman_filter = kalman_filter
#         self.track_id = self.next_id()
#         self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

#         self.tracklet_len = 0
#         self.state = TrackState.Tracked
#         if frame_id == 1:
#             self.is_activated = True
#         self.frame_id = frame_id
#         self.start_frame = frame_id

#     def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
#         """Reactivate a previously lost track using new detection data and update its state and attributes."""
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, self.convert_coords(new_track.tlwh)
#         )
#         self.tracklet_len = 0
#         self.state = TrackState.Tracked
#         self.is_activated = True
#         self.frame_id = frame_id
#         if new_id:
#             self.track_id = self.next_id()
#         self.score = new_track.score
#         self.cls = new_track.cls
#         self.angle = new_track.angle
#         self.idx = new_track.idx

#     def update(self, new_track: STrack, frame_id: int):
#         """Update the state of a matched track.

#         Args:
#             new_track (STrack): The new track containing updated information.
#             frame_id (int): The ID of the current frame.

#         Examples:
#             Update the state of a track with new detection information
#             >>> track = STrack([100, 200, 50, 80, 0.9, 1])
#             >>> new_track = STrack([105, 205, 55, 85, 0.95, 1])
#             >>> track.update(new_track, 2)
#         """
#         self.frame_id = frame_id
#         self.tracklet_len += 1

#         new_tlwh = new_track.tlwh
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, self.convert_coords(new_tlwh)
#         )
#         self.state = TrackState.Tracked
#         self.is_activated = True

#         self.score = new_track.score
#         self.cls = new_track.cls
#         self.angle = new_track.angle
#         self.idx = new_track.idx

#     def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
#         """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
#         return self.tlwh_to_xyah(tlwh)

#     @property
#     def tlwh(self) -> np.ndarray:
#         """Get the bounding box in top-left-width-height format from the current state estimate."""
#         if self.mean is None:
#             return self._tlwh.copy()
#         ret = self.mean[:4].copy()
#         ret[2] *= ret[3]
#         ret[:2] -= ret[2:] / 2
#         return ret

#     @property
#     def xyxy(self) -> np.ndarray:
#         """Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
#         ret = self.tlwh.copy()
#         ret[2:] += ret[:2]
#         return ret

#     @staticmethod
#     def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
#         """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
#         ret = np.asarray(tlwh).copy()
#         ret[:2] += ret[2:] / 2
#         ret[2] /= ret[3]
#         return ret

#     @property
#     def xywh(self) -> np.ndarray:
#         """Get the current position of the bounding box in (center x, center y, width, height) format."""
#         ret = np.asarray(self.tlwh).copy()
#         ret[:2] += ret[2:] / 2
#         return ret

#     @property
#     def xywha(self) -> np.ndarray:
#         """Get position in (center x, center y, width, height, angle) format, warning if angle is missing."""
#         if self.angle is None:
#             LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
#             return self.xywh
#         return np.concatenate([self.xywh, self.angle[None]])

#     @property
#     def result(self) -> list[float]:
#         """Get the current tracking results in the appropriate bounding box format."""
#         coords = self.xyxy if self.angle is None else self.xywha
#         return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

#     def __repr__(self) -> str:
#         """Return a string representation of the STrack object including start frame, end frame, and track ID."""
#         return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


# class BYTETracker:
#     """BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

#     This class encapsulates the functionality for initializing, updating, and managing the tracks for detected objects
#     in a video sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman
#     filtering for predicting the new object locations, and performs data association.

#     Attributes:
#         tracked_stracks (list[STrack]): List of successfully activated tracks.
#         lost_stracks (list[STrack]): List of lost tracks.
#         removed_stracks (list[STrack]): List of removed tracks.
#         frame_id (int): The current frame ID.
#         args (Namespace): Command-line arguments.
#         max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
#         kalman_filter (KalmanFilterXYAH): Kalman Filter object.

#     Methods:
#         update: Update object tracker with new detections.
#         get_kalmanfilter: Return a Kalman filter object for tracking bounding boxes.
#         init_track: Initialize object tracking with detections.
#         get_dists: Calculate the distance between tracks and detections.
#         multi_predict: Predict the location of tracks.
#         reset_id: Reset the ID counter of STrack.
#         reset: Reset the tracker by clearing all tracks.
#         joint_stracks: Combine two lists of stracks.
#         sub_stracks: Filter out the stracks present in the second list from the first list.
#         remove_duplicate_stracks: Remove duplicate stracks based on IoU.

#     Examples:
#         Initialize BYTETracker and update with detection results
#         >>> tracker = BYTETracker(args, frame_rate=30)
#         >>> results = yolo_model.detect(image)
#         >>> tracked_objects = tracker.update(results)
#     """

#     def __init__(self, args, frame_rate: int = 30):
#         """Initialize a BYTETracker instance for object tracking.

#         Args:
#             args (Namespace): Command-line arguments containing tracking parameters.
#             frame_rate (int): Frame rate of the video sequence.
#         """
#         self.tracked_stracks = []  # type: list[STrack]
#         self.lost_stracks = []  # type: list[STrack]
#         self.removed_stracks = []  # type: list[STrack]

#         self.frame_id = 0
#         self.args = args
#         self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
#         self.kalman_filter = self.get_kalmanfilter()
#         gmc_method = getattr(args, "gmc_method", "sparseOptFlow")
#         self.gmc = GMC(method=gmc_method)

#         self.reset_id()

#     def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
#         """Update the tracker with new detections and return the current list of tracked objects."""
#         self.frame_id += 1
#         activated_stracks = []
#         refind_stracks = []
#         lost_stracks = []
#         removed_stracks = []

#         scores = results.conf
#         remain_inds = scores >= self.args.track_high_thresh
#         inds_low = scores > self.args.track_low_thresh
#         inds_high = scores < self.args.track_high_thresh

#         inds_second = inds_low & inds_high
#         results_second = results[inds_second]
#         results = results[remain_inds]
#         feats_keep = feats_second = img
#         if feats is not None and len(feats):
#             feats_keep = feats[remain_inds]
#             feats_second = feats[inds_second]

#         detections = self.init_track(results, feats_keep)
#         # Add newly detected tracklets to tracked_stracks
#         unconfirmed = []
#         tracked_stracks = []  # type: list[STrack]
#         for track in self.tracked_stracks:
#             if not track.is_activated:
#                 unconfirmed.append(track)
#             else:
#                 tracked_stracks.append(track)
#         # Step 2: First association, with high score detection boxes
#         strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
#         # Predict the current location with KF
#         self.multi_predict(strack_pool)
#         if hasattr(self, "gmc") and img is not None:
#             # use try-except here to bypass errors from gmc module
#             try:
#                 warp = self.gmc.apply(img, results.xyxy)
#             except Exception:
#                 warp = np.eye(2, 3)
#             STrack.multi_gmc(strack_pool, warp)
#             STrack.multi_gmc(unconfirmed, warp)

#         dists = self.get_dists(strack_pool, detections)
#         matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

#         for itracked, idet in matches:
#             track = strack_pool[itracked]
#             det = detections[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(det, self.frame_id)
#                 activated_stracks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False)
#                 refind_stracks.append(track)
#         # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
#         detections_second = self.init_track(results_second, feats_second)
#         r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
#         # TODO: consider fusing scores or appearance features for second association.
#         dists = matching.iou_distance(r_tracked_stracks, detections_second)
#         matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)
#         for itracked, idet in matches:
#             track = r_tracked_stracks[itracked]
#             det = detections_second[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(det, self.frame_id)
#                 activated_stracks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False)
#                 refind_stracks.append(track)

#         for it in u_track:
#             track = r_tracked_stracks[it]
#             if track.state != TrackState.Lost:
#                 track.mark_lost()
#                 lost_stracks.append(track)
#         # Deal with unconfirmed tracks, usually tracks with only one beginning frame
#         detections = [detections[i] for i in u_detection]
#         dists = self.get_dists(unconfirmed, detections)
#         matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
#         for itracked, idet in matches:
#             unconfirmed[itracked].update(detections[idet], self.frame_id)
#             activated_stracks.append(unconfirmed[itracked])
#         for it in u_unconfirmed:
#             track = unconfirmed[it]
#             track.mark_removed()
#             removed_stracks.append(track)
#         # Step 4: Init new stracks
#         for inew in u_detection:
#             track = detections[inew]
#             if track.score < self.args.new_track_thresh:
#                 continue
#             track.activate(self.kalman_filter, self.frame_id)
#             activated_stracks.append(track)
#         # Step 5: Update state
#         for track in self.lost_stracks:
#             if self.frame_id - track.end_frame > self.max_time_lost:
#                 track.mark_removed()
#                 removed_stracks.append(track)

#         self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
#         self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
#         self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
#         self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
#         self.lost_stracks.extend(lost_stracks)
#         self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
#         self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
#         self.removed_stracks.extend(removed_stracks)
#         if len(self.removed_stracks) > 1000:
#             self.removed_stracks = self.removed_stracks[-1000:]  # clip removed stracks to 1000 maximum

#         return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

#     def get_kalmanfilter(self) -> KalmanFilterXYAH:
#         """Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
#         return KalmanFilterXYAH()

#     def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
#         """Initialize object tracking with given detections, scores, and class labels using the STrack algorithm."""
#         if len(results) == 0:
#             return []
#         bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
#         bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
#         return [STrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

#     def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
#         """Calculate the distance between tracks and detections using IoU and optionally fuse scores."""
#         dists = matching.iou_distance(tracks, detections)
#         if self.args.fuse_score:
#             dists = matching.fuse_score(dists, detections)
#         return dists

#     def multi_predict(self, tracks: list[STrack]):
#         """Predict the next states for multiple tracks using Kalman filter."""
#         STrack.multi_predict(tracks)

#     @staticmethod
#     def reset_id():
#         """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
#         STrack.reset_id()

#     def reset(self):
#         """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
#         self.tracked_stracks = []  # type: list[STrack]
#         self.lost_stracks = []  # type: list[STrack]
#         self.removed_stracks = []  # type: list[STrack]
#         self.frame_id = 0
#         self.kalman_filter = self.get_kalmanfilter()
#         self.reset_id()
#         if hasattr(self, "gmc") and self.gmc is not None:
#             self.gmc.reset_params()

#     @staticmethod
#     def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
#         """Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
#         exists = {}
#         res = []
#         for t in tlista:
#             exists[t.track_id] = 1
#             res.append(t)
#         for t in tlistb:
#             tid = t.track_id
#             if not exists.get(tid, 0):
#                 exists[tid] = 1
#                 res.append(t)
#         return res

#     @staticmethod
#     def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
#         """Filter out the stracks present in the second list from the first list."""
#         track_ids_b = {t.track_id for t in tlistb}
#         return [t for t in tlista if t.track_id not in track_ids_b]

#     @staticmethod
#     def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]:
#         """Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
#         pdist = matching.iou_distance(stracksa, stracksb)
#         pairs = np.where(pdist < 0.15)
#         dupa, dupb = [], []
#         for p, q in zip(*pairs):
#             timep = stracksa[p].frame_id - stracksa[p].start_frame
#             timeq = stracksb[q].frame_id - stracksb[q].start_frame
#             if timep > timeq:
#                 dupb.append(q)
#             else:
#                 dupa.append(p)
#         resa = [t for i, t in enumerate(stracksa) if i not in dupa]
#         resb = [t for i, t in enumerate(stracksb) if i not in dupb]
#         return resa, resb





# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


class STrack(BaseTrack):
    """Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (Any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.
        angle (float | None): Optional angle information for oriented bounding boxes.

    Methods:
        predict: Predict the next state of the object using Kalman filter.
        multi_predict: Predict the next states for multiple tracks.
        multi_gmc: Update multiple track states using a homography matrix.
        activate: Activate a new tracklet.
        re_activate: Reactivate a previously lost tracklet.
        update: Update the state of a matched track.
        convert_coords: Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah: Convert tlwh bounding box to xyah format.

    Examples:
        Initialize and activate a new track
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: list[float], score: float, cls: Any):
        """Initialize a new STrack instance.

        Args:
            xywh (list[float]): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y)
                is the center, (w, h) are width and height, and `idx` is the detection index.
            score (float): Confidence score of the detection.
            cls (Any): Class label for the detected object.
        """
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None
        # ===== OA-SORT add: occlusion cache =====
        self.occ_pred = 0.0          # OAM on KF prediction, used by OAO
        self.occ_obs = 0.0           # OAM on latest observation, used by BAM
        self.last_obs_tlwh = self._tlwh.copy()
        # =======================================

    def predict(self):
        """Predict the next state (mean and covariance) of the object using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[STrack]):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks: list[STrack], H: np.ndarray = np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix for multiple tracks."""
        if stracks:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        # ===== OA-SORT add =====
        self.last_obs_tlwh = self._tlwh.copy()
        self.occ_pred = 0.0
        self.occ_obs = 0.0
        # =======================

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Reactivate a previously lost track using new detection data and update its state and attributes."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx
        # ===== OA-SORT add =====
        self.last_obs_tlwh = new_track.tlwh.copy()
        # occ_obs 不在这里直接算，统一在帧末刷新
        # =======================

    def update(self, new_track: STrack, frame_id: int):
        """Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.

        Examples:
            Update the state of a track with new detection information
            >>> track = STrack([100, 200, 50, 80, 0.9, 1])
            >>> new_track = STrack([105, 205, 55, 85, 0.95, 1])
            >>> track.update(new_track, 2)
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx
        # ===== OA-SORT add =====
        self.last_obs_tlwh = new_tlwh.copy()
        # occ_obs 不在这里直接算，统一在帧末刷新
        # =======================

    def update_with_tlwh(self, tlwh: np.ndarray, score: float, cls: Any, idx: int, frame_id: int, angle=None):
        """
        KF measurement update with a manually provided tlwh observation.
        Used by OA-SORT BAM in low-score association.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        tlwh = np.asarray(tlwh, dtype=np.float32)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(tlwh)
        )

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = score
        self.cls = cls
        self.idx = idx
        self.angle = angle
        self.last_obs_tlwh = tlwh.copy()

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        """Get the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self) -> np.ndarray:
        """Get position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        if self.angle is None:
            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """Get the current tracking results in the appropriate bounding box format."""
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """Return a string representation of the STrack object including start frame, end frame, and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

    This class encapsulates the functionality for initializing, updating, and managing the tracks for detected objects
    in a video sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman
    filtering for predicting the new object locations, and performs data association.

    Attributes:
        tracked_stracks (list[STrack]): List of successfully activated tracks.
        lost_stracks (list[STrack]): List of lost tracks.
        removed_stracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (Namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (KalmanFilterXYAH): Kalman Filter object.

    Methods:
        update: Update object tracker with new detections.
        get_kalmanfilter: Return a Kalman filter object for tracking bounding boxes.
        init_track: Initialize object tracking with detections.
        get_dists: Calculate the distance between tracks and detections.
        multi_predict: Predict the location of tracks.
        reset_id: Reset the ID counter of STrack.
        reset: Reset the tracker by clearing all tracks.
        joint_stracks: Combine two lists of stracks.
        sub_stracks: Filter out the stracks present in the second list from the first list.
        remove_duplicate_stracks: Remove duplicate stracks based on IoU.

    Examples:
        Initialize BYTETracker and update with detection results
        >>> tracker = BYTETracker(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize a BYTETracker instance for object tracking.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video sequence.
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

        # ===== OA-SORT add: default hyper-params =====
        defaults = {
            "oa_use_gm": True,              # whether to use Gaussian Map in OAM
            "oa_sigma_x": 0.25,             # sigma_x = oa_sigma_x * bbox_width
            "oa_sigma_y": 0.20,             # sigma_y = oa_sigma_y * bbox_height
            "oa_beta": 0.15,                # OAO strength
            "oa_depth_margin": 5.0,         # paper uses threshold 5 for bottom-edge ordering
            "oa_occ_gate": 0.05,            # ignore tiny occlusion coefficients
            "oa_bam_min_alpha": 0.15,       # minimum trust on low-score detection
            "oa_second_match_thresh": 0.5,  # second-stage association threshold
        }
        for k, v in defaults.items():
            if not hasattr(self.args, k):
                setattr(self.args, k, v)
        # =============================================


    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update the tracker with new detections and return the current list of tracked objects."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        inds_second = inds_low & inds_high

        results_second = results[inds_second]
        results = results[remain_inds]
        print(len(results_second),len(results))
        feats_keep = feats_second = img
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        detections = self.init_track(results, feats_keep)

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # ------------------------------------------------------------------
        # Step 2: First association, with high score detection boxes
        # ------------------------------------------------------------------
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        self.multi_predict(strack_pool)

        if hasattr(self, "gmc") and img is not None:
            # use try-except here to bypass errors from gmc module
            try:
                warp = self.gmc.apply(img, results.xyxy)
            except Exception:
                warp = np.eye(2, 3)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # ===== OA-SORT insert #1: OAM on KF predictions =====
        self._update_pred_occlusion(strack_pool)
        # ===== OA-SORT insert #2: OAO in first-stage association =====
        dists = self.get_dists(strack_pool, detections, stage="first")
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # ------------------------------------------------------------------
        # Step 3: Second association, with low score detection boxes
        # ------------------------------------------------------------------
        detections_second = self.init_track(results_second, feats_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # if self.args.fuse_score:
        #     dists = matching.fuse_score(dists, detections_second)

        matches, u_track, _u_detection_second = matching.linear_assignment(
            dists, thresh=float(self.args.oa_second_match_thresh)
        )
        print(len(matches))
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            # ===== OA-SORT insert #3: BAM before low-score KF update =====
            fused_tlwh = self._bam_fused_tlwh(track, det)
            track.update_with_tlwh(
                tlwh=fused_tlwh,
                score=det.score,
                cls=det.cls,
                idx=det.idx,
                frame_id=self.frame_id,
                angle=det.angle,
            )
            # track.update(det, self.frame_id)
            activated_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ------------------------------------------------------------------
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        # ------------------------------------------------------------------
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections, stage="plain")
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # ------------------------------------------------------------------
        # Step 4: Init new stracks
        # ------------------------------------------------------------------
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # ------------------------------------------------------------------
        # Step 5: Update state
        # ------------------------------------------------------------------
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)

        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]  # clip removed stracks to 1000 maximum

        # ===== OA-SORT insert #4: OAM on latest observations for next-frame BAM =====
        self._update_obs_occlusion(self.tracked_stracks)

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self) -> KalmanFilterXYAH:
        """Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    # ===== OA-SORT add: helper functions =====
    @staticmethod
    def _tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
        x, y, w, h = tlwh.astype(np.float32)
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    @staticmethod
    def _xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
        """
        xyah = [cx, cy, a, h], where a = w / h
        """
        xyah = np.asarray(xyah, dtype=np.float32).copy()
        w = xyah[2] * xyah[3]
        h = xyah[3]
        x = xyah[0] - w / 2
        y = xyah[1] - h / 2
        return np.array([x, y, w, h], dtype=np.float32)


    def _xywh_to_tlwh(self, xywh: np.ndarray) -> np.ndarray:
        xywh = np.asarray(xywh, dtype=np.float32).copy()
        xywh[:2] -= xywh[2:] / 2
        return xywh

    def _make_box_gaussian(self, w: int, h: int) -> np.ndarray:
        """
        Gaussian Map inside one box.
        Center pixels get larger weights, edges get smaller weights.
        """
        w = max(int(w), 1)
        h = max(int(h), 1)
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        gx = np.exp(-0.5 * (xs / max(float(self.args.oa_sigma_x), 1e-6)) ** 2)
        gy = np.exp(-0.5 * (ys / max(float(self.args.oa_sigma_y), 1e-6)) ** 2)
        gm = np.outer(gy, gx).astype(np.float32)
        gm /= (gm.sum() + 1e-6)
        return gm

    def _compute_occ_coeffs(self, tlwh_list: list[np.ndarray]) -> np.ndarray:
        """
        OAM:
        1) depth ordering by bottom-edge y
        2) occlusion coefficient from overlap ratio
        3) optional Gaussian Map refinement
        """
        n = len(tlwh_list)
        if n == 0:
            return np.zeros((0,), dtype=np.float32)

        tlbr = np.asarray([self._tlwh_to_tlbr(t) for t in tlwh_list], dtype=np.float32)
        bottoms = tlbr[:, 3]
        occ = np.zeros((n,), dtype=np.float32)

        for j in range(n):
            x1j, y1j, x2j, y2j = tlbr[j]
            wj = max(x2j - x1j, 1.0)
            hj = max(y2j - y1j, 1.0)

            H = max(int(np.ceil(hj)), 1)
            W = max(int(np.ceil(wj)), 1)
            occ_mask = np.zeros((H, W), dtype=np.bool_)

            if self.args.oa_use_gm:
                weight = self._make_box_gaussian(W, H)
            else:
                weight = np.ones((H, W), dtype=np.float32)
                weight /= (weight.sum() + 1e-6)

            for i in range(n):
                if i == j:
                    continue

                # Depth ordering:
                # i is in front of j if bottom_i > bottom_j + margin
                if bottoms[i] <= bottoms[j] + float(self.args.oa_depth_margin):
                    continue

                x1i, y1i, x2i, y2i = tlbr[i]
                xx1 = max(x1i, x1j)
                yy1 = max(y1i, y1j)
                xx2 = min(x2i, x2j)
                yy2 = min(y2i, y2j)

                if xx2 <= xx1 or yy2 <= yy1:
                    continue

                # map overlap region to local coordinates of target j
                lx1 = int(np.clip(np.floor(xx1 - x1j), 0, W))
                ly1 = int(np.clip(np.floor(yy1 - y1j), 0, H))
                lx2 = int(np.clip(np.ceil(xx2 - x1j), 0, W))
                ly2 = int(np.clip(np.ceil(yy2 - y1j), 0, H))

                if lx2 > lx1 and ly2 > ly1:
                    occ_mask[ly1:ly2, lx1:lx2] = True

            occ[j] = float((weight * occ_mask.astype(np.float32)).sum())

        return np.clip(occ, 0.0, 1.0)

    def _update_pred_occlusion(self, tracks: list[STrack]) -> None:
        """
        OAM on KF predictions, used by OAO.
        """
        occ = self._compute_occ_coeffs([t.tlwh.copy() for t in tracks])
        for t, c in zip(tracks, occ):
            t.occ_pred = float(c)

    def _update_obs_occlusion(self, tracks: list[STrack]) -> None:
        """
        OAM on latest observations, used by BAM.
        """
        occ = self._compute_occ_coeffs([t.last_obs_tlwh.copy() for t in tracks])
        for t, c in zip(tracks, occ):
            t.occ_obs = float(c)

    def _oao_adjust_row_penalty(self, dists: np.ndarray, tracks: list[STrack]) -> np.ndarray:
        """
        OAO:
        apply extra penalty to rows (tracks) that are more occluded.
        This keeps ByteTrack's original IoU/detection-score logic intact,
        but makes highly occluded predictions less competitive in first-stage matching.
        """
        if dists.size == 0:
            return dists

        occ = np.asarray([t.occ_pred for t in tracks], dtype=np.float32)[:, None]
        occ = np.where(occ >= float(self.args.oa_occ_gate), occ, 0.0)

        # Conservative engineering version of Eq.(8)
        dists = dists + float(self.args.oa_beta) * occ
        return np.clip(dists, 0.0, 1.0)

    def _bam_fused_tlwh(self, track: STrack, det: STrack) -> np.ndarray:
        """
        BAM:
        blend predicted KF state with low-score detection before KF update.
        Larger discrepancy / heavier occlusion -> trust prediction more.
        """
        pred_xyah = track.mean[:4].copy() if track.mean is not None else track.convert_coords(track.tlwh)
        det_xyah = track.convert_coords(det.tlwh)

        # IoU between current prediction and low-score detection
        iou_pd = float(1.0 - matching.iou_distance([track], [det])[0, 0])

        occ = max(float(track.occ_obs) - float(self.args.oa_occ_gate), 0.0)
        alpha = (1.0 - occ) * iou_pd
        alpha = float(np.clip(alpha, float(self.args.oa_bam_min_alpha), 1.0))

        fused_xyah = alpha * det_xyah + (1.0 - alpha) * pred_xyah
        # fused_xyah = det_xyah
        # return self._xyah_to_tlwh(fused_xyah)
        tracker_type = getattr(self.args, "tracker_type", "bytetrack").lower()
        if tracker_type == "botsort":
            return self._xywh_to_tlwh(fused_xyah)
        else:
            return self._xyah_to_tlwh(fused_xyah)
        # return  self._xywh_to_tlwh(fused_xyah)
    # ===========================================


    def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
        """Initialize object tracking with given detections, scores, and class labels using the STrack algorithm."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        return [STrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[STrack], detections: list[STrack], stage: str = "first") -> np.ndarray:
        """
        stage:
            - 'first': apply OAO
            - others : keep original ByteTrack behavior
        """
        dists = matching.iou_distance(tracks, detections)

        # OA-SORT says OAO is only used in first-stage association for high-score detections
        if stage == "first":
            dists = self._oao_adjust_row_penalty(dists, tracks)

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        return dists

    def multi_predict(self, tracks: list[STrack]):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Filter out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]:
        """Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb