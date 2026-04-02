# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


class BOTrack(STrack):
    """An extended version of the STrack class for YOLO, adding object tracking features.

    This class extends the STrack class to include additional functionalities for object tracking, such as feature
    smoothing, Kalman filter prediction, and reactivation of tracks.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features: Update features vector and smooth it using exponential moving average.
        predict: Predict the mean and covariance using Kalman filter.
        re_activate: Reactivate a track with updated features and optionally new ID.
        update: Update the track with new detection and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict: Predict the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords: Convert tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh: Convert bounding box to xywh format `(center x, center y, width, height)`.

    Examples:
        Create a BOTrack instance and update its features
        >>> bo_track = BOTrack(xywh=np.array([100, 50, 80, 40, 0]), score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.predict()
        >>> new_track = BOTrack(xywh=np.array([110, 60, 80, 40, 0]), score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(
        self, xywh: np.ndarray, score: float, cls: int, feat: np.ndarray | None = None, feat_history: int = 50
    ):
        """Initialize a BOTrack object with temporal parameters, such as feature history, alpha, and current features.

        Args:
            xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y) is
                the center, (w, h) are width and height, and `idx` is the detection index.
            score (float): Confidence score of the detection.
            cls (int): Class ID of the detected object.
            feat (np.ndarray, optional): Feature vector associated with the detection.
            feat_history (int): Maximum length of the feature history deque.
        """
        super().__init__(xywh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        self.prev_post_mean = None
        self.prev_post_cov = None

        self.pred_mean = None
        self.pred_cov = None

        self.post_mean = None
        self.post_cov = None

        self.meas_xywh = None
        self.innovation = None

        self.matched_this_frame = False

    def update_features(self, feat: np.ndarray) -> None:
        """Update the feature vector and apply exponential moving average smoothing."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    # def predict(self) -> None:
    #     """Predict the object's future state using the Kalman filter to update its mean and covariance."""
    #     mean_state = self.mean.copy()
    #     if self.state != TrackState.Tracked:
    #         mean_state[6] = 0
    #         mean_state[7] = 0

    #     self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)


    def predict(self) -> None:
        if self.mean is not None:
            self.prev_post_mean = self.mean.copy()
        if self.covariance is not None:
            self.prev_post_cov = self.covariance.copy()

        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

        self.pred_mean = self.mean.copy()
        self.pred_cov = self.covariance.copy()
        self.matched_this_frame = False



    # def re_activate(self, new_track: BOTrack, frame_id: int, new_id: bool = False) -> None:
    #     """Reactivate a track with updated features and optionally assign a new ID."""
    #     if new_track.curr_feat is not None:
    #         self.update_features(new_track.curr_feat)
    #     super().re_activate(new_track, frame_id, new_id)

    # def update(self, new_track: BOTrack, frame_id: int) -> None:
    #     """Update the track with new detection information and the current frame ID."""
    #     if new_track.curr_feat is not None:
    #         self.update_features(new_track.curr_feat)
    #     super().update(new_track, frame_id)


    def update(self, new_track, frame_id):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        new_tlwh = new_track.tlwh
        meas_xywh = self.convert_coords(new_tlwh).copy()   # BOTSORT 这里是 xywh
        self.meas_xywh = meas_xywh

        prior = self.pred_mean if self.pred_mean is not None else self.mean
        if prior is not None:
            self.innovation = meas_xywh - prior[:4]
        else:
            self.innovation = np.zeros(4, dtype=np.float32)

        super().update(new_track, frame_id)

        self.post_mean = self.mean.copy()
        self.post_cov = self.covariance.copy()
        self.matched_this_frame = True


    def re_activate(self, new_track, frame_id, new_id=False):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        new_tlwh = new_track.tlwh
        meas_xywh = self.convert_coords(new_tlwh).copy()
        self.meas_xywh = meas_xywh

        prior = self.pred_mean if self.pred_mean is not None else self.mean
        if prior is not None:
            self.innovation = meas_xywh - prior[:4]
        else:
            self.innovation = np.zeros(4, dtype=np.float32)

        super().re_activate(new_track, frame_id, new_id)

        self.post_mean = self.mean.copy()
        self.post_cov = self.covariance.copy()
        self.matched_this_frame = True


    def activate(self, kalman_filter, frame_id):
        super().activate(kalman_filter, frame_id)
        self.post_mean = self.mean.copy()
        self.post_cov = self.covariance.copy()
        self.pred_mean = self.mean.copy()
        self.pred_cov = self.covariance.copy()
        self.meas_xywh = self.mean[:4].copy()
        self.innovation = np.zeros(4, dtype=np.float32)
        self.matched_this_frame = True


    def xywh_to_xyxy(xywh):
        cx, cy, w, h = map(float, xywh[:4])
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return x1, y1, x2, y2

    def get_track_kf_state(t):
        out = {
            "pred_xyxy": (np.nan, np.nan, np.nan, np.nan),
            "post_xyxy": (np.nan, np.nan, np.nan, np.nan),
            "vx": np.nan,
            "vy": np.nan,
            "vw": np.nan,
            "vh": np.nan,
            "innov_x": np.nan,
            "innov_y": np.nan,
            "innov_w": np.nan,
            "innov_h": np.nan,
            "matched": 0,
        }

        pred_mean = getattr(t, "pred_mean", None)
        post_mean = getattr(t, "post_mean", None)
        innovation = getattr(t, "innovation", None)

        if pred_mean is not None:
            out["pred_xyxy"] = xywh_to_xyxy(pred_mean[:4])

        if post_mean is not None:
            out["post_xyxy"] = xywh_to_xyxy(post_mean[:4])
            if len(post_mean) >= 8:
                out["vx"] = float(post_mean[4])
                out["vy"] = float(post_mean[5])
                out["vw"] = float(post_mean[6])
                out["vh"] = float(post_mean[7])

        if innovation is not None and len(innovation) >= 4:
            out["innov_x"] = float(innovation[0])
            out["innov_y"] = float(innovation[1])
            out["innov_w"] = float(innovation[2])
            out["innov_h"] = float(innovation[3])

        out["matched"] = int(bool(getattr(t, "matched_this_frame", False)))
        return out

    @property
    def tlwh(self) -> np.ndarray:
        """Return the current bounding box position in `(top left x, top left y, width, height)` format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks: list[BOTrack]) -> None:
        """Predict the mean and covariance for multiple object tracks using a shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

            # stracks[i].pred_mean = mean.copy()
            # stracks[i].pred_cov = cov.copy()
            # stracks[i].matched_this_frame = False

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert tlwh bounding box coordinates to xywh format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """An extended version of the BYTETracker class for YOLO, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (Any): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (Any): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter: Return an instance of KalmanFilterXYWH for object tracking.
        init_track: Initialize track with detections, scores, and classes.
        get_dists: Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict: Predict and track multiple objects with a YOLO model.
        reset: Reset the BOTSORT tracker to its initial state.

    Examples:
        Initialize BOTSORT and process detections
        >>> bot_sort = BOTSORT(args, frame_rate=30)
        >>> bot_sort.init_track(dets, scores, cls, img)
        >>> bot_sort.multi_predict(tracks)

    Notes:
        The class is designed to work with a YOLO object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args: Any, frame_rate: int = 30):
        """Initialize BOTSORT object with ReID module and GMC algorithm.

        Args:
            args (Any): Parsed command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video being processed.
        """
        super().__init__(args, frame_rate)
        self.gmc = GMC(method=args.gmc_method)

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = (
            (lambda feats, s: [f.cpu().numpy() for f in feats])  # native features do not require any model
            if args.with_reid and self.args.model == "auto"
            else ReID(args.model)
            if args.with_reid
            else None
        )

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        """Return an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process."""
        return KalmanFilterXYWH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[BOTrack]:
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder(img, bboxes)
            return [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features_keep)]
        else:
            return [BOTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    # def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
    #     """Calculate distances between tracks and detections using IoU and optionally ReID embeddings."""
    #     dists = matching.iou_distance(tracks, detections)
    #     dists_mask = dists > (1 - self.proximity_thresh)

    #     if self.args.fuse_score:
    #         dists = matching.fuse_score(dists, detections)

    #     if self.args.with_reid and self.encoder is not None:
    #         emb_dists = matching.embedding_distance(tracks, detections) / 2.0
    #         emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
    #         emb_dists[dists_mask] = 1.0
    #         dists = np.minimum(dists, emb_dists)
    #     return dists


    def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack], stage: str = "first") -> np.ndarray:
        """Calculate distances between tracks and detections using IoU and optionally ReID embeddings."""
        dists = super().get_dists(tracks, detections, stage)
        # dists = matching.iou_distance(tracks, detections)
        # dists_mask = dists > (1 - self.proximity_thresh)

        # if self.args.fuse_score:
        #     dists = matching.fuse_score(dists, detections)

        # if self.args.with_reid and self.encoder is not None:
        #     emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        #     emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
        #     emb_dists[dists_mask] = 1.0
        #     dists = np.minimum(dists, emb_dists)
        return dists


    def multi_predict(self, tracks: list[BOTrack]) -> None:
        """Predict the mean and covariance of multiple object tracks using a shared Kalman filter."""
        BOTrack.multi_predict(tracks)

    def reset(self) -> None:
        """Reset the BOTSORT tracker to its initial state, clearing all tracked objects and internal states."""
        super().reset()
        self.gmc.reset_params()


class ReID:
    """YOLO model as encoder for re-identification."""

    def __init__(self, model: str):
        """Initialize encoder for re-identification.

        Args:
            model (str): Path to the YOLO model for re-identification.
        """
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.model(embed=[len(self.model.model.model) - 2 if ".pt" in model else -1], verbose=False, save=False)  # init

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract embeddings for detected objects."""
        feats = self.model.predictor(
            [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        )
        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]  # batched prediction with non-PyTorch backend
        return [f.cpu().numpy() for f in feats]
