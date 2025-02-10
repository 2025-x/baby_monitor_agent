import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MotionDetector:
    """
    映像フレーム間の差分から動きを検出するモジュール。
    ガウスブラーを適用し、照明変化などによる全体的な変化を誤検出しないように工夫している。
    """
    def __init__(self, threshold: int = 1000):
        """
        初期化処理。
        :param threshold: 動作検出のための差分閾値（後から調整可能）
        """
        self.threshold = threshold
        self.previous_frame = None
        # 全体の照明変化と判断するための変化割合の閾値（例：80%以上が変化していればグローバルな照明変化とみなす）
        self.global_change_ratio_threshold = 0.8

    def detect_motion(self, current_frame: np.ndarray) -> bool:
        """
        現在のフレームと前回のフレームを比較して動きを検出する。
        初回のフレームは、前のフレームが存在しないため、動作ありと判定する。
        また、全体的な照明変化の場合は動きを検出しないようにしている。
        
        :param current_frame: 現在の映像フレーム（BGR画像）
        :return: 動いていると判断した場合はTrue、そうでなければFalse
        """
        if current_frame is None:
            logger.warning("No frame received for motion detection")
            return False

        # ノイズ低減のためにガウスブラーを適用
        current_blur = cv2.GaussianBlur(current_frame, (5, 5), 0)

        if self.previous_frame is None:
            self.previous_frame = current_blur.copy()
            logger.debug("最初のフレームのため、動作なしと判定")
            return False  # 初回フレームは動作なしと判定

        # 差分計算
        frame_diff = cv2.absdiff(current_blur, self.previous_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # 全体の変化割合を計算（全画素中、どれだけのピクセルが変化しているか）
        height, width = diff_mask.shape
        num_pixels = height * width
        white_pixels = cv2.countNonZero(diff_mask)
        global_change_ratio = white_pixels / num_pixels

        logger.debug(
            f"Diff pixels: {np.sum(diff_mask)}, Threshold: {self.threshold}, Global change ratio: {global_change_ratio}"
        )

        if global_change_ratio > self.global_change_ratio_threshold:
            # 全体がほぼ一様な変化の場合、照明の変化とみなす（誤検出回避）
            motion_detected = False
            logger.debug("全体的な照明変化のため、誤検出と判断")
        else:
            diff_pixels = np.sum(diff_mask)
            motion_detected = diff_pixels > self.threshold

        self.previous_frame = current_blur.copy()
        return motion_detected

    def get_motion_score(self, current_frame: np.ndarray) -> int:
        """
        現在のフレームと前回のフレームの差分から動きのスコアを取得する。
        ※初回フレームの場合は差分がないため、スコアは0とする。
        
        :param current_frame: 現在の映像フレーム（BGR画像）
        :return: 差分スコア（動きの大きさを示す値）
        """
        current_blur = cv2.GaussianBlur(current_frame, (5, 5), 0)
        if self.previous_frame is None:
            return 0

        frame_diff = cv2.absdiff(current_blur, self.previous_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        diff_score = np.sum(diff_mask)
        return diff_score 