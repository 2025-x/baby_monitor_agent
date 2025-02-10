import cv2
import logging
import socket
import subprocess
import shutil
import os
import tempfile
from typing import Optional
import numpy as np
from backend.config import config
from urllib.parse import quote

logger = logging.getLogger(__name__)

class CameraCapture:
    """
    カメラから映像を取得するモジュール。
    """
    def __init__(self, rtsp_url: str, frame_rate: int = 30):
        """
        初期化処理。
        """
        # URLエンコーディングを適用
        try:
            # URLからユーザー名、パスワード、ホスト部分を抽出
            import re
            match = re.match(r'rtsp://([^:]+):([^@]+)@(.+)', rtsp_url)
            if match:
                username = quote(match.group(1))
                password = quote(match.group(2))
                host = match.group(3)
                self.rtsp_url = f"rtsp://{username}:{password}@{host}"
            else:
                self.rtsp_url = rtsp_url
        except Exception as e:
            logger.error(f"Error encoding RTSP URL: {e}")
            self.rtsp_url = rtsp_url

        self.frame_rate = frame_rate
        self.cap = None
        self.is_capturing = False
        self.ffmpeg_path = shutil.which("ffmpeg")
        if not self.ffmpeg_path:
            logger.error("ffmpeg not found in system PATH")

        # リトライ設定をconfigから取得
        self.max_retry = config.camera_max_retry
        self.retry_wait = config.camera_retry_wait

    def check_camera_connection(self) -> bool:
        """
        カメラとの接続をTCPレベルで確認
        """
        try:
            # URLからホストとポートを抽出
            import re
            match = re.search(r'@([^:]+):(\d+)', self.rtsp_url)
            if not match:
                logger.error("Invalid RTSP URL format")
                return False
            
            host = match.group(1)
            port = int(match.group(2))
            
            # TCPソケット接続を試行
            socket.create_connection((host, port), timeout=10)
            logger.info("Camera TCP connection successful")
            return True
        except (socket.timeout, socket.error) as e:
            logger.error(f"Camera TCP connection failed: {e}")
            return False

    def start_capture(self) -> bool:
        """カメラキャプチャを開始"""
        if not self.ffmpeg_path:
            logger.error("ffmpeg not found")
            return False

        if not self.check_camera_connection():
            logger.error("Camera connection check failed")
            return False

        try:
            # 一時的な動画ファイルを作成して、RTSPストリームから1フレームを取得
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name

            command = [
                self.ffmpeg_path,
                '-y',  # 既存ファイルの上書きを自動的に許可
                '-i', self.rtsp_url,
                '-t', '1',  # 1秒だけ取得
                '-vcodec', 'copy',
                temp_path
            ]

            subprocess.run(command, check=True, timeout=10)

            # 一時ファイルからフレームを読み込んでテスト
            self.cap = cv2.VideoCapture(temp_path)
            ret, _ = self.cap.read()
            self.cap.release()
            os.unlink(temp_path)

            if not ret:
                logger.error("Failed to read test frame from RTSP stream")
                return False

            # 実際のストリームを開く
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video capture from URL: {self.rtsp_url}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
            self.is_capturing = True
            logger.info("Camera capture started successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg command failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error starting camera capture: {e}")
            self.is_capturing = False
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """フレームを取得"""
        if not self.is_active():
            logger.warning("Camera is not active")
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                return None
            return frame
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None

    def reinit_capture(self) -> bool:
        """キャプチャを再初期化"""
        self.release()
        return self.start_capture()

    def release(self):
        """リソースを解放"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_capturing = False

    def is_active(self) -> bool:
        """キャプチャがアクティブかどうかを確認"""
        return self.is_capturing and self.cap is not None and self.cap.isOpened()

    def __del__(self):
        """デストラクタ"""
        self.release() 