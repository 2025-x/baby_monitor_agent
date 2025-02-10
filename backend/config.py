import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    アプリケーション全体の設定を管理するクラス。
    環境変数、設定ファイルの値を読み込む。
    """
    def __init__(self):
        # カメラ設定
        self.camera_rtsp_url = os.environ.get("CAMERA_RTSP_URL", "rtsp://your_camera_ip/stream")
        self.frame_rate = int(os.environ.get("CAMERA_FRAME_RATE", 5))

        # モーション検出設定
        self.motion_threshold = int(os.environ.get("MOTION_THRESHOLD", 5000))

        # 危険検出設定
        self.danger_config = {
            "stuck_time_threshold": int(os.environ.get("STUCK_TIME_THRESHOLD", 300)),  # うつ伏せで動かない秒数
            "min_confidence_score": float(os.environ.get("MIN_CONFIDENCE_SCORE", 0.7)) # Gemini解析の信頼度閾値
        }

        # リトライ設定 (カメラ)
        self.camera_max_retry = int(os.environ.get("CAMERA_MAX_RETRY", 3))
        self.camera_retry_wait = float(os.environ.get("CAMERA_RETRY_WAIT", 2.0))

        # リトライ設定 (Gemini)
        self.gemini_max_retry = int(os.environ.get("GEMINI_MAX_RETRY", 3))
        self.gemini_retry_wait = float(os.environ.get("GEMINI_RETRY_WAIT", 1.0))
        self.gemini_timeout = float(os.environ.get("GEMINI_TIMEOUT", 15.0))

        # 通知設定 (Gmail)
        self.gmail_config = {
            "smtp_server": os.environ.get("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.environ.get("SMTP_PORT", 587)),
            "from_email": os.environ.get("FROM_EMAIL", "your_email@gmail.com"),
            "to_email": os.environ.get("TO_EMAIL", "parent_email@gmail.com"),
            "username": os.environ.get("SMTP_USERNAME", "your_email@gmail.com"),
            "password": os.environ.get("SMTP_PASSWORD", "your_gmail_app_password")
        }

        # ログ設定
        self.log_file = os.environ.get("LOG_FILE", "baby_monitor.log")
        self.log_dir_gdrive = os.environ.get("LOG_DIR_GDRIVE", "baby_monitor_logs")

        # Diary設定 (Google Drive)
        self.diary_dir_gdrive = os.environ.get("DIARY_DIR_GDRIVE", "baby_monitor_diary")
        self.daily_digest_time = os.environ.get("DAILY_DIGEST_TIME", "21:00")

        # Gemini API
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env file")

        # Google Drive API 認証情報
        self.google_drive_credentials_path = os.environ.get("GOOGLE_DRIVE_CREDENTIALS_PATH")
        if not self.google_drive_credentials_path:
            raise ValueError("GOOGLE_DRIVE_CREDENTIALS_PATH is not set in .env file")

        # 画像処理設定
        self.image_max_size = int(os.environ.get("IMAGE_MAX_SIZE", 1024))
        self.image_quality = int(os.environ.get("IMAGE_QUALITY", 85))

config = Config() 