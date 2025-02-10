import logging
import datetime
import os
from typing import Dict
from backend.google_drive_service import GoogleDriveService
from backend.config import config

logger = logging.getLogger(__name__)

class EventLogger:
    """
    イベントログを記録するモジュール。
    ローカルファイルとGoogle Driveにログを保存。
    """
    def __init__(self, log_file: str, log_dir_gdrive: str):
        self.log_file = os.path.abspath(log_file)
        self.log_dir_gdrive = log_dir_gdrive
        
        # ログファイルのディレクトリを作成
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        self.gdrive_service = GoogleDriveService(config.google_drive_credentials_path)
        self._create_gdrive_log_dir()
        self.log_event_called = None

    def _create_gdrive_log_dir(self):
        try:
            if not self.gdrive_service.directory_exists(self.log_dir_gdrive):
                self.gdrive_service.create_directory(self.log_dir_gdrive)
                logger.info(f"Created Google Drive log directory: '{self.log_dir_gdrive}'")
            else:
                logger.info(f"Google Drive log directory '{self.log_dir_gdrive}' already exists.")
        except Exception as e:
            logger.error(f"Error creating Google Drive log directory: {e}")

    def log_event(self, event: Dict) -> None:
        if hasattr(self, 'log_event_called') and isinstance(self.log_event_called, bool):
            raise TypeError("log_event_called should not be a boolean")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"Event: {event.get('event_type', 'unknown')}, Details: {event.get('details', 'N/A')}"
        log_level_str = event.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        # ローカルファイルにログを記録
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                log_entry = f"{timestamp} - {logging.getLevelName(log_level)} - {__name__} - {log_message}\n"
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Error writing to local log file: {e}")

        # Google Driveにログを記録
        try:
            log_filename_gdrive = f"baby_monitor_log_{datetime.datetime.now().strftime('%Y%m%d')}.log"
            log_content = f"{timestamp} - {logging.getLevelName(log_level)} - {__name__} - {log_message}\n"
            
            # Google Driveにファイルを追加
            self.gdrive_service.append_text_file(
                self.log_dir_gdrive,
                log_filename_gdrive,
                log_content
            )
            logger.debug(f"Event logged to Google Drive: {log_filename_gdrive}")
        except Exception as e:
            logger.error(f"Error logging to Google Drive: {e}")

    def rotate_logs(self) -> None:
        logger.info("Log rotation (dummy) - Not implemented yet.") 