import datetime
import logging
import os
import cv2
import numpy as np
from typing import Dict, Optional

from backend.notifier import Notifier
from backend.config import config
from backend.google_drive_service import GoogleDriveService

# 追加: Gemini API および Pydantic のインポート
import google.generativeai as genai
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class Diary:
    """
    育児日記を記録・管理するモジュール。
    ローカルファイル、Google Driveに日記を保存し、日次ダイジェストをGmail送信する。
    """
    def __init__(self, config, notifier: Notifier):
        self.config = config
        self.notifier = notifier
        self.diary_entries = []
        self.last_digest_sent_date = None
        self.gdrive_service = GoogleDriveService(config.google_drive_credentials_path)
        self.diary_dir_gdrive = config.diary_dir_gdrive
        self._create_gdrive_diary_dir()

    def _create_gdrive_diary_dir(self):
        try:
            if not self.gdrive_service.directory_exists(self.diary_dir_gdrive):
                self.gdrive_service.create_directory(self.diary_dir_gdrive)
                logger.info(f"Created Google Drive diary directory: '{self.diary_dir_gdrive}'")
            else:
                logger.info(f"Google Drive diary directory '{self.diary_dir_gdrive}' already exists.")
        except Exception as e:
            logger.error(f"Error creating Google Drive diary directory: {e}")

    def record_event(self, event_data: Dict) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event_type = event_data.get("event_type", "unknown")
        details = event_data.get("details", "N/A")
        image_frame = event_data.get("image")
        is_highlight = event_data.get("is_highlight", False)
        if is_highlight:
            entry_text = f"{timestamp} - 【ハイライト】 イベント: {event_type}, 内容: {details}"
        else:
            entry_text = f"{timestamp} - イベント: {event_type}, 内容: {details}"
        logger.info(f"Diary entry: {entry_text}")
        self.diary_entries.append({
            "timestamp": timestamp,
            "text": entry_text,
            "image": image_frame,
            "is_highlight": is_highlight
        })
        self._save_diary_entry_local(entry_text, event_type, timestamp, image_frame)
        self._save_diary_entry_gdrive(entry_text, event_type, timestamp, image_frame)

    def _save_diary_entry_local(self, entry_text: str, event_type: str, timestamp: str, image_frame: Optional[np.ndarray]) -> None:
        diary_dir = "diary_local_logs"
        os.makedirs(diary_dir, exist_ok=True)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        log_filename = os.path.join(diary_dir, f"diary_{date_str}.log")
        try:
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(entry_text + "\n")
                if image_frame is not None and event_type != "daily_digest":
                    image_filename = os.path.join(diary_dir, f"event_image_{timestamp.replace(':', '-')}.jpg")
                    cv2.imwrite(image_filename, image_frame)
                    f.write(f"  - Image saved: {image_filename}\n")
            logger.info(f"Diary entry saved to local file: {log_filename}")
        except Exception as e:
            logger.error(f"Error saving diary entry to local file: {e}")

    def _save_diary_entry_gdrive(self, entry_text: str, event_type: str, timestamp: str, image_frame: Optional[np.ndarray]) -> None:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        diary_text_filename_gdrive = f"diary_text_{date_str}.txt"
        try:
            diary_line = f"{timestamp} - {entry_text}\n"
            self.gdrive_service.append_text_file(self.diary_dir_gdrive, diary_text_filename_gdrive, diary_line)
            logger.debug(f"Diary text entry appended to Google Drive: {diary_text_filename_gdrive}")
            if image_frame is not None and event_type != "daily_digest":
                image_filename_gdrive = f"event_image_{timestamp.replace(':', '-')}.jpg"
                image_bytes = cv2.imencode('.jpg', image_frame)[1].tobytes()
                self.gdrive_service.upload_file(self.diary_dir_gdrive, image_filename_gdrive, image_bytes, mime_type='image/jpeg')
                logger.debug(f"Diary image saved to Google Drive: {image_filename_gdrive}")
        except Exception as e:
            logger.error(f"Error saving diary entry to Google Drive: {e}")

    def generate_daily_digest(self) -> Optional[Dict]:
        today = datetime.date.today()
        if self.last_digest_sent_date == today:
            logger.info("本日のダイジェストは既に送信済みです。")
            return None
        if not self.diary_entries:
            logger.info("ダイジェスト生成のための日記エントリがありません。")
            return None

        # 全体ログの集約
        aggregated_text = "\n".join(entry["text"] for entry in self.diary_entries)

        # ハイライトとなるエントリは、フラグが立っているものから１件採用
        highlight_entry = None
        for entry in self.diary_entries:
            if entry.get("is_highlight", False):
                highlight_entry = entry
                break

        # 代表画像は、ハイライトエントリの画像、なければ最新の画像を使用
        representative_image = None
        if highlight_entry and highlight_entry.get("image") is not None:
            representative_image = highlight_entry["image"]
        else:
            for entry in reversed(self.diary_entries):
                if entry.get("image") is not None:
                    representative_image = entry["image"]
                    break

        # Gemini APIへ渡すプロンプト（集約したログを元に、構造化出力を生成する）
        prompt = f"""
以下は、今日の育児日記のログです:
{aggregated_text}

上記ログをもとに、以下の形式でJSON形式の出力を生成してください。

形式:
{{
    "subject_highlight": "件名に含む今日のハイライトの一言（10字以内）",
    "summary": "今日一日の内容を200字程度でまとめた文章（見出しなし）",
    "highlight": "今日のハイライトとなる一場面を200字程度で説明した文章",
    "notifications": ["リスクや注意事項を箇条書きにした文章（各項目）", ...]
}}

出力は上記の形式に厳密に従ってJSONで返してください。
"""

        # Gemini API の初期設定（APIキーはconfigから）
        genai.configure(api_key=self.config.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Pydanticモデルによる構造化出力のスキーマ定義
        class DigestOutput(BaseModel):
            subject_highlight: str
            summary: str
            highlight: str
            notifications: list[str]

        # Gemini API 呼び出しによる構造化出力の生成
        try:
            response = model.generate_content(
                contents=prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": DigestOutput.__annotations__,
                }
            )
            parsed_output = DigestOutput.parse_raw(response.text)
        except Exception as e:
            logger.error(f"LLM生成の構造化出力でエラーが発生しました: {e}")
            # 失敗時は簡易的な出力でフォールバック
            parsed_output = DigestOutput(
                subject_highlight="ハイライト",
                summary=aggregated_text[:200],
                highlight=aggregated_text[:200],
                notifications=[]
            )

        # メールの件名と本文の組み立て
        subject = f"Diary ({today.strftime('%Y/%m/%d')}) - {parsed_output.subject_highlight} -"
        body = f"{parsed_output.summary}\n\n【今日のハイライト】\n{parsed_output.highlight}\n\n【ご連絡】\n"
        if parsed_output.notifications:
            body += "\n".join(f"- {note}" for note in parsed_output.notifications)
        else:
            body += "特に連絡事項はありません。"

        digest_data = {
            "subject": subject,
            "text": body,
            "representative_image": representative_image
        }
        return digest_data

    def send_daily_digest_gmail(self) -> bool:
        digest_data = self.generate_daily_digest()
        if digest_data is None:
            logger.info("送信するダイジェストがありません。")
            return True
        subject = digest_data["subject"]
        message_text = digest_data["text"]
        image_attachment = digest_data["representative_image"]
        send_success = self.notifier.send_notification(subject, message_text, image_attachment)
        if send_success:
            self.last_digest_sent_date = datetime.date.today()
            logger.info(f"日次ダイジェストメールが {self.last_digest_sent_date} に送信されました。")
            self.diary_entries = []
            self._save_daily_digest_gdrive(digest_data)
        else:
            logger.error("日次ダイジェストメールの送信に失敗しました。")
        return send_success

    def _save_daily_digest_gdrive(self, digest_data: Dict) -> None:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        digest_text_filename_gdrive = f"daily_digest_text_{date_str}.txt"
        digest_image_filename_gdrive = f"daily_digest_image_{date_str}.jpg"
        try:
            digest_text_content = digest_data["text"]
            self.gdrive_service.upload_file(self.diary_dir_gdrive, digest_text_filename_gdrive, digest_text_content.encode('utf-8'), mime_type='text/plain')
            logger.debug(f"Daily digest text saved to Google Drive: {digest_text_filename_gdrive}")
            image_attachment = digest_data["representative_image"]
            if image_attachment is not None:
                image_bytes = cv2.imencode('.jpg', image_attachment)[1].tobytes()
                self.gdrive_service.upload_file(self.diary_dir_gdrive, digest_image_filename_gdrive, image_bytes, mime_type='image/jpeg')
                logger.debug(f"Daily digest image saved to Google Drive: {digest_image_filename_gdrive}")
            else:
                logger.warning("日次ダイジェストに代表画像が見つかりません。")
            sent_flag_filename_gdrive = f"daily_digest_sent_{date_str}.flag"
            self.gdrive_service.upload_file(self.diary_dir_gdrive, sent_flag_filename_gdrive, b"sent", mime_type='text/plain')
            logger.debug(f"Daily digest sent flag saved to Google Drive: {sent_flag_filename_gdrive}")
        except Exception as e:
            logger.error(f"Google Driveへの日次ダイジェストの保存中にエラーが発生しました: {e}")

    def save_extra_content_gdrive(self) -> bool:
        logger.info("Google Driveへ追加の日記内容を保存しています（ダミー） - 未実装です。")
        return True

    def should_send_daily_digest(self) -> bool:
        """
        日次ダイジェストを送信すべきかどうかを判断する。
        以下の条件を満たす場合にTrueを返す：
        1. 設定された送信時刻に達している
        2. 本日まだダイジェストを送信していない
        3. 送信すべき日記エントリが存在する
        """
        now = datetime.datetime.now()
        digest_time_str = self.config.daily_digest_time
        try:
            digest_hour = int(digest_time_str.split(":")[0])
            digest_minute = int(digest_time_str.split(":")[1])
            digest_time = now.replace(hour=digest_hour, minute=digest_minute, second=0, microsecond=0)
        except:
            logger.warning("設定ファイルの日次ダイジェスト時刻の形式が不正です。デフォルト時刻 21:00 を使用します。")
            digest_time = now.replace(hour=21, minute=0, second=0, microsecond=0)

        return (
            now >= digest_time and
            (self.last_digest_sent_date != now.date()) and
            len(self.diary_entries) > 0
        )

    def check_and_send_daily_digest(self) -> None:
        now = datetime.datetime.now()
        digest_time_str = self.config.daily_digest_time
        try:
            digest_hour = int(digest_time_str.split(":")[0])
            digest_minute = int(digest_time_str.split(":")[1])
            digest_time = now.replace(hour=digest_hour, minute=digest_minute, second=0, microsecond=0)
        except:
            logger.warning("設定ファイルの日次ダイジェスト時刻の形式が不正です。デフォルト時刻 21:00 を使用します。")
            digest_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
        if now >= digest_time and (self.last_digest_sent_date != now.date()):
            logger.info("日次ダイジェスト送信時刻に到達しました。ダイジェストメールの送信を開始します...")
            self.send_daily_digest_gmail() 