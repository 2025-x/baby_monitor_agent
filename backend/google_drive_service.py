import logging
import os
from typing import Optional, List, Dict
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
import io

logger = logging.getLogger(__name__)

class GoogleDriveService:
    """
    Google Drive API を操作するサービスモジュール。
    """
    def __init__(self, credentials_path: str):
        self.credentials_path = credentials_path
        self.drive_service = self._initialize_drive_service()

    def _initialize_drive_service(self):
        if os.path.basename(self.credentials_path) == "mock_credentials.json":
            logger.info("Using dummy Google Drive service for testing.")
            return DummyDriveService()
        try:
            creds = Credentials.from_service_account_file(
                self.credentials_path, scopes=['https://www.googleapis.com/auth/drive']
            )
            return build('drive', 'v3', credentials=creds)
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            raise

    def upload_file(self, parent_dir_name: str, file_name: str, file_content: bytes, mime_type: str = 'application/octet-stream') -> None:
        try:
            parent_folder_id = self._get_or_create_directory(parent_dir_name)
            file_metadata = {
                'name': file_name,
                'parents': [parent_folder_id]
            }
            media = MediaIoBaseUpload(io.BytesIO(file_content), mimetype=mime_type, resumable=True)
            file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            logger.info(f"File '{file_name}' uploaded to Google Drive directory '{parent_dir_name}'. File ID: {file.get('id')}")
        except HttpError as error:
            logger.error(f"An error occurred during file upload to Google Drive: {error}")
            raise

    def append_text_file(self, directory_name: str, file_name: str, content: str) -> None:
        """
        指定されたディレクトリ内のテキストファイルに内容を追加します。
        ファイルが存在しない場合は新規作成します。

        Args:
            directory_name (str): ディレクトリ名
            file_name (str): ファイル名
            content (str): 追加するテキスト内容
        """
        try:
            # ディレクトリの存在確認と作成
            directory_id = self._get_or_create_directory(directory_name)
            
            # ファイルの存在確認
            query = f"name='{file_name}' and '{directory_id}' in parents"
            results = self.drive_service.files().list(q=query, spaces='drive').execute()
            files = results.get('files', [])
            
            if files:
                # 既存のファイルに追加
                file_id = files[0]['id']
                # 既存の内容を取得
                existing_content = self._get_file_content(file_id)
                # 新しい内容を追加
                updated_content = existing_content + content
                # ファイルを更新
                self._update_file_content(file_id, updated_content)
            else:
                # 新規ファイル作成
                file_metadata = {
                    'name': file_name,
                    'parents': [directory_id],
                    'mimeType': 'text/plain'
                }
                
                # テキストデータをUTF-8でエンコード
                content_bytes = content.encode('utf-8')
                
                # ファイルを作成
                media = MediaIoBaseUpload(
                    io.BytesIO(content_bytes),
                    mimetype='text/plain',
                    resumable=True
                )
                self.drive_service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                
            logger.info(f"Successfully appended content to {file_name} in {directory_name}")
            
        except Exception as e:
            logger.error(f"Error appending to file in Google Drive: {e}")
            raise

    def _get_file_content(self, file_id: str) -> str:
        """
        Google Driveのファイルの内容を取得します。

        Args:
            file_id (str): ファイルID

        Returns:
            str: ファイルの内容
        """
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            return fh.getvalue().decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading file content from Google Drive: {e}")
            return ""

    def _update_file_content(self, file_id: str, content: str) -> None:
        """
        Google Driveのファイルの内容を更新します。

        Args:
            file_id (str): ファイルID
            content (str): 新しい内容
        """
        try:
            # テキストデータをUTF-8でエンコード
            content_bytes = content.encode('utf-8')
            
            media = MediaIoBaseUpload(
                io.BytesIO(content_bytes),
                mimetype='text/plain',
                resumable=True
            )
            self.drive_service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
        except Exception as e:
            logger.error(f"Error updating file content in Google Drive: {e}")
            raise

    def create_directory(self, dir_name: str) -> str:
        try:
            parent_folder_id = 'root'
            folder_metadata = {
                'name': dir_name,
                'parents': [parent_folder_id],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            file = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
            folder_id = file.get('id')
            logger.info(f"Directory '{dir_name}' created in Google Drive. Folder ID: {folder_id}")
            return folder_id
        except HttpError as error:
            logger.error(f"An error occurred during directory creation in Google Drive: {error}")
            raise

    def directory_exists(self, directory_name: str) -> bool:
        try:
            query = f"name='{directory_name}' and mimeType='application/vnd.google-apps.folder'"
            results = self.drive_service.files().list(q=query, spaces='drive').execute()
            return len(results.get('files', [])) > 0
        except Exception as e:
            logger.error(f"Error checking directory existence: {e}")
            return False

    def list_files(self, directory_name: str) -> List[Dict]:
        """
        指定されたディレクトリ内のファイル一覧を取得します。

        Args:
            directory_name (str): ファイル一覧を取得するディレクトリ名

        Returns:
            List[Dict]: ファイル情報のリスト。各ファイルの情報は辞書形式で、
                      'name'（ファイル名）、'id'（ファイルID）、'mimeType'（MIMEタイプ）、
                      'modifiedTime'（最終更新日時）を含む
        """
        try:
            # ディレクトリIDを取得
            query = f"name='{directory_name}' and mimeType='application/vnd.google-apps.folder'"
            results = self.drive_service.files().list(q=query, spaces='drive').execute()
            folders = results.get('files', [])
            
            if not folders:
                logger.warning(f"Directory '{directory_name}' not found")
                return []
            
            directory_id = folders[0]['id']
            
            # ディレクトリ内のファイル一覧を取得
            query = f"'{directory_id}' in parents"
            results = self.drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType, modifiedTime)'
            ).execute()
            
            return results.get('files', [])
            
        except Exception as e:
            logger.error(f"Error listing files in directory: {e}")
            return []

    def _get_or_create_directory(self, dir_name: str) -> str:
        if self.directory_exists(dir_name):
            return self._get_directory_id(dir_name)
        else:
            return self.create_directory(dir_name)

    def _get_directory_id(self, dir_name: str) -> str:
        try:
            parent_folder_id = 'root'
            query = f"mimeType='application/vnd.google-apps.folder' and name='{dir_name}' and '{parent_folder_id}' in parents and trashed=false"
            results = self.drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
            items = results.get('files', [])
            if items:
                return items[0]['id']
            else:
                return None
        except HttpError as error:
            logger.error(f"An error occurred during directory ID retrieval from Google Drive: {error}")
            return None

    def _get_file_id(self, parent_folder_id: str, file_name: str) -> Optional[str]:
        try:
            query = f"name='{file_name}' and '{parent_folder_id}' in parents and trashed=false"
            results = self.drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
            items = results.get('files', [])
            if items:
                return items[0]['id']
            else:
                return None
        except HttpError as error:
            logger.error(f"An error occurred during file ID retrieval from Google Drive: {error}")
            return None

    def read_file(self, file_id: str) -> str:
        """
        Google Driveのファイルの内容を読み込みます。

        Args:
            file_id (str): 読み込むファイルのID

        Returns:
            str: ファイルの内容
        """
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            return fh.getvalue().decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error reading file from Google Drive: {e}")
            return ""

# Dummy classes for testing without real Google Drive API
class DummyExecutable:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result

class DummyFilesService:
    def create(self, *args, **kwargs):
        return DummyExecutable({'id': 'dummy_created_id'})

    def update(self, *args, **kwargs):
        return DummyExecutable({'id': 'dummy_updated_id'})

    def list(self, *args, **kwargs):
        return DummyExecutable({'files': []})

    def get_media(self, *args, **kwargs):
        return DummyExecutable(b"dummy content")

class DummyDriveService:
    def files(self):
        return DummyFilesService() 