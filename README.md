# 赤ちゃん見守りエージェント (Baby Monitor Agent)

## 概要

**赤ちゃん見守りエージェント**は、TP-Link Tapo C200 WebカメラとAIを活用した、赤ちゃん見守りアプリケーションです。

カメラ映像をリアルタイムで解析し、赤ちゃんの安全を遠隔から確認できます。危険な状態を検知すると即座にGmailで通知し、日々の赤ちゃんの様子を日記として記録・Gmail送信します。

**主な機能:**

* **リアルタイム映像モニタリング:** Tapo カメラの映像を常時監視。
* **AI駆動の動作検知:** OpenCVとGoogle Gemini 1.5 Flash APIを活用し、赤ちゃんの動きを検知・詳細解析。
* **危険状態検知とGmail通知:** うつ伏せ、顔覆われ、転落などの危険な状態をAIが検知し、Gmailで即時通知。
* **育児日記の自動記録:** 赤ちゃんの活動をAIが解析し、日時、状況説明とともに日記形式で記録。
* **日次ダイジェストのGmail送信:** 一日の日記をまとめ、代表写真とともに毎日Gmailで送信。
* **日記のGoogle Drive保存:** 日記データ（文章、写真）をGoogle Driveに保存し、長期的な記録を保管。
* **実行ログのGoogle Drive保存:** アプリケーションの実行ログをGoogle Driveに保存し、システム監視・問題解析に活用。
* **バックエンドAPI:** フロントエンド開発や外部連携を見据えたバックエンドAPI設計。

**技術スタック:**

* **Python:**  主要な開発言語
* **OpenCV:**  映像処理、モーション検知
* **Google Gemini 1.5 Flash API:**  詳細な画像解析、状況認識
* **Langchain (将来拡張):**  ワークフロー管理、エージェントフレームワーク
* **Google Drive API:**  ログ、日記データ保存
* **smtplib:**  Gmail送信 (通知、日次ダイジェスト)
* **python-dotenv:**  環境変数管理 (.envファイル)
* **Pytest:**  テストフレームワーク

## インストール

### 前提条件

* **Python 3.8 以上**
* **pip** (Python パッケージ管理ツール)
* **Google Cloud Platform プロジェクト** および **Google Drive API** の有効化
* **Gemini API キー** の取得 ([Google AI Studio](https://aistudio.google.com/app/apikey) で取得)
* **Tapo カメラ** および **RTSP ストリーム URL** の準備

### セットアップ手順

1. **仮想環境を作成:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate  # Windows
   ```

3. **依存パッケージをインストール:**
   ```bash
   pip install -r requirements.txt
   ```

4. **環境変数を設定:**
   - プロジェクトルートディレクトリに `.env` ファイルを作成し、以下の環境変数を設定します。
     ```
     CAMERA_RTSP_URL="rtsp://your_camera_ip/stream"
     CAMERA_FRAME_RATE=5
     MOTION_THRESHOLD=5000
     STUCK_TIME_THRESHOLD=300
     MIN_CONFIDENCE_SCORE=0.7

     SMTP_SERVER="smtp.gmail.com"
     SMTP_PORT=587
     FROM_EMAIL="your_email@gmail.com"
     TO_EMAIL="parent_email@gmail.com"
     SMTP_USERNAME="your_email@gmail.com"
     SMTP_PASSWORD="your_gmail_app_password" # Gmailアプリパスワード推奨

     LOG_FILE="baby_monitor_logs/baby_monitor.log"
     LOG_DIR_GDRIVE="baby_monitor_logs"
     DIARY_DIR_GDRIVE="baby_monitor_diary"
     DAILY_DIGEST_TIME="21:00"

     GEMINI_API_KEY="YOUR_GEMINI_API_KEY" # 取得した Gemini API キー
     GOOGLE_DRIVE_CREDENTIALS_PATH="path/to/your/google_drive_credentials.json" # Google Drive 認証情報ファイルパス
     ```
   - 各項目の設定値を適切に書き換えてください。
   - `GOOGLE_DRIVE_CREDENTIALS_PATH` には、Google Cloud Console で作成したサービスアカウントの認証情報ファイル (JSON) のパスを指定します。

5. **Google Drive 認証情報ファイルを用意:**
   - [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成し、Google Drive API を有効にします。
   - サービスアカウントを作成し、認証情報ファイル (JSON) をダウンロードします。
   - ダウンロードした JSON ファイルを、`.env` ファイルで指定したパス (`GOOGLE_DRIVE_CREDENTIALS_PATH`) に配置します。

## 使用方法

### アプリケーションの起動

backend ディレクトリに移動し、`workflow_orchestrator.py` を実行します。

```bash
cd backend
python workflow_orchestrator.py
```

アプリケーションが起動し、カメラ映像の監視を開始します。

### 設定

* **config.py:** アプリケーションの各種設定 (カメラURL、閾値、APIキーなど) は `backend/config.py` で管理されています。必要に応じて設定値を変更してください。
* **.envファイル:** 環境変数は `.env` ファイルで設定します。APIキー、Gmailアカウント情報、Google Drive 認証情報ファイルパスなどは、`.env` ファイルに記述することを推奨します。

### ログ確認

* **ローカルログ:** `baby_monitor_logs/baby_monitor.log` にアプリケーションの実行ログが出力されます。
* **Google Drive ログ:** Google Drive の `baby_monitor_logs` ディレクトリにもログファイルが保存されます。

### 日記確認

* **Gmail:** 毎日指定時刻 (`DAILY_DIGEST_TIME` 設定) に、育児日記のダイジェストメールが送信されます。
* **Google Drive 日記:** Google Drive の `baby_monitor_diary` ディレクトリに、日記データ (テキスト、写真) が保存されます。


## ディレクトリ構成

```
baby_monitor_agent/
├── backend/                # バックエンドコード
│   ├── __init__.py
│   ├── camera_capture.py     # カメラ映像取得モジュール
│   ├── motion_detection.py   # 動作検出モジュール
│   ├── detail_analysis.py    # 詳細解析モジュール (Gemini API連携)
│   ├── danger_detection.py   # 危険検知モジュール
│   ├── logger.py             # ログ記録モジュール (ローカル、Google Drive)
│   ├── notifier.py           # 通知モジュール (Gmail送信)
│   ├── diary.py              # 育児日記モジュール (ローカル、Google Drive、Gmail送信)
│   ├── workflow_orchestrator.py # ワークフロー制御モジュール
│   ├── config.py             # 設定ファイル
│   └── google_drive_service.py # Google Drive API サービスモジュール
└── requirements.txt         # 依存パッケージリスト除
```

## ライセンス (License)


---
