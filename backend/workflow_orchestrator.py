import time
import logging
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Annotated, TypedDict, List, Union

from backend.camera_capture import CameraCapture
from backend.motion_detection import MotionDetector
from backend.detail_analysis import DetailAnalyzer
from backend.danger_detection import DangerDetector
from backend.logger import EventLogger
from backend.notifier import Notifier
from backend.diary import Diary
from backend.config import config

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolExecutor
from langgraph.channels import LastValue

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """
    LangGraph の StateGraph で状態管理するためのクラス。
    各モジュールの出力や、ワークフロー全体の状態を保持する。
    """
    frame: Annotated[Optional[np.ndarray], "LastValue[frame]"]  # フレームデータ
    motion_detected: Annotated[bool, "LastValue[motion_detected]"]  # モーション検知結果
    analysis_result: Annotated[Optional[Dict], "LastValue[analysis_result]"]  # 詳細解析結果
    risk_evaluation: Annotated[Optional[Dict], "LastValue[risk_evaluation]"]  # 危険評価結果
    should_notify: Annotated[bool, "LastValue[should_notify]"]  # 通知要否
    event_data: Annotated[Dict, "LastValue[event_data]"]  # イベントデータ
    workflow_status: Annotated[str, "LastValue[workflow_status]"]  # ワークフローの状態


class WorkflowOrchestrator:
    """
    ベビーモニターのワークフローを制御するクラス。
    LangGraph を使用して、各モジュール間の状態遷移を管理する。
    """
    def __init__(self, config):
        """
        初期化処理。
        各モジュールのインスタンスを生成し、LangGraph ワークフローを構築する。
        """
        self.config = config
        self.logger = logger
        self.camera_capture = CameraCapture(config.camera_rtsp_url, config.frame_rate)
        self.motion_detector = MotionDetector(config.motion_threshold)
        self.detail_analyzer = DetailAnalyzer()
        self.danger_detector = DangerDetector(config.danger_config)
        self.event_logger = EventLogger(config.log_file, config.log_dir_gdrive)
        self.notifier = Notifier(config.gmail_config)
        self.diary = Diary(config, self.notifier)
        self._workflow_status = "idle"

        # LangGraph ワークフローを構築・コンパイル
        self.workflow = self.create_baby_monitor_workflow()
        logger.info("WorkflowOrchestrator initialized successfully")

    @property
    def workflow_status(self) -> str:
        return self._workflow_status

    @workflow_status.setter
    def workflow_status(self, value: str):
        self._workflow_status = value

    def get_frame_node(self, state: dict) -> dict:
        """
        カメラからフレームを取得するノード関数 (LangGraph)。
        """
        logger.info("Start: フレーム取得")

        try:
            if not self.camera_capture.is_active():
                self.camera_capture.start_capture()
                if not self.camera_capture.is_active():
                    state["workflow_status"] = "error"
                    state["event_data"] = {
                        "event_type": "camera_error",
                        "details": "Failed to start camera capture.",
                        "log_level": "ERROR"
                    }
                    return state
        except Exception as e:
            state["workflow_status"] = "error"
            state["event_data"] = {
                "event_type": "camera_error",
                "details": str(e),
                "log_level": "ERROR"
            }
            return state

        current_frame = self.camera_capture.get_frame()
        if current_frame is None:
            logger.warning("No frame received from camera.")
            state["workflow_status"] = "monitoring"
            state["frame"] = None
            state["event_data"] = {
                "event_type": "frame_error",
                "details": "No frame received from camera.",
                "log_level": "WARNING"
            }
            return state
        else:
            logger.debug("Frame acquired successfully.")
            state["workflow_status"] = "monitoring"
            state["frame"] = current_frame
            return state

    def motion_detection_node(self, state: dict) -> dict:
        """
        モーション検知を実行するノード関数 (LangGraph)。
        """
        logger.info("Start: モーション検知")
        frame = state["frame"]
        
        if frame is None:
            logger.warning("No frame to detect motion. Skipping motion detection.")
            state["motion_detected"] = False
            state["workflow_status"] = "error"
            state["event_data"] = {
                "event_type": "frame_error",
                "details": "No frame available for motion detection.",
                "log_level": "ERROR"
            }
            return state

        try:
            # 既に motion_detected が True の場合は、その状態を維持
            if state.get("motion_detected", False):
                state["workflow_status"] = "analyzing"
                logger.info("Motion already detected: Transitioning to analyzing state")
                return state

            motion_detected = self.motion_detector.detect_motion(frame)
            logger.debug(f"Motion detected: {motion_detected}")
            state["motion_detected"] = motion_detected
            
            if motion_detected:
                state["workflow_status"] = "analyzing"
                logger.info("Motion detected: Transitioning to analyzing state")
            else:
                state["workflow_status"] = "monitoring"
                logger.info("No motion detected: Continuing monitoring")
            
            return state
        except Exception as e:
            logger.error(f"Error in motion detection: {str(e)}")
            state["motion_detected"] = False
            state["workflow_status"] = "error"
            state["event_data"] = {
                "event_type": "motion_detection_error",
                "details": str(e),
                "log_level": "ERROR"
            }
            return state

    def detail_analysis_node(self, state: dict) -> dict:
        """詳細分析ノード: フレームの詳細な解析を行う"""
        try:
            frame = state.get("frame")
            if frame is None:
                state["workflow_status"] = "error"
                state["event_data"] = {
                    "event_type": "api_error",
                    "details": "No frame available for analysis",
                    "log_level": "ERROR"
                }
                self.event_logger.log_event(state["event_data"])
                return state

            analysis_result = self.detail_analyzer.analyze_frame(frame)
            if analysis_result is None:
                state["workflow_status"] = "error"
                state["event_data"] = {
                    "event_type": "api_error",
                    "details": "Analysis failed",
                    "log_level": "ERROR"
                }
                self.event_logger.log_event(state["event_data"])
                return state

            state["analysis_result"] = analysis_result
            state["workflow_status"] = "detecting_danger"
            return state

        except Exception as e:
            state["workflow_status"] = "error"
            state["event_data"] = {
                "event_type": "api_error",
                "details": str(e),
                "log_level": "ERROR"
            }
            self.event_logger.log_event(state["event_data"])
            return state

    def danger_detection_node(self, state: dict) -> dict:
        """
        危険検知を行うノード関数 (LangGraph)。
        """
        logger.info("Start: 危険検知")
        analysis_result = state["analysis_result"]

        risk_evaluation = self.danger_detector.evaluate_risk(analysis_result)
        should_notify = self.danger_detector.should_notify(risk_evaluation)

        logger.info(f"Risk evaluated. Danger: {should_notify}, Reason: {risk_evaluation.get('reason')}")
        self.event_logger.log_event({
            "event_type": "danger_detection",
            "details": f"Danger: {should_notify}, Reason: {risk_evaluation.get('reason')}",
            "log_level": "INFO"
        })

        state["risk_evaluation"] = risk_evaluation
        state["should_notify"] = should_notify
        state["workflow_status"] = "deciding_notification"
        return state

    def notify_node(self, state: dict) -> dict:
        """
        危険通知を送信するノード関数 (LangGraph)。
        """
        logger.info("Start: 危険通知")
        frame = state["frame"]
        risk_evaluation = state["risk_evaluation"]

        if risk_evaluation is None:
            logger.error("No risk evaluation available for notification")
            state["workflow_status"] = "error"
            state["event_data"] = {
                "event_type": "notification_error",
                "details": "No risk evaluation available",
                "log_level": "ERROR"
            }
            return state

        reason = risk_evaluation.get("reason", "unknown_danger")
        message = risk_evaluation.get("message", "Unknown danger detected!")
        subject = f"【危険】赤ちゃんに異常が発生しました ({reason})"
        notification_message = f"【緊急通知】\n\n{message}\n\n状況をすぐに確認してください。"

        # 通知の再試行を実装
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                notification_sent = self.notifier.send_notification(subject, notification_message, frame)
                if notification_sent:
                    logger.info("Notification sent successfully")
                    state["workflow_status"] = "recording_diary"
                    state["event_data"] = risk_evaluation
                    return state
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Notification failed, retrying ({retry_count}/{max_retries})")
                    time.sleep(2)  # 2秒待機してから再試行
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Notification error, retrying ({retry_count}/{max_retries})")
                    time.sleep(2)

        logger.error("Failed to send notification after all retries")
        state["workflow_status"] = "error"
        state["event_data"] = {
            "event_type": "notification_error",
            "details": f"Failed to send notification for: {reason} after {max_retries} retries",
            "log_level": "ERROR"
        }
        return state

    def record_diary_node(self, state: dict) -> dict:
        """日記記録ノード: イベントを日記に記録する"""
        try:
            event_data = {
                "timestamp": time.time(),
                "frame": state.get("frame"),
                "analysis_result": state.get("analysis_result"),
                "risk_evaluation": state.get("risk_evaluation"),
                "notification_sent": state.get("should_notify", False)
            }
            self.diary.record_event(event_data)
            state["workflow_status"] = "checking_digest"
            return state
        except Exception as e:
            state["workflow_status"] = "error"
            state["event_data"] = {
                "event_type": "diary_error",
                "details": str(e),
                "log_level": "ERROR"
            }
            return state

    def check_daily_digest_node(self, state: dict) -> dict:
        """
        日次サマリーの送信要否を確認するノード関数 (LangGraph)。
        """
        logger.info("Start: 日次サマリー確認")
        try:
            if self.diary.should_send_daily_digest():
                logger.info("Sending daily digest...")
                self.diary.send_daily_digest_gmail()
            state["workflow_status"] = "monitoring"
            return state
        except Exception as e:
            logger.error(f"Error checking daily digest: {e}")
            state["workflow_status"] = "error"
            state["event_data"] = {
                "event_type": "digest_error",
                "details": str(e),
                "log_level": "ERROR"
            }
            return state

    def on_error_node(self, state: dict) -> dict:
        """
        エラー発生時の処理を行うノード関数 (LangGraph)。
        """
        logger.info("Start: エラー処理")
        event_data = state.get("event_data", {})
        event_type = event_data.get("event_type", "unknown_error")
        details = event_data.get("details", "Unknown error occurred")

        logger.error(f"Error occurred: {event_type} - {details}")
        self.event_logger.log_event({
            "event_type": event_type,
            "details": details,
            "log_level": "ERROR"
        })

        # エラー状態を記録
        state["workflow_status"] = "error"
        return state

    def motion_detection_router(self, state: dict) -> str:
        """モーション検知の結果に基づいて次のノードを決定"""
        if state.get("workflow_status") == "analyzing":
            return "detail_analysis"
        return "monitoring"

    def danger_detection_router(self, state: dict) -> str:
        """危険検知の結果に基づいて次のノードを決定"""
        if state.get("should_notify", False):
            return "notify"
        return "record_diary"

    def error_router(self, state: dict) -> str:
        """エラー状態からの復帰を決定"""
        if state.get("workflow_status") == "error":
            time.sleep(5)  # エラー状態での待機
            return "error"
        return "monitoring"

    def create_baby_monitor_workflow(self) -> StateGraph:
        """
        ベビーモニターのワークフローを構築する
        """
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(AgentState)

        # ノードの追加
        workflow.add_node("get_frame", self.get_frame_node)
        workflow.add_node("motion_detection", self.motion_detection_node)
        workflow.add_node("detail_analysis", self.detail_analysis_node)
        workflow.add_node("danger_detection", self.danger_detection_node)
        workflow.add_node("notify", self.notify_node)
        workflow.add_node("record_diary", self.record_diary_node)
        workflow.add_node("check_digest", self.check_daily_digest_node)
        workflow.add_node("error", self.on_error_node)

        # エッジの追加
        workflow.set_entry_point("get_frame")
        
        workflow.add_conditional_edges(
            "get_frame",
            self.error_router,
            {
                "monitoring": "motion_detection",
                "error": "error"
            }
        )
        workflow.add_conditional_edges(
            "motion_detection",
            self.motion_detection_router,
            {
                "monitoring": "get_frame",
                "detail_analysis": "detail_analysis"
            }
        )
        workflow.add_conditional_edges(
            "detail_analysis",
            self.danger_detection_router,
            {
                "notify": "notify",
                "record_diary": "record_diary"
            }
        )
        workflow.add_edge("notify", "record_diary")
        workflow.add_edge("record_diary", "check_digest")
        workflow.add_edge("check_digest", "get_frame")
        workflow.add_conditional_edges(
            "error",
            self.error_router,
            {
                "monitoring": "get_frame",
                "error": END
            }
        )

        return workflow.compile()

    def run(self):
        """ワークフローを実行する."""
        try:
            self.logger.info("Starting workflow execution...")
            self.workflow_status = "monitoring"
            
            # ワークフローの実行
            workflow = self.create_baby_monitor_workflow()
            config = {"recursion_limit": 100}  # 再帰制限を増やす
            
            # テスト用の実行制限を追加
            max_iterations = 10
            iteration = 0
            
            for state in workflow.stream({"workflow_status": "monitoring"}, config=config):
                self.workflow_status = state.get("workflow_status", "monitoring")
                self.logger.debug(f"Current workflow status: {self.workflow_status}")
                
                # テスト用の実行制限をチェック
                iteration += 1
                if iteration >= max_iterations:
                    self.logger.info("Reached maximum iterations for testing")
                    break
                
        except Exception as e:
            self.logger.error(f"Error in workflow execution: {str(e)}")
            self.workflow_status = "error"
        finally:
            if hasattr(self.camera_capture, "release"):
                self.camera_capture.release()


def main():
    """
    メイン関数 (ワークフローの起動)。
    """
    logging.basicConfig(level=logging.INFO)
    orchestrator = WorkflowOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
