import logging
from typing import Dict, Optional, TypedDict, Annotated
import numpy as np
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, START

logger = logging.getLogger(__name__)

class NotifierState(TypedDict):
    """
    通知処理の状態を管理するためのクラス (LangGraph)
    """
    subject: str
    message: str
    attachment: Optional[np.ndarray]
    tool_calls: list
    notification_sent: bool
    retry_count: int
    error_message: Optional[str]

class Notifier:
    """
    危険通知を送信するモジュール (LangGraph + Function Calling + Gemini 1.5 Flash)
    """
    def __init__(self, smtp_config: Dict):
        self.smtp_config = smtp_config
        self.llm_agent = ChatGoogleGenerativeAI(model="gemini-pro")
        self.tools = [
            Tool.from_function(
                func=self._send_email_tool,
                name="send_email_notification",
                description="保護者に危険通知メールを送信する",
            ),
        ]
        self.llm_agent = self.llm_agent.bind_tools(self.tools)
        self.workflow = self._create_notification_workflow()

    def _create_notification_workflow(self) -> StateGraph:
        """通知処理のワークフローを構築（内部利用用）"""
        workflow = StateGraph(NotifierState)

        # ノード定義（本来は各ノードが連結される形ですが、
        # send_notification() 内では順次ノード関数を直接呼び出すため、ここでは参考情報として残します）
        workflow.add_node("prepare_notification", self._prepare_notification_node)
        workflow.add_node("function_calling", self._function_calling_node)
        workflow.add_node("send_notification", self._send_notification_node)
        workflow.add_node("handle_error", self._handle_error_node)

        workflow.add_edge(START, "prepare_notification")
        workflow.add_edge("prepare_notification", "function_calling")
        workflow.add_conditional_edges(
            "function_calling",
            self._function_calling_router,
            {
                "send_notification": "send_notification",
                "handle_error": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "send_notification",
            self._send_notification_router,
            {
                "end": END,
                "retry": "function_calling",
                "error": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "handle_error",
            self._error_router,
            {
                "retry": "function_calling",
                "end": END
            }
        )

        workflow = workflow.compile()
        logger.debug("Workflow compiled successfully")
        return workflow

    def _prepare_notification_node(self, state: NotifierState) -> NotifierState:
        """通知の準備を行うノード"""
        logger.info("Preparing notification...")
        return {
            **state,
            "tool_calls": [],
            "notification_sent": False,
            "retry_count": 0,
            "error_message": None
        }

    def _function_calling_node(self, state: NotifierState) -> NotifierState:
        """Function Callingを実行するノード"""
        logger.info("Executing Function Calling...")
        # 前回エラーが残っている場合はリセット
        state["error_message"] = None
        content = f"件名: {state['subject']}\nメッセージ: {state['message']}"
        if state.get("attachment") is not None:
            content += "\n画像ファイルを添付"
        human_msg = HumanMessage(content=content)
        logger.debug(f"Sending message to LLM: {content}")

        try:
            response: AIMessage = self.llm_agent.invoke([human_msg])
            logger.debug(f"LLM response: {response}")

            if not hasattr(response, 'additional_kwargs'):
                logger.warning("No additional_kwargs in response")
                return {**state, "tool_calls": [], "error_message": "No additional_kwargs in response"}

            if 'tool_calls' not in response.additional_kwargs:
                logger.warning("No tool_calls in response")
                return {**state, "tool_calls": [], "error_message": "No tool_calls in response"}

            tool_calls = response.additional_kwargs['tool_calls']
            if not tool_calls:
                logger.warning("Empty tool_calls in response")
                return {**state, "tool_calls": [], "error_message": "Empty tool_calls in response"}

            # ツール呼び出しの形式を検証
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    logger.warning("Invalid tool call format")
                    continue
                if 'function' not in tool_call:
                    logger.warning("No function in tool call")
                    continue
                if 'name' not in tool_call['function']:
                    logger.warning("No name in function")
                    continue
                if tool_call['function']['name'] != 'send_email_notification':
                    logger.warning(f"Unexpected function name: {tool_call['function']['name']}")
                    continue

                logger.info("Function calling successful")
                logger.debug(f"Tool calls: {tool_calls}")
                return {**state, "tool_calls": tool_calls}

            logger.warning("No valid tool calls found")
            return {**state, "tool_calls": [], "error_message": "No valid tool calls found"}

        except Exception as e:
            logger.error(f"Error in function calling: {e}")
            return {**state, "tool_calls": [], "error_message": str(e)}

    def _send_notification_node(self, state: NotifierState) -> NotifierState:
        """通知を送信するノード"""
        logger.info("Sending notification...")
        if not state["tool_calls"]:
            logger.error("No tool calls available")
            return {**state, "notification_sent": False, "error_message": "No tool calls available"}

        try:
            tool_call = state["tool_calls"][0]
            if not isinstance(tool_call, dict) or "function" not in tool_call:
                logger.error("Invalid tool call format")
                return {**state, "notification_sent": False, "error_message": "Invalid tool call format"}

            function_data = tool_call["function"]
            if not isinstance(function_data, dict) or "name" not in function_data:
                logger.error("Invalid function data format")
                return {**state, "notification_sent": False, "error_message": "Invalid function data format"}

            if function_data["name"] != "send_email_notification":
                logger.error("Invalid tool call name")
                return {**state, "notification_sent": False, "error_message": "Invalid tool call name"}

            # 添付ファイルの処理
            attachment = state.get("attachment")
            if attachment is not None:
                if not isinstance(attachment, np.ndarray):
                    logger.warning(f"Unsupported attachment type: {type(attachment)}")
                    return {**state, "notification_sent": False, "error_message": "Unsupported attachment type"}
                logger.debug("Using numpy array attachment")

            # _send_email_impl を呼び出して実際のメール送信を実行
            success = self._send_email_impl(
                state["subject"],
                state["message"],
                attachment
            )

            if success:
                logger.info("Notification sent successfully")
                return {**state, "notification_sent": True}
            else:
                logger.error("Failed to send notification")
                return {**state, "notification_sent": False, "error_message": "Failed to send notification"}

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return {**state, "notification_sent": False, "error_message": str(e)}

    def _handle_error_node(self, state: NotifierState) -> NotifierState:
        """エラー処理を行うノード（retry_count を更新し、error_message をクリアする）"""
        logger.error(f"Error in notification workflow: {state['error_message']}")
        return {
            **state,
            "retry_count": state["retry_count"] + 1,
            "tool_calls": [],
            "notification_sent": False,
            "error_message": None
        }

    def _function_calling_router(self, state: NotifierState) -> str:
        """Function Callingの結果に基づいてルーティング"""
        logger.debug(f"Function calling router state: error_message={state.get('error_message')}, tool_calls={bool(state.get('tool_calls'))}")
        
        if state.get("error_message"):
            logger.warning(f"Function calling error: {state['error_message']}")
            return "handle_error"
        
        if state.get("tool_calls"):
            logger.info("Valid tool calls found, proceeding to send notification")
            return "send_notification"
        
        logger.warning("No valid tool calls found, handling as error")
        return "handle_error"

    def _send_notification_router(self, state: NotifierState) -> str:
        """送信結果に基づいてルーティング"""
        logger.debug(f"Send notification router state: notification_sent={state.get('notification_sent')}, retry_count={state.get('retry_count')}, error_message={state.get('error_message')}")
        
        if state.get("notification_sent"):
            logger.info("Notification sent successfully")
            return "end"
        
        if state.get("retry_count", 0) >= 3:
            logger.error("Max retry attempts reached")
            return "error"
        
        logger.info(f"Retrying notification (attempt {state.get('retry_count', 0) + 1})")
        return "retry"

    def _error_router(self, state: NotifierState) -> str:
        """エラー状態に基づいてルーティング"""
        retry_count = state.get("retry_count", 0)
        logger.debug(f"Error router state: retry_count={retry_count}")
        
        if retry_count >= 3:
            logger.error("Max retry attempts reached in error handler")
            return "end"
        
        logger.info(f"Attempting retry from error handler (attempt {retry_count + 1})")
        return "retry"

    def send_notification(self, risk_evaluation: Dict) -> bool:
        """
        危険通知を送信する。
        """
        if risk_evaluation is None:
            logger.error("No risk evaluation available for notification")
            return False

        try:
            # 通知の準備
            notification_state = {
                "risk_evaluation": risk_evaluation,
                "reason": risk_evaluation.get("reason", "Unknown reason"),
                "message": risk_evaluation.get("message", "No details available"),
                "notification_sent": False,
                "retry_count": 0
            }

            # 最大3回まで再試行
            max_retries = 3
            while notification_state["retry_count"] < max_retries:
                try:
                    # 通知を送信
                    subject = f"危険通知: {notification_state['reason']}"
                    notification_message = f"危険を検知しました。\n\n理由: {notification_state['reason']}\n\n詳細: {notification_state['message']}"
                    
                    self._send_email_tool(subject=subject, message=notification_message)
                    notification_state["notification_sent"] = True
                    logger.info("Notification sent successfully")
                    return True

                except Exception as e:
                    notification_state["retry_count"] += 1
                    if notification_state["retry_count"] < max_retries:
                        logger.warning(f"Notification failed, retrying ({notification_state['retry_count']}/{max_retries})")
                        time.sleep(2)  # 2秒待機してから再試行
                    else:
                        logger.error(f"Failed to send notification after {max_retries} attempts: {e}")
                        return False

        except Exception as e:
            logger.error(f"Error in notification process: {e}")
            return False

    def _send_email_tool(self, subject: str, message: str, attachment: Optional[np.ndarray] = None) -> str:
        """メール送信ツール (Function)"""
        attachment_np = None
        if attachment is not None:
            if isinstance(attachment, np.ndarray):
                attachment_np = attachment
            elif isinstance(attachment, str):
                try:
                    attachment_np = np.load(attachment)
                except Exception as e:
                    logger.error(f"添付ファイル読み込みエラー: {e}")
                    return "Error: 添付ファイル読み込み失敗"

        success = self._send_email_impl(subject, message, attachment_np)
        return "メール送信成功" if success else "メール送信失敗"

    def _send_email_impl(self, subject: str, message: str, attachment: Optional[np.ndarray] = None) -> bool:
        """実際のメール送信処理"""
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.image import MIMEImage
        import cv2

        smtp_config = self.smtp_config
        server = None
        try:
            server = smtplib.SMTP(smtp_config["smtp_server"], smtp_config["smtp_port"])
            server.starttls()
            server.login(smtp_config["username"], smtp_config["password"])
            msg = MIMEMultipart()
            msg['From'] = smtp_config["from_email"]
            msg['To'] = smtp_config["to_email"]
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain'))

            if attachment is not None:
                _, img_encoded = cv2.imencode('.jpg', attachment)
                img_mime = MIMEImage(img_encoded.tobytes(), name="baby_status.jpg")
                msg.attach(img_mime)

            server.sendmail(smtp_config["from_email"], smtp_config["to_email"], msg.as_string())
            logger.info(f"Notification email sent successfully to: {smtp_config['to_email']}")
            return True

        except Exception as e:
            logger.error(f"Error sending notification email: {e}")
            return False
        finally:
            if server:
                server.quit()