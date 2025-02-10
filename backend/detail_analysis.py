import logging
import numpy as np
from typing import Optional, Dict, List
import google.generativeai as genai
import time
import cv2
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from PIL import Image

from backend.config import config

# LangGraph のインポート
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

class DetailAnalyzer:
    """
    フレーム画像の詳細解析を行うモジュール (Gemini API連携)。
    複数の Gemini 呼び出しを異なる観点で実施し、結果を集約する仕組みを LangGraph で実現。
    
    ２つの並列ブランチで評価を行います。
      ブランチ１：
         - 赤ちゃん自身の感情・行動
         - 赤ちゃんの安全性
      ブランチ２：
         - 赤ちゃんの周りで発生しているイベント
         - 赤ちゃんの安全性
         
    各ブランチの結果には、どのブランチで実施したかの情報も含めます。
    今後、新たな LLM を追加する場合は、branches 内の構成を追加するだけで対応可能です。
    """
    def __init__(self):
        """
        初期化処理。Gemini API クライアント初期化 (gemini-1.5-flash)。
        """
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.gemini_max_retry = config.gemini_max_retry
        self.gemini_retry_wait = config.gemini_retry_wait
        self.gemini_timeout = config.gemini_timeout
        self.image_max_size = config.image_max_size
        self.image_quality = config.image_quality

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        フレーム画像を前処理して最適化する。
        """
        # サイズの最適化
        height, width = frame.shape[:2]
        if max(height, width) > self.image_max_size:
            scale = self.image_max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        # JPEG品質の最適化
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return frame

    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        フレーム画像を複数の Gemini へ渡して解析し、２つの並列ブランチの結果を集約する。
        入出力のインターフェースは従来と同じ。
        """
        if not self.is_valid_frame(frame):
            logger.warning("Invalid frame. Skipping analysis.")
            return None

        # 画像の前処理
        frame = self.preprocess_frame(frame)

        # 各ブランチのタスク定義（今後追加や変更を行いやすい設計）
        branches = [
            {
                "branch": "branch1",
                "tasks": [
                    {
                        "perspective": "emotion_behavior",
                        "prompt": "赤ちゃんの感情と行動を分析: 1. 表情（喜び/悲しみ）2. 体の動き 3. 異常な行動の有無"
                    },
                    {
                        "perspective": "safety",
                        "prompt": "赤ちゃんの安全性を評価: 1. 姿勢（仰向け/うつ伏せ）2. 呼吸の様子 3. 危険な状態の有無"
                    }
                ]
            },
            {
                "branch": "branch2",
                "tasks": [
                    {
                        "perspective": "environment_event",
                        "prompt": "環境の安全性を評価: 1. 周囲の物の状態 2. 異常な動きの有無 3. 危険要素の特定"
                    },
                    {
                        "perspective": "safety",
                        "prompt": "総合的な安全性を評価: 1. 危険度（低/中/高）2. 具体的なリスク 3. 緊急性の判断"
                    }
                ]
            }
        ]

        # 各ブランチごとに LangGraph ワークフローを構築・実行し、結果を集約する
        branch_results: List[Dict] = []

        # 各ブランチ毎の処理関数を定義
        def call_gemini_with_task(task, frame):
            """
            指定タスクで Gemini API を呼び出し、リトライ付きで結果に観点情報を付与して返す。
            """
            retry_count = 0
            prompt = task["prompt"]
            while retry_count <= self.gemini_max_retry:
                try:
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            self._call_gemini_api,
                            prompt=prompt,
                            frame=frame
                        )
                        try:
                            response = future.result(timeout=self.gemini_timeout)
                            if response and response.text:
                                result = self._parse_gemini_response(response.text)
                                result["raw_response"] = response.text
                                result["perspective"] = task["perspective"]
                                return result
                        except TimeoutError:
                            logger.warning(
                                f"Gemini API timeout for perspective '{task['perspective']}'. Retry {retry_count+1}/{self.gemini_max_retry}"
                            )
                except Exception as e:
                    logger.error(f"Gemini analyze_frame error for perspective '{task['perspective']}': {e}")
                retry_count += 1
                if retry_count <= self.gemini_max_retry:
                    logger.info(
                        f"Retrying Gemini analysis for perspective '{task['perspective']}'. Attempt {retry_count}/{self.gemini_max_retry}"
                    )
                    time.sleep(self.gemini_retry_wait)
                else:
                    logger.error(
                        f"Max retry reached for perspective '{task['perspective']}' after error. Returning default result."
                    )
                    return {
                        "perspective": task["perspective"],
                        "status": "unknown",
                        "posture": "unknown",
                        "action": "unknown",
                        "confidence": 0.5,
                        "raw_response": ""
                    }
            return None

        def process_task(state_dict: dict) -> dict:
            """
            LangGraph ノード：タスク一覧から１件取り出して Gemini の呼び出しを行い、その結果を追加する。
            """
            tasks = state_dict.get("tasks", [])
            if tasks:
                current_task = tasks.pop(0)
                result = call_gemini_with_task(current_task, state_dict["frame"])
                state_dict["results"].append(result)
            return state_dict

        def aggregate_results(state_dict: dict) -> dict:
            """
            LangGraph ノード：ブランチ内で各タスクの結果を集約する。
            """
            aggregated = {"branch_analyses": state_dict.get("results", [])}
            return aggregated

        def decide_next(state_dict: dict) -> str:
            """
            LangGraph 条件関数：タスクが残っている場合は "process"、なければ "aggregate" へ分岐。
            """
            if state_dict.get("tasks"):
                return "process"
            return "aggregate"

        # 同一処理を各ブランチで実行
        for branch in branches:
            # 各ブランチの初期状態
            state = {
                "frame": frame,
                "tasks": branch["tasks"].copy(),
                "results": []
            }
            # LangGraph ワークフロー構築
            graph_builder = StateGraph(dict)
            graph_builder.add_node("process", process_task)
            graph_builder.add_node("aggregate", aggregate_results)
            graph_builder.add_edge(START, "process")
            graph_builder.add_conditional_edges(
                "process",
                decide_next,
                {
                    "process": "process",
                    "aggregate": "aggregate"
                }
            )
            graph_builder.add_edge("aggregate", END)
            compiled_graph = graph_builder.compile()
            branch_state = compiled_graph.invoke(state)
            
            # ブランチ毎の結果に branch 名を付与
            analyses = branch_state.get("branch_analyses", [])
            for analysis in analyses:
                analysis["branch"] = branch["branch"]
            branch_results.extend(analyses)

        # 全ブランチ分の解析結果を集約（キーは "analyses" のリスト）
        final_result = {"analyses": branch_results}
        logger.info("Frame analysis completed with two parallel branch evaluations (LangGraph).")
        return final_result

    def is_valid_frame(self, frame: np.ndarray) -> bool:
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if frame.shape[0] < 100 or frame.shape[1] < 100:
            return False
        return True

    def _call_gemini_api(self, prompt: str, frame: np.ndarray):
        """
        Gemini APIを呼び出す内部メソッド（base64エンコード使用）
        """
        try:
            # フレームをJPEGにエンコード（画質設定を適用）
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            # base64エンコード
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            prompt_contents = [
                prompt,
                {
                    "mime_type": "image/jpeg",
                    "data": base64_image
                }
            ]
            response = self.model.generate_content(prompt_contents)
            response.resolve()
            return response
        except Exception as e:
            logger.error(f"Error in _call_gemini_api: {e}")
            return None

    def _parse_gemini_response(self, text: str) -> Dict:
        """
        受信した Gemini のテキストレスポンスをパースして、基本的な解析結果を返す。
        ※解析ロジックは従来実装と互換性のある形で実装しています。
        """
        analysis = {
            "status": "unknown",
            "posture": "unknown",
            "action": "unknown",
            "confidence": 0.5
        }
        lower_text = text.lower()
        if "supine" in lower_text or "lying on back" in lower_text:
            analysis["posture"] = "supine"
            analysis["confidence"] = 0.8
        elif "prone" in lower_text or "face down" in lower_text:
            analysis["posture"] = "prone"
            analysis["confidence"] = 0.8
        if "moving" in lower_text or "active" in lower_text:
            analysis["action"] = "moving_limbs"
        elif "still" in lower_text or "idle" in lower_text:
            analysis["action"] = "idle"
        if "danger" in lower_text or "risk" in lower_text:
            analysis["status"] = "danger"
            analysis["confidence"] = max(analysis["confidence"], 0.7)
        elif "safe" in lower_text or "normal" in lower_text:
            analysis["status"] = "normal"
            analysis["confidence"] = max(analysis["confidence"], 0.9)
        logger.debug(f"Parsed gemini text => {analysis}")
        return analysis 