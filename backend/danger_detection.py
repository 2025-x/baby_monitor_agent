import logging

logger = logging.getLogger(__name__)

class DangerDetector:
    """
    危険検出を行うモジュール。
    詳細解析結果に基づいて、赤ちゃんの状況が正常か危険かを評価する。
    """
    def __init__(self, danger_config):
        """
        初期化処理。

        Args:
            danger_config (dict): 危険検出に使用する設定。例: {"stuck_time_threshold": 秒数, "min_confidence_score": float}
        """
        self.config = danger_config

    def evaluate_risk(self, analysis_result: dict) -> dict:
        """
        詳細解析結果に基づいてリスク評価を行う。

        Args:
            analysis_result (dict): DetailAnalyzerからの解析結果。

        Returns:
            dict: リスク評価結果。例:
                  {
                      "risk_score": 0.8,
                      "message": "赤ちゃんに危険が発生している可能性があります。",
                      "reason": "detected_danger"
                  }
        """
        if analysis_result is None:
            return {
                "risk_score": 0.0,
                "message": "解析結果が取得できませんでした。",
                "reason": "analysis_failed"
            }

        # 新しい形式（LangGraph並列ブランチ）の解析結果を処理
        analyses = analysis_result.get("analyses", [])
        if not analyses:
            return {
                "risk_score": 0.0,
                "message": "解析結果が空でした。",
                "reason": "no_analysis"
            }

        # 全ブランチの結果を統合して評価
        max_risk_score = 0.0
        danger_messages = []
        danger_reasons = []

        for analysis in analyses:
            confidence = analysis.get("confidence", 0.5)
            status = analysis.get("status", "unknown")
            posture = analysis.get("posture", "unknown")
            action = analysis.get("action", "unknown")
            perspective = analysis.get("perspective", "unknown")

            risk_score = 0.0
            if status == "danger":
                risk_score = confidence
                danger_messages.append(f"{perspective}の観点で危険を検出")
                danger_reasons.append("detected_danger")
            elif posture == "prone" and action == "idle":
                risk_score = 0.6
                danger_messages.append(f"{perspective}の観点で長時間の同一姿勢を検出")
                danger_reasons.append("prolonged_prone_idle")

            max_risk_score = max(max_risk_score, risk_score)

        if danger_messages:
            return {
                "risk_score": max_risk_score,
                "message": "、".join(danger_messages),
                "reason": "+".join(danger_reasons)
            }
        else:
            return {
                "risk_score": 0.0,
                "message": "正常な状態です。",
                "reason": "normal"
            }

    def should_notify(self, risk_evaluation: dict) -> bool:
        """
        リスク評価に基づいて通知が必要かどうか判断する。

        Args:
            risk_evaluation (dict): evaluate_riskからの結果。

        Returns:
            bool: 通知が必要ならTrue、そうでなければFalse。
        """
        threshold = self.config.get("min_confidence_score", 0.7)
        risk_score = risk_evaluation.get("risk_score", 0.0)
        notify = risk_score >= threshold
        logger.debug(f"Should notify: risk_score={risk_score} vs threshold={threshold} => {notify}")
        return notify 