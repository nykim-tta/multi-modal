"""[공유] 연구용 모델 비교 요약 — 모델 조합별 집계 + (선택) 사람 정답 대비 상관/오차."""

from typing import Optional

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr


def summarize_model_comparison(df: pd.DataFrame, human_scores: Optional[pd.DataFrame] = None):
    group_cols = ["claim_model", "vlm_model", "embedding_model"]
    summary = df.groupby(group_cols, dropna=False).agg(
        mean_semantic_score=("semantic_score", "mean"),
        std_semantic_score=("semantic_score", "std"),
        mean_evaluable_rate=("evaluable_claim_rate", "mean"),
        mean_claim_agreement=("mean_claim_agreement", "mean"),
        mean_claim_count=("claim_count", "mean"),
        mean_claim_seconds=("claim_extraction_seconds", "mean"),
        mean_verify_seconds=("verification_seconds", "mean"),
    ).reset_index()

    if human_scores is not None:
        merged = df.merge(human_scores[["sample_id", "human_score"]], on="sample_id", how="inner")
        metrics = []
        for keys, g in merged.groupby(group_cols):
            valid = g.dropna(subset=["semantic_score", "human_score"])
            if len(valid) < 2:
                continue
            metrics.append({
                **dict(zip(group_cols, keys)),
                "n": len(valid),
                "mae": mean_absolute_error(valid["human_score"], valid["semantic_score"]),
                "rmse": mean_squared_error(valid["human_score"], valid["semantic_score"]) ** 0.5,
                "pearson": pearsonr(valid["human_score"], valid["semantic_score"])[0],
                "spearman": spearmanr(valid["human_score"], valid["semantic_score"])[0],
            })
        metrics_df = pd.DataFrame(metrics)
        summary = summary.merge(metrics_df, on=group_cols, how="left")

    sort_cols = [c for c in ["spearman", "mae", "mean_evaluable_rate"] if c in summary.columns]
    ascending = [False, True, False][:len(sort_cols)]
    return summary.sort_values(sort_cols, ascending=ascending) if sort_cols else summary
