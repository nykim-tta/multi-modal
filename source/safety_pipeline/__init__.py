"""안전신문고 이미지-텍스트 의미 정확성 평가 파이프라인 (모듈 패키지).

두 담당자가 나눠 작업할 수 있도록 단계별로 모듈을 분리했습니다.

[공유]  config, common, records, dataio, analysis
[담당 A: 추출·합의]  extraction, consensus, stage_a
[담당 B: 검증·점수]  verification, scoring, stage_b

전형적 사용 흐름:
    from safety_pipeline import config
    from safety_pipeline.dataio import load_samples_from_data_dir
    from safety_pipeline.stage_a import run_stage_a
    from safety_pipeline.stage_b import run_stage_b
    from safety_pipeline.analysis import summarize_model_comparison

    samples = load_samples_from_data_dir(config.DEFAULT_DATA_DIR)
    claim_sets = run_stage_a(samples)          # A: LLM별 claim 집합 생성·저장
    summary_df, records = run_stage_b(claim_sets)  # B: VLM별 검증·채점
    summarize_model_comparison(summary_df)
"""

from . import config
from . import common
from . import records

__all__ = ["config", "common", "records"]
