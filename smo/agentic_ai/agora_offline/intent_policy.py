"""Versioned research recommendations and user-owned goal/threshold policies."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .contracts import CELLS, HORIZONS, ContractError, validate_intent

VERSION = "threshold-v3-app5"
REFERENCE_PATH = Path(__file__).parent / "context" / "threshold_reference.json"
PURPOSES = {
    "energy_saving": "Energy Saving · TxP",
    "throughput": "Throughput Maximization · TxP / RET",
    "load_balancing": "Load Balancing · CIO",
    "handover_control": "이동성 최적화",
    "coverage": "커버리지 개선",
    "custom": "복합 / 사용자 정의",
}

# Each option explicitly defines aggregation. Ratios always pool numerators and
# denominators; they are never arithmetic means of cell ratios.
OPTIONS = [
    (
        "energy",
        "소비 에너지",
        "energy_j",
        "sum",
        "J",
        "decrease",
        "TxP와 ON 상태 전력 수식으로 계산",
    ),
    (
        "goodput",
        "전달 처리량",
        "goodput_mbps",
        "sum",
        "Mbps",
        "increase",
        "선택한 셀의 goodput 합계",
    ),
    (
        "goodput_retention",
        "처리량 보존율",
        "goodput_retention_pct",
        "sum",
        "%",
        "increase",
        "같은 미래 구간의 설정 유지 예측 대비 비율",
    ),
    ("rlf", "RLF 횟수", "rlfs", "sum", "회", "decrease", "평가 구간 누적 횟수; 예측은 실수값"),
    (
        "pp_ratio",
        "Ping-pong / HO 시도",
        "pingpong_per_attempt_pct",
        "sum",
        "%",
        "decrease",
        "성공 HO 대비 비율의 대용 지표; HO 시도 10회 미만은 판정 불가",
    ),
    (
        "prb_max",
        "최대 셀 PRB 이용률",
        "prb_pct",
        "max",
        "%",
        "decrease",
        "셀별 구간 평균 이용률의 최댓값; 순간 최대가 아님",
    ),
    (
        "prb_spread",
        "셀 간 PRB 격차",
        "prb_pct",
        "spread",
        "%p",
        "decrease",
        "구간 평균 PRB의 최댓값 - 최솟값",
    ),
    (
        "prb_jain",
        "PRB 균형 지수 (Jain)",
        "prb_jain",
        "sum",
        "0–1",
        "increase",
        "0 부하에서는 판정 불가; 1에 가까울수록 균등",
    ),
    (
        "ee",
        "네트워크 에너지 효율",
        "network_energy_eff_mbit_per_j",
        "sum",
        "Mbit/J",
        "increase",
        "전체 전달량 / 전체 소비 에너지",
    ),
    (
        "ee_retention",
        "에너지 효율 보존율",
        "energy_eff_retention_pct",
        "sum",
        "%",
        "increase",
        "같은 미래 구간의 설정 유지 대비 에너지 효율",
    ),
    (
        "rsrp",
        "평균 RSRP (가장 낮은 셀)",
        "rsrp_interval_mean_dbm",
        "min",
        "dBm",
        "increase",
        "유효 UE·시간 표본의 dBm 산술평균; P10과 함께 평가",
    ),
    (
        "sinr",
        "평균 SINR (가장 낮은 셀)",
        "sinr_interval_mean_db",
        "min",
        "dB",
        "increase",
        "셀 평균; 백분율 개선 대신 dB 변화량 사용",
    ),
    (
        "delay",
        "평균 지연 (가장 높은 셀)",
        "delay_ms",
        "max",
        "ms",
        "decrease",
        "평균 지연이며 P95 지연이 아님; 서비스 요구값 직접 입력",
    ),
    (
        "loss",
        "손실률 (가장 높은 셀)",
        "loss_pct",
        "max",
        "%",
        "decrease",
        "HO 전후 집계 대상 정합성 미검증; 강제 제약 사용 시 한계 확인 필요",
    ),
    (
        "bler",
        "BLER (가장 높은 셀)",
        "bler_pct",
        "max",
        "%",
        "decrease",
        "서비스 / 재전송 정의에 맞는 기준값 직접 입력",
    ),
    (
        "ho",
        "HO 시도 횟수",
        "ho_attempts",
        "sum",
        "회",
        "decrease",
        "HO 감소만으로 이동성 개선을 의미하지 않음",
    ),
    ("pingpongs", "Ping-pong 횟수", "pingpongs", "sum", "회", "decrease", "평가 구간 누적 횟수"),
    (
        "power",
        "소비 전력",
        "power_w",
        "sum",
        "W",
        "decrease",
        "ON 상태 수식: 130 + 4.7 * 10^((TxP-30)/10) W/cell",
    ),
    (
        "ue_share",
        "UE 점유율 격차",
        "ue_share_pct",
        "spread",
        "%p",
        "decrease",
        "사용자 재분배 설명용; 균등 UE 수가 균등 부하를 뜻하지 않음",
    ),
    (
        "traffic_share",
        "전달 트래픽 점유율 격차",
        "traffic_share_pct",
        "spread",
        "%p",
        "decrease",
        "제공 수요가 아닌 전달량의 점유율; 처리량 보호와 함께 사용",
    ),
]

OPTIONS += [
    (
        "pp_success",
        "Ping-pong / 성공 HO",
        "pingpong_per_success_pct",
        "sum",
        "%",
        "decrease",
        "동일 구간 핑퐁 / 성공 HO 합계; 성공 HO 10회 미만은 판정 보류",
    ),
    (
        "prb_imbalance",
        "PRB 불균형 (1 − Jain)",
        "prb_imbalance",
        "sum",
        "0–1",
        "decrease",
        "0에 가까울수록 균등; 최대 PRB와 처리량 제약을 함께 사용",
    ),
    (
        "rsrp_p10",
        "P10 RSRP (가장 낮은 셀)",
        "rsrp_p10_dbm",
        "min",
        "dBm",
        "increase",
        "미래 구간 UE·시간 표본의 하위 10% 지점; 셀 P10의 최솟값이며 네트워크 전체 P10이 아님",
    ),
    (
        "rsrp_gap",
        "평균 − P10 RSRP (가장 큰 셀)",
        "rsrp_mean_p10_gap_db",
        "max",
        "dB",
        "decrease",
        "신호 분포의 편차; 평균과 P10이 함께 악화되지 않도록 P10 하한과 병용",
    ),
    (
        "delay_tail",
        "100ms 초과 수신 패킷 비율",
        "delay_over_100ms_pct",
        "max",
        "%",
        "decrease",
        "수신 패킷 지연 히스토그램을 합산; 손실 패킷은 제외되므로 처리량 보호와 병용",
    ),
    (
        "ho_success",
        "성공 HO 횟수",
        "ho_successes",
        "sum",
        "회",
        "increase",
        "핑퐁 비율의 분모; 횟수 증가만으로 이동성 개선을 의미하지 않음",
    ),
]
ACTIVE_OPTIONS = {
    "ee",
    "goodput",
    "goodput_retention",
    "rlf",
    "pp_success",
    "prb_max",
    "prb_jain",
    "prb_imbalance",
    "rsrp",
    "rsrp_p10",
    "rsrp_gap",
    "sinr",
    "bler",
    "delay_tail",
    "energy",
    "ee_retention",
}


def recommendation(option, horizon, cells=None):
    cells = cells or list(CELLS)
    whole = set(cells) == set(CELLS)
    new_refs = {
        "rlf": ("network_rlf_count", "q75", "<="),
        "pp_success": ("network_pp_per_success_pct", "q75", "<="),
        "rsrp_p10": ("min_cell_rsrp_p10_dbm", "q10", ">="),
        "rsrp_gap": ("max_cell_rsrp_gap_db", "q75", "<="),
        "bler": ("max_cell_bler_pct", "q75", "<="),
        "delay_tail": ("max_cell_delay_over_100ms_pct", "q75", "<="),
    }
    if option in new_refs and whole:
        ref = json.loads(
            REFERENCE_PATH.with_name("threshold_reference_v16.json").read_text(encoding="utf-8")
        )
        metric, quantile, operator = new_refs[option]
        row = next(
            r for r in ref["distributions"] if r["horizon_s"] == horizon and r["metric"] == metric
        )
        return dict(
            target=row[quantile],
            operator=operator,
            source="train_" + quantile,
            note=f"학습 720개 실행, 30~720초 판단 시점(10초 간격), {horizon}초 중첩 구간 {quantile.upper()}; 연구 기준",
        )
    if option == "goodput_retention":
        return dict(
            target=95.0,
            operator=">=",
            source="research_design",
            note="동일 미래 구간 설정 유지 대비 95% 보존; 실험 정책값",
        )
    if option == "ee_retention":
        return dict(
            target=100.0,
            operator=">=",
            source="research_design",
            note="설정 유지 대비 에너지 효율 비악화",
        )
    if option == "prb_max":
        return dict(
            target=80.0,
            operator="<=",
            source="research_design",
            note="여유 용량을 위한 연구 상한; 180초 train 네트워크 최대 PRB Q90 약 79.3%",
        )
    if option == "prb_jain":
        return dict(
            target=0.9,
            operator=">=",
            source="research_design",
            note="균형 목표의 초기 연구값; 저부하에서는 필요성 확인",
        )
    if option in ("rlf", "pp_ratio") and whole:
        ref = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        metric = "network_rlf_count" if option == "rlf" else "network_pp_per_attempt_pct"
        row = next(
            r
            for r in ref["distributions"]
            if r["horizon_s"] == horizon and r["cell"] == "network" and r["metric"] == metric
        )
        return dict(
            target=row["q75"],
            operator="<=",
            source="train_q75",
            note=f"학습 720개 실행, {horizon}초 비중첩 구간 Q75; 워밍업 포함; 운영 SLA 아님",
        )
    return dict(
        target=None,
        operator="<=",
        source="user_required",
        note="이 범위의 검증된 기본 임계값이 없습니다. 서비스 요구값을 직접 입력하세요.",
    )


def entry(option, target, group="goal", operator=None, target_kind=None, cells=None, ident=None):
    descriptor = next((r for r in OPTIONS if r[0] == option), None)
    if descriptor is None:
        raise ContractError("Unknown KPI option")
    key, label, metric, aggregation, unit, direction, note = descriptor
    return dict(
        id=ident or f"{group}_{key}",
        option=key,
        label=label,
        metric=metric,
        cells=list(cells or CELLS),
        aggregation=aggregation,
        operator=operator or ("<=" if direction == "decrease" else ">="),
        target_kind=target_kind or ("relative_fraction" if group == "goal" else "absolute"),
        target=target,
        reference_kind="keep_forecast_same_window",
        weight=1.0,
        enabled=True,
        unit=unit,
        note=note,
    )


def default_spec(horizon=180, objective="energy_saving"):
    if horizon not in HORIZONS or objective not in PURPOSES:
        raise ContractError("Unknown policy horizon or objective")
    option, fraction = {
        "energy_saving": ("ee", 0.05),
        "throughput": ("goodput", 0.05),
        "load_balancing": ("prb_imbalance", -0.20),
        "handover_control": ("pp_success", -0.20),
        "coverage": ("rsrp_p10", 3.0),
        "custom": ("goodput", 0.05),
    }[objective]
    goals = [
        entry(
            option,
            fraction,
            target_kind="absolute_delta" if objective == "coverage" else "relative_fraction",
        )
    ]
    if objective == "coverage":
        goals.append(entry("rsrp", 1.0, target_kind="absolute_delta", ident="goal_rsrp_mean"))
    base = []
    for key in ("goodput_retention", "rlf", "pp_success", "rsrp_p10"):
        rec = recommendation(key, horizon)
        row = entry(key, rec["target"], "base", rec["operator"])
        row["recommendation"] = rec
        row["follows_recommendation"] = True
        base.append(row)
    extra = []
    if objective in ("energy_saving", "load_balancing"):
        key = "ee_retention" if objective == "energy_saving" else "prb_max"
        rec = recommendation(key, horizon)
        row = entry(key, rec["target"], "extra", rec["operator"])
        row.update(recommendation=rec, follows_recommendation=True)
        extra.append(row)
    if objective in ("coverage", "throughput"):
        key = "rsrp_gap" if objective == "coverage" else "bler"
        rec = recommendation(key, horizon)
        row = entry(key, rec["target"], "extra", rec["operator"])
        row.update(recommendation=rec, follows_recommendation=True)
        extra.append(row)
    return dict(
        version=VERSION,
        horizon_s=horizon,
        objective=objective,
        goals=goals,
        base_constraints=base,
        additional_constraints=extra,
    )


def compile_spec(spec, horizon):
    """Compile exactly the editor's active rows; no agent defaults are merged."""
    if not isinstance(spec, dict) or spec.get("version") not in (VERSION, "threshold-v2"):
        raise ContractError("intent_spec requires version=" + VERSION)
    if spec.get("horizon_s") != horizon or horizon not in HORIZONS:
        raise ContractError(
            "Policy horizon differs from the requested action; align the editor and prompt"
        )
    if spec.get("objective") not in PURPOSES:
        raise ContractError("Unknown policy objective")
    result = dict(
        policy_version=spec["version"],
        horizon_s=horizon,
        objective=spec["objective"],
        goals=[],
        constraints=[],
        safety={"multiplier": 0.0},
        policy_provenance="editable_research_thresholds_not_operator_sla",
    )
    ids = set()
    for group in ("goals", "base_constraints", "additional_constraints"):
        rows = spec.get(group)
        if not isinstance(rows, list) or len(rows) > 24:
            raise ContractError(f"{group} requires a list of at most 24 expectations")
        for raw in rows:
            if not isinstance(raw, dict) or not isinstance(raw.get("enabled", True), bool):
                raise ContractError("Each policy row must be an object with a boolean enabled flag")
            row = deepcopy(raw)
            if row.get("id") in ids:
                raise ContractError("Policy row IDs must be unique, including disabled rows")
            ids.add(row.get("id"))
            if not row.get("enabled", True):
                continue
            if group != "goals" and row.get("target_kind") != "absolute":
                raise ContractError("Constraints must specify a threshold, not an improvement goal")
            row["constraint_group"] = group
            # Recommendations are server evidence, never trusted client text.
            option = next(
                (
                    x[0]
                    for x in OPTIONS
                    if x[2] == row.get("metric") and x[3] == row.get("aggregation")
                ),
                "",
            )
            row["recommendation"] = recommendation(option, horizon, row.get("cells"))
            row["reference_kind"] = "keep_forecast_same_window"
            result["goals" if group == "goals" else "constraints"].append(row)
    result = validate_intent(result)
    result["definition_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    result["editor_spec"] = deepcopy(spec)
    return result


def catalog(horizon=180):
    if horizon not in HORIZONS:
        raise ContractError("Supported horizons are 90 and 180 seconds")
    return dict(
        version=VERSION,
        purposes=PURPOSES,
        metrics=[
            dict(
                key=k,
                label=l,
                metric=m,
                aggregation=a,
                unit=u,
                direction=d,
                note=n,
                recommendation=recommendation(k, horizon),
            )
            for k, l, m, a, u, d, n in OPTIONS
            if k in ACTIVE_OPTIONS
        ],
        templates={purpose: default_spec(horizon, purpose) for purpose in PURPOSES},
        reference={
            "partition": "train",
            "runs": 720,
            "seconds_per_run": 900,
            "horizon_s": horizon,
            "includes_warmup": True,
            "origin_stride_s": 10,
            "overlapping_windows": True,
            "status": "research_recommendations_not_validated_operator_sla",
        },
    )
