# Implementation Plan: SMO Agentic AI Framework

## Overview

이 계획은 `smo/` 아래에 Python/FastAPI 기반 최소 AIMLFW를 구축하고, 기존 `smo/agentic_ai` 런타임에 KPI Advisor Agent를 통합한 뒤 GNN 배치 프로빙, 간접 충돌 분석, 시간 구간 계획, A1 정책 발행, 재학습, 대시보드 및 근거 추적을 단계적으로 연결한다. 각 단계는 앞선 산출물을 재사용하며 마지막 단계에서 자동화된 종단간 흐름으로 배선한다.

## Tasks

- [x] 1. 공통 AIMLFW 패키지와 도메인 계약 구성
  - [x] 1.1 공통 모델, 검증 규칙, 설정 로더 구현
    - `smo/aimlfw/` 패키지 구조와 `common/` 모듈을 만들고 `FeatureRecord`, 셀별 `ParameterSet`, `ProbePlan`, `TemporalPlan`, `TrainingJob`, `ModelVersionRecord`, `EvidenceRecord` 스키마를 Python 타입/Pydantic 모델로 정의한다.
    - 5개 제어 파라미터 범위·스텝·TTT 허용값, Time_Step 0~4, Target_KPI 목록과 공통 오류 응답 계약을 단일 소스로 정의한다.
    - Feature Group, KPI 임계값, KPI 집계 방식, 충돌/재학습 임계값 설정 파일과 로더를 추가한다.
    - _요구사항: 1.1, 1.5, 5.3, 5.5, 7.1, 8.1, 9.5, 10.5, 11.14, 13.7_

  - [x] 1.2 Hypothesis 공통 생성기와 테스트 픽스처 구현
    - `smo/tests/property/strategies.py`에 유효/무효 셀별 ParameterSet, FeatureRecord, CSV 값, Threshold, ProbePlan, TemporalPlan 생성기를 구현한다.
    - 모든 속성 테스트가 `max_examples=100`과 설계 문서의 프로퍼티 태그 주석을 공통으로 사용하도록 설정한다.
    - _요구사항: 1.1~15.7_

- [x] 2. 데이터 추출 및 Feature Store 구현
  - [x] 2.1 Feature Store 영속화와 직렬화 API 구현
    - `smo/aimlfw/feature_store/`에 `(feature_group, time_step, cell_id)` 키 기반 원자적 upsert/조회와 포트 8102 FastAPI 서버를 구현한다.
    - 정렬된 CSV 직렬화·파싱, 임시 파일 후 원자적 rename, 스키마/손상 검증, 빈 집합 처리와 무부분결과 오류를 구현한다.
    - _요구사항: 1.9, 2.1~2.8_

  - [x] 2.2 Data Extractor와 180초 집계 API 구현
    - `smo/aimlfw/data_extractor/`와 `feature_groups/`에 포트 8101 `/extract`, `cell_kpi.csv` 검증, Decimal 기반 Time_Step 매핑 및 셀별 집계를 구현한다.
    - `command.sh`의 셀별 파라미터 맵과 seed를 파싱하고 CSV 최빈값 폴백, 범위 검증, 표본 카운트, data_quality, 6자리 반올림을 구현해 Feature Store에 원자적으로 반영한다.
    - 1,000,000행 스트리밍/청크 처리를 지원하고 외부 문자열 셀 식별자인 `cell` 열을 키로 사용한다.
    - _요구사항: 1.1~1.12_
  - [x] 2.3 Property 1 속성 테스트 작성: Feature_Record 키 유일성과 Time_Step 매핑
    - 임의 CSV 행에서 `(Time_Step, cell_id)` 유일성과 180초 경계 매핑을 검증한다.
    - **Property 1: Feature_Record 키 유일성과 Time_Step 매핑**
    - **검증: 요구사항 1.1, 1.6**

  - [x] 2.4 Property 2 속성 테스트 작성: 집계 규칙과 반올림
    - 임의 실수 목록의 sum/mean 결과와 소수점 6자리 반올림을 검증한다.
    - **Property 2: 집계 규칙과 반올림**
    - **검증: 요구사항 1.2**

  - [x] 2.5 Property 3 속성 테스트 작성: 유효/제외 표본 카운트 보존
    - 정상·nan·빈 값·비실수 혼합에서 카운트 보존과 유효 표본 집계를 검증한다.
    - **Property 3: 유효/제외 표본 카운트 보존**
    - **검증: 요구사항 1.3**

  - [x] 2.6 Property 4 속성 테스트 작성: data_quality 판정 규칙
    - complete/partial/insufficient 조건의 상호 배타성과 정확한 분류를 검증한다.
    - **Property 4: data_quality 판정 규칙**
    - **검증: 요구사항 1.4, 1.11**

  - [x] 2.7 Property 5 속성 테스트 작성: 실행 메타데이터 전파
    - 셀별 파라미터, 원본 경로와 seed가 모든 FeatureRecord에 보존되는지 검증한다.
    - **Property 5: 실행 메타데이터 전파**
    - **검증: 요구사항 1.5**

  - [x] 2.8 Property 6 속성 테스트 작성: time_s 범위 필터링
    - 범위 밖/비실수 행만 제외되고 제외 수가 정확한지 검증한다.
    - **Property 6: time_s 범위 필터링**
    - **검증: 요구사항 1.7**

  - [x] 2.9 Property 7 속성 테스트 작성: 필수 열/집계 규칙 누락 시 무변경 오류
    - 누락 목록, 레코드 미생성 및 Feature Store 무변경을 검증한다.
    - **Property 7: 필수 열/집계 규칙 누락 시 무변경 오류**
    - **검증: 요구사항 1.8**

  - [x] 2.10 Property 8 속성 테스트 작성: 재실행 멱등성
    - 동일 결과 디렉터리 재추출 후 개수·키·필드와 중복 부재를 검증한다.
    - **Property 8: 재실행 멱등성**
    - **검증: 요구사항 1.9**

  - [x] 2.11 Property 9 속성 테스트 작성: 잘못된 입력 경로 오류
    - 미존재·읽기 불가·빈 CSV에서 원인 오류와 저장소 무변경을 검증한다.
    - **Property 9: 잘못된 입력 경로 오류**
    - **검증: 요구사항 1.10**

  - [x] 2.12 Property 10 속성 테스트 작성: Control_Parameter 범위 위반 시 무변경 오류
    - 모든 범위/스텝 위반 조합에서 위반 상세와 무변경을 검증한다.
    - **Property 10: Control_Parameter 범위 위반 시 무변경 오류**
    - **검증: 요구사항 1.12**

  - [x] 2.13 Property 11 속성 테스트 작성: 직렬화 순서와 개수
    - 최대 100,000개 레코드의 정렬 순서와 기록 개수를 검증한다.
    - **Property 11: 직렬화 순서와 개수**
    - **검증: 요구사항 2.1**

  - [x] 2.14 Property 12 속성 테스트 작성: 직렬화-파싱 라운드트립
    - 빈 집합을 포함해 필드·개수·수치 오차·문자열 동일성을 검증한다.
    - **Property 12: 직렬화-파싱 라운드트립**
    - **검증: 요구사항 2.2, 2.3, 2.8**

  - [x] 2.15 Property 13 속성 테스트 작성: 필드 이름 불일치 파싱 오류
    - 누락/미정의 필드 상세와 무부분결과를 검증한다.
    - **Property 13: 필드 이름 불일치 파싱 오류**
    - **검증: 요구사항 2.4**

  - [x] 2.16 Property 14 속성 테스트 작성: 손상된 레코드 파싱 오류
    - 임의 위치 손상에서 최초 인덱스와 무부분결과를 검증한다.
    - **Property 14: 손상된 레코드 파싱 오류**
    - **검증: 요구사항 2.7**

  - [x] 2.17 실제 ns-3 결과 기반 데이터 계층 통합 및 성능 테스트 작성
    - 실제 `scenarios/results/**/cell_kpi.csv`와 `command.sh`로 셀당 5개 구간, 폴백 메타데이터와 원자적 재실행을 검증한다.
    - 합성 1,000,000행 입력이 300초 이내 처리되는지 단발성 성능 테스트로 검증한다.
    - _요구사항: 1.1, 1.5, 1.6, 1.9, 1.10, 2.1~2.8_
- [x] 3. 모델 저장소, 레지스트리 및 학습 파이프라인 구현
  - [x] 3.1 버전별 Model Storage 구현
    - `smo/aimlfw/model_storage/`에 `models/<model_name>/<version>/model.pt` 고유 경로 할당, 존재 확인, 비덮어쓰기 저장을 구현한다.
    - _요구사항: 4.6, 4.8_

  - [x] 3.2 MongoDB Model Registry API 구현
    - `smo/aimlfw/model_registry/`에 MongoDB `aimlfw.model_registry`와 포트 8104 등록/목록/최신/특정 버전 API를 구현한다.
    - 원자적 버전 증가, 최대 500개 정렬 조회, 메타데이터·산출물 검증과 오류 계약을 구현한다.
    - _요구사항: 4.1~4.9_

  - [x] 3.3 GNN 학습 스크립트와 Training Manager API 구현
    - `smo/aimlfw/training_manager/`에 포트 8103 작업 생성/조회, JSON 상태 스냅샷, asyncio subprocess 실행 및 시간초과 중단을 구현한다.
    - `train_gnn.py`에 셀 그래프/Time_Step/셀별 5개 파라미터 입력과 Target_KPI 다중 출력을 갖는 결정론적 GNN 학습·평가·산출물 저장을 구현한다.
    - 4단계 순서, 조기 중단, 실패 분류, 지표 재시도와 완료 메타데이터를 Model Registry에 연결한다.
    - _요구사항: 3.1~3.9, 4.1, 4.6_

  - [x] 3.4 Property 15 속성 테스트 작성: 파이프라인 단계 순서와 조기 중단
    - **Property 15: 파이프라인 단계 순서와 조기 중단**
    - **검증: 요구사항 3.2**

  - [x] 3.5 Property 16 속성 테스트 작성: 학습 완료 메타데이터 완전성
    - **Property 16: 학습 완료 메타데이터 완전성**
    - **검증: 요구사항 3.4**

  - [x] 3.6 Property 17 속성 테스트 작성: 학습 실패 메타데이터
    - **Property 17: 학습 실패 메타데이터**
    - **검증: 요구사항 3.5**

  - [x] 3.7 Property 18 속성 테스트 작성: 동일 시드 재현성
    - **Property 18: 동일 시드 재현성**
    - **검증: 요구사항 3.6**

  - [x] 3.8 Property 19 속성 테스트 작성: 지표 기록 재시도와 최종 상태
    - **Property 19: 지표 기록 재시도와 최종 상태**
    - **검증: 요구사항 3.7**

  - [x] 3.9 Property 20 속성 테스트 작성: 학습 데이터 부족 오류
    - **Property 20: 학습 데이터 부족 오류**
    - **검증: 요구사항 3.8**

  - [x] 3.10 Property 21 속성 테스트 작성: 최대 실행 시간 초과 처리
    - 모킹된 시계/학습 함수로 중단과 실패 사유를 검증한다.
    - **Property 21: 최대 실행 시간 초과 처리**
    - **검증: 요구사항 3.9**

  - [x] 3.11 Property 22 속성 테스트 작성: 버전 자동 증가
    - **Property 22: 버전 자동 증가**
    - **검증: 요구사항 4.1**

  - [x] 3.12 Property 23 속성 테스트 작성: 버전 목록 정렬과 개수 제한
    - **Property 23: 버전 목록 정렬과 개수 제한**
    - **검증: 요구사항 4.2, 4.9**

  - [x] 3.13 Property 24 속성 테스트 작성: 최신 버전은 최대 버전
    - **Property 24: 최신 버전은 최대 버전**
    - **검증: 요구사항 4.3**

  - [x] 3.14 Property 25 속성 테스트 작성: 모델/버전 not found
    - **Property 25: 모델/버전 not found**
    - **검증: 요구사항 4.4, 4.5**

  - [x] 3.15 Property 26 속성 테스트 작성: 산출물 경로 고유성
    - **Property 26: 산출물 경로 고유성**
    - **검증: 요구사항 4.6**

  - [x] 3.16 Property 27 속성 테스트 작성: 등록 메타데이터 검증
    - **Property 27: 등록 메타데이터 검증**
    - **검증: 요구사항 4.7**

  - [x] 3.17 Property 28 속성 테스트 작성: 산출물 미조회 시 등록 거부
    - **Property 28: 산출물 미조회 시 등록 거부**
    - **검증: 요구사항 4.8**

  - [x] 3.18 학습 작업과 MongoDB 레지스트리 통합 테스트 작성
    - 작업 생성 5초, 상태 조회 1초, 단계 이력, 실제 산출물 등록과 동일 seed 재현성을 자동 검증한다.
    - _요구사항: 3.1~3.9, 4.1~4.9_

- [x] 4. 체크포인트 - 데이터와 학습 계층 테스트 확인
  - 모든 테스트가 통과하는지 확인하고, 질문이 생기면 사용자에게 문의한다.
- [x] 5. GNN 배치 추론 서비스 구현
  - [x] 5.1 Inference Service API와 모델 로딩 구현
    - `smo/aimlfw/inference_service/`에 포트 8105 `/status`, `/predict/batch`와 기동 시 60초 제한 모델 로딩을 구현한다.
    - 1~64개 순서 보존 배치, Time_Step 조건, 전체/셀별 KPI 완전성, 결정론적 forward pass 및 원자적 오류 응답을 구현한다.
    - _요구사항: 5.1~5.13_

  - [x] 5.2 Property 29 속성 테스트 작성: 배치 예측 순서 보존과 완전성
    - **Property 29: 배치 예측 순서 보존과 완전성**
    - **검증: 요구사항 5.2, 5.3, 5.9**

  - [x] 5.3 Property 30 속성 테스트 작성: Time_Step 적용과 응답 반영
    - **Property 30: Time_Step 적용과 응답 반영**
    - **검증: 요구사항 5.4, 5.13**

  - [x] 5.4 Property 31 속성 테스트 작성: 파라미터 범위 위반 시 전체 거부
    - **Property 31: 파라미터 범위 위반 시 전체 거부**
    - **검증: 요구사항 5.5**

  - [x] 5.5 Property 32 속성 테스트 작성: 배치 크기 경계 오류
    - **Property 32: 배치 크기 경계 오류**
    - **검증: 요구사항 5.6, 5.7**

  - [x] 5.6 Property 33 속성 테스트 작성: 예측 결정론
    - **Property 33: 예측 결정론**
    - **검증: 요구사항 5.8**

  - [x] 5.7 Property 34 속성 테스트 작성: 오류 응답 완전성과 부분결과 배제
    - **Property 34: 오류 응답 완전성과 부분결과 배제**
    - **검증: 요구사항 5.10**

  - [x] 5.8 Property 35 속성 테스트 작성: 모델 미로드 상태 전이
    - **Property 35: 모델 미로드 상태 전이**
    - **검증: 요구사항 5.11**

  - [x] 5.9 Property 36 속성 테스트 작성: Time_Step 유효성 검증
    - **Property 36: Time_Step 유효성 검증**
    - **검증: 요구사항 5.12**

  - [x] 5.10 추론 서비스 로딩·성능·실패 통합 테스트 작성
    - 실제 등록 산출물 로드, 64개 요청 5초 SLA, 60초 로딩 실패 상태와 부분 결과 부재를 검증한다.
    - _요구사항: 5.1, 5.6, 5.10, 5.11_

- [x] 6. GNN MCP 도구 서버 구현
  - [x] 6.1 KPI Advisor용 읽기 전용 MCP 도구 구현
    - `smo/agentic_ai/agents/kpi-advisor/mcp_server.py`에 `predict_batch`, `marginal_effect`, `current_model`과 도구 목록 메타데이터를 구현한다.
    - 사전 입력 검증, 30초 무재시도 타임아웃, 성공/오류 pass-through를 구현하고 변경 도구를 노출하지 않는다.
    - _요구사항: 6.1~6.10, 7.9_

  - [x] 6.2 Property 37 속성 테스트 작성: 성공 응답 pass-through 불변식
    - **Property 37: 성공 응답 pass-through 불변식**
    - **검증: 요구사항 6.4**

  - [x] 6.3 Property 38 속성 테스트 작성: 오류 응답 pass-through 불변식
    - **Property 38: 오류 응답 pass-through 불변식**
    - **검증: 요구사항 6.5**

  - [x] 6.4 Property 39 속성 테스트 작성: 입력 검증 선차단
    - **Property 39: 입력 검증 선차단**
    - **검증: 요구사항 6.9**

  - [x] 6.5 Property 40 속성 테스트 작성: 타임아웃 시 무변경 오류
    - **Property 40: 타임아웃 시 무변경 오류**
    - **검증: 요구사항 6.10**

  - [x] 6.6 MCP 도구 계약과 읽기 전용 표면 테스트 작성
    - 3개 도구 시그니처, 인자 필수 여부/범위, 변경 도구 부재와 Inference Service 연동을 검사한다.
    - _요구사항: 6.1~6.10_
- [x] 7. KPI Advisor Agent의 프로빙 및 판정 구현
  - [x] 7.1 KPI Advisor 에이전트 계약과 서버 골격 구현
    - `smo/agentic_ai/agents/kpi-advisor/`에 `agent.md`, `schemas.py`, `server.py`, `pyproject.toml`을 기존 5파일 컨벤션에 맞춰 추가하고 `/health`, `/invoke`만 노출한다.
    - health 모델 상태, 입력 검증, 한국어 근거, 60초 응답, 추천 개수 및 금지 명령 문자열 검증 계약을 정의한다.
    - _요구사항: 12.1~12.3, 12.6~12.8, 12.11_

  - [x] 7.2 Baseline 구성, Probe Plan 생성 및 배치 실행 구현
    - `agent.py`에 최근 FeatureRecord로 셀별 미지정 값을 채우는 Baseline과 각 셀×파라미터 ±1스텝 variant 생성을 구현한다.
    - 64개 순서 보존 분할, 재전송, 배치 모델 버전 일치 검증 및 0 baseline 변화율 처리를 구현한다.
    - _요구사항: 7.1~7.12_

  - [x] 7.3 Threshold 판정과 대안 추천 구현
    - `agent.py`에 Threshold 설정 검증, acceptable/degrading/unknown 판정, 누락 예측 부분 배제와 추천 필터·결정론적 정렬을 추가한다.
    - 판정·추천·Evidence 식별자를 단일 응답으로 조립하되 정책 변경은 Policy Manager 경로로만 위임한다.
    - _요구사항: 7.9, 8.1~8.12, 12.6~12.8_

  - [x] 7.4 Property 41 속성 테스트 작성: Baseline_Parameter_Set 구성
    - **Property 41: Baseline_Parameter_Set 구성**
    - **검증: 요구사항 7.1**

  - [x] 7.5 Property 42 속성 테스트 작성: Probe_Plan variant 불변식
    - **Property 42: Probe_Plan variant 불변식**
    - **검증: 요구사항 7.2, 7.3, 7.4**

  - [x] 7.6 Property 43 속성 테스트 작성: 배치 분할 순서 보존
    - **Property 43: 배치 분할 순서 보존**
    - **검증: 요구사항 7.5, 7.6**

  - [x] 7.7 Property 44 속성 테스트 작성: 변화율 계산
    - **Property 44: 변화율 계산**
    - **검증: 요구사항 7.7, 7.10**

  - [x] 7.8 Property 45 속성 테스트 작성: 빈 요청 거부
    - **Property 45: 빈 요청 거부**
    - **검증: 요구사항 7.8**

  - [x] 7.9 Property 46 속성 테스트 작성: 배치 재전송과 실패 중단
    - **Property 46: 배치 재전송과 실패 중단**
    - **검증: 요구사항 7.11**

  - [x] 7.10 Property 47 속성 테스트 작성: 배치 간 모델 버전 일치 검증
    - **Property 47: 배치 간 모델 버전 일치 검증**
    - **검증: 요구사항 7.12**

  - [x] 7.11 Property 48 속성 테스트 작성: Degradation_Verdict 경계 포함 판정
    - **Property 48: Degradation_Verdict 경계 포함 판정**
    - **검증: 요구사항 8.2, 8.3**

  - [x] 7.12 Property 49 속성 테스트 작성: unknown 판정 조건
    - **Property 49: unknown 판정 조건**
    - **검증: 요구사항 8.4, 8.12**

  - [x] 7.13 Property 50 속성 테스트 작성: 추천 후보 필터링과 결정론적 정렬
    - **Property 50: 추천 후보 필터링과 결정론적 정렬**
    - **검증: 요구사항 8.5~8.8**

  - [x] 7.14 Property 51 속성 테스트 작성: 임계값 설정 검증
    - **Property 51: 임계값 설정 검증**
    - **검증: 요구사항 8.1, 8.10**

  - [x] 7.15 Property 52 속성 테스트 작성: 누락 예측값의 부분 배제
    - **Property 52: 누락 예측값의 부분 배제**
    - **검증: 요구사항 8.11**

  - [x] 7.16 KPI Advisor 프로빙·판정 API 통합 테스트 작성
    - MCP 오류/타임아웃/버전 불일치, 한국어 응답, health 3초 및 invoke 60초 계약을 자동 검증한다.
    - _요구사항: 7.1~8.12, 12.2, 12.3, 12.6~12.8, 12.11_
- [x] 8. 간접 충돌 분석과 시간 구간 스케줄링 구현
  - [x] 8.1 Conflict Analyzer 구현
    - `smo/agentic_ai/agents/kpi-advisor/conflict_analyzer.py`에 Marginal Effect 클램프/반올림, 경계 포함 충돌 판정, 누락 부분 배제와 결정론적 정렬을 순수 함수로 구현한다.
    - _요구사항: 9.1~9.11_

  - [x] 8.2 Property 53 속성 테스트 작성: Marginal_Effect 클램프와 반올림
    - **Property 53: Marginal_Effect 클램프와 반올림**
    - **검증: 요구사항 9.1**

  - [x] 8.3 Property 54 속성 테스트 작성: Indirect_Conflict 경계 포함 판정
    - **Property 54: Indirect_Conflict 경계 포함 판정**
    - **검증: 요구사항 9.2~9.4**

  - [x] 8.4 Property 55 속성 테스트 작성: 결정론적 정렬 재현
    - **Property 55: 결정론적 정렬 재현 (idempotence)**
    - **검증: 요구사항 9.7**

  - [x] 8.5 Property 56 속성 테스트 작성: Conflict_Threshold 검증과 기본값
    - **Property 56: Conflict_Threshold 유효성 검증과 기본값**
    - **검증: 요구사항 9.5, 9.6, 9.9**

  - [x] 8.6 Property 57 속성 테스트 작성: 누락 파라미터의 부분 배제
    - **Property 57: 누락 파라미터의 부분 배제**
    - **검증: 요구사항 9.8**

  - [x] 8.7 Property 58 속성 테스트 작성: 유효 파라미터 부족 시 빈 목록
    - **Property 58: 유효 파라미터 부족 시 빈 목록**
    - **검증: 요구사항 9.10**

  - [x] 8.8 Temporal Scheduler 구현
    - `smo/agentic_ai/agents/kpi-advisor/temporal_scheduler.py`에 Time_Step 0~4 배치 평가, objective별 최적 선택과 objective 미지정 위반 최소 선택을 구현한다.
    - Feature Store 후보 폴백, 결정론적 동률 해소, KPI 합/평균, 임계 위반과 전환 목록, 무부분계획 오류를 구현한다.
    - _요구사항: 10.1~10.13_

  - [x] 8.9 Property 59 속성 테스트 작성: Time_Step별 최적 배정과 구조 완전성
    - **Property 59: Time_Step별 최적 배정과 구조 완전성**
    - **검증: 요구사항 10.1, 10.4, 10.6, 10.10**

  - [x] 8.10 Property 60 속성 테스트 작성: objective 미지정 시 위반 최소 후보 선택
    - **Property 60: objective 미지정 시 위반 최소 후보 선택**
    - **검증: 요구사항 10.2**

  - [x] 8.11 Property 61 속성 테스트 작성: 평가 배치 분할
    - **Property 61: 평가 배치 분할**
    - **검증: 요구사항 10.3**

  - [x] 8.12 Property 62 속성 테스트 작성: 5구간 누적값 집계 규칙
    - **Property 62: 5구간 누적값 집계 규칙**
    - **검증: 요구사항 10.5**

  - [x] 8.13 Property 63 속성 테스트 작성: 전환 목록 계산
    - **Property 63: 전환 목록 계산**
    - **검증: 요구사항 10.7**

  - [x] 8.14 Property 64 속성 테스트 작성: 후보 0개 시 Feature Store 기반 도출
    - **Property 64: 후보 0개 시 Feature_Store 기반 도출**
    - **검증: 요구사항 10.8, 10.9**

  - [x] 8.15 Property 65 속성 테스트 작성: 동률 배정의 결정론적 해소
    - **Property 65: 동률 배정의 결정론적 해소**
    - **검증: 요구사항 10.11**

  - [x] 8.16 Property 66 속성 테스트 작성: Time_Step 범위 검증
    - **Property 66: Time_Step 범위 검증**
    - **검증: 요구사항 10.12**

  - [x] 8.17 Property 67 속성 테스트 작성: 예측 불가 시 무부분계획 오류
    - **Property 67: 예측 불가 시 무변경 오류**
    - **검증: 요구사항 10.13**

  - [x] 8.18 충돌 분석 및 시간 계획 통합·성능 테스트 작성
    - 200쌍 분석 2000ms, 5구간 전체 배치, 후보 폴백과 예측 실패 전파를 검증한다.
    - _요구사항: 9.1~9.11, 10.1~10.13_
- [x] 9. A1 Policy Manager 연동 구현
  - [x] 9.1 A1 Policy Type/Instance 변환과 발행 구현
    - `smo/agentic_ai/agents/kpi-advisor/policy_manager.py`에 포트 9000 A1 Mediator HTTP 클라이언트와 8개 엔드포인트 화이트리스트를 구현한다.
    - 3개 고정 셀×5개 파라미터의 15개 정수 flat schema, CIO/HYS x10 변환, 직렬화 왕복, 승인 게이팅, 5개 시간 인스턴스와 상태 조회를 구현한다.
    - 4xx/5xx 무재시도, 연결 실패 총 3회 시도, 발행 시도 기록 및 `status_unavailable` 처리를 구현한다.
    - _요구사항: 11.1~11.16_

  - [x] 9.2 Property 68 속성 테스트 작성: Policy Type 등록 멱등성
    - **Property 68: Policy_Type 등록 멱등성**
    - **검증: 요구사항 11.2**

  - [x] 9.3 Property 69 속성 테스트 작성: Policy Instance 개수 불변식
    - **Property 69: Policy_Instance 개수 불변식**
    - **검증: 요구사항 11.3, 11.4**

  - [x] 9.4 Property 70 속성 테스트 작성: Policy Instance 직렬화 왕복
    - **Property 70: Policy_Instance 직렬화 왕복**
    - **검증: 요구사항 11.5~11.8**

  - [x] 9.5 Property 71 속성 테스트 작성: A1 4xx/5xx 무재시도 실패
    - 모킹된 HTTP 클라이언트로 호출 횟수와 시도 기록을 검증한다.
    - **Property 71: A1 4xx/5xx 무재시도 실패**
    - **검증: 요구사항 11.9**

  - [x] 9.6 Property 72 속성 테스트 작성: 승인 게이팅
    - **Property 72: 승인 게이팅**
    - **검증: 요구사항 11.10**

  - [x] 9.7 Property 73 속성 테스트 작성: 상태 조회 실패 시 인스턴스 유지
    - **Property 73: 상태 조회 실패 시 인스턴스 유지**
    - **검증: 요구사항 11.11, 11.12**

  - [x] 9.8 Property 74 속성 테스트 작성: 미등록 Policy Type 발행 거부
    - **Property 74: 미등록 Policy_Type 발행 거부**
    - **검증: 요구사항 11.13**

  - [x] 9.9 Property 75 속성 테스트 작성: 파라미터 검증 실패 거부
    - **Property 75: 파라미터 검증 실패 거부**
    - **검증: 요구사항 11.14**

  - [x] 9.10 Property 76 속성 테스트 작성: 연결 실패 재시도 한도
    - **Property 76: 연결 실패 재시도 한도**
    - **검증: 요구사항 11.15**

  - [x] 9.11 실제 A1 Mediator 계약 통합 테스트 작성
    - `A1_Mediator_standalone/A1_Mediator/app/main.py`를 대상으로 정책 유형 조회/등록, 인스턴스 생성/상태 조회 및 flat integer schema 호환성을 자동 검증한다.
    - 허용된 8개 외 엔드포인트 호출이 없고 TemporalPlan 발행 시 정확히 5개 인스턴스가 생성되는지 검증한다.
    - _요구사항: 11.1~11.16_

- [x] 10. 체크포인트 - 판단과 A1 계층 테스트 확인
  - 모든 테스트가 통과하는지 확인하고, 질문이 생기면 사용자에게 문의한다.
- [x] 11. 기존 Multi-Agent Runtime에 KPI Advisor 통합
  - [x] 11.1 planner_manifest 등록과 Planning Agent 위임 경로 구현
    - 기존 `knowledge_registry.py` 계약에 맞춰 KPI Advisor의 이름, 능력, 제약, 입출력 스키마 참조와 `contract_ref`를 MongoDB `knowledge.agents`에 등록한다.
    - RAN 파라미터 영향 예측 subtask의 `langgraph_spec`에 Baseline ParameterSet과 Time_Step을 받는 KPI Advisor 노드를 포함하고, 등록 실패 시 후보 제외 및 실행 실패 시 `advisor_unavailable`로 나머지 노드를 계속 실행한다.
    - 새 오케스트레이션 계층을 만들지 않고 기존 deep-agents 실행 경로에 연결한다.
    - _요구사항: 12.1, 12.4, 12.5, 12.9, 12.10_

  - [x] 11.2 Property 77 속성 테스트 작성: planner_manifest 등록과 후보 노출
    - **Property 77: planner_manifest 등록과 후보 노출**
    - **검증: 요구사항 12.4, 12.9**

  - [x] 11.3 Property 78 속성 테스트 작성: langgraph_spec 노드 포함
    - **Property 78: langgraph_spec 노드 포함**
    - **검증: 요구사항 12.5**

  - [x] 11.4 Property 79 속성 테스트 작성: 한국어 응답 텍스트 제약
    - **Property 79: 한국어 응답 텍스트 제약**
    - **검증: 요구사항 12.6**

  - [x] 11.5 Property 80 속성 테스트 작성: 추천 목록 개수 제약
    - **Property 80: 추천 목록 개수 제약**
    - **검증: 요구사항 12.7**

  - [x] 11.6 Property 81 속성 테스트 작성: 금지 문자열 부재
    - **Property 81: 금지 문자열 부재**
    - **검증: 요구사항 12.8**

  - [x] 11.7 Property 82 속성 테스트 작성: advisor_unavailable 허용 진행
    - **Property 82: advisor_unavailable 허용 진행**
    - **검증: 요구사항 12.10**

  - [x] 11.8 Property 83 속성 테스트 작성: 입력 스키마 검증 선차단
    - **Property 83: 입력 스키마 검증 선차단**
    - **검증: 요구사항 12.11**

  - [x] 11.9 Multi-Agent Runtime 계약 통합 테스트 작성
    - KPI Advisor의 정확한 5파일 구성과 `/health`, `/invoke` 두 엔드포인트만의 노출을 정적 검사한다.
    - 실제 Knowledge DB에서 manifest 등록/조회, Planning Agent 후보 선택, 실패 허용 실행, 한국어 응답 및 3초/60초 SLA를 자동 검증한다.
    - _요구사항: 12.1~12.11_

- [x] 12. Evidence Record 생성과 원자적 저장 구현
  - [x] 12.1 판단 근거 조립기와 MongoDB 저장소 구현
    - `knowledge.evidence_records` 저장소와 고유 식별자 생성을 구현하고 모델/Probe Plan/예측/임계값/판정/한계효과/충돌/시간 배정 근거를 하나의 Evidence Record로 조립한다.
    - 저장 성공 후에만 식별자를 KPI Advisor 응답에 연결하고, 실패 시 부분 문서를 제거하거나 트랜잭션을 중단하며 `no_prediction_evidence`와 저장 실패 계약을 구현한다.
    - KPI Advisor 응답의 모든 KPI 수치를 저장된 Evidence Record의 값에서만 투영하도록 판정 응답 경로에 배선한다.
    - _요구사항: 15.1~15.7_

  - [x] 12.2 Property 93 속성 테스트 작성: Evidence Record 필드 완전성
    - **Property 93: Evidence_Record 필드 완전성**
    - **검증: 요구사항 15.1, 15.2, 15.3**

  - [x] 12.3 Property 94 속성 테스트 작성: Evidence Record 식별자 고유성
    - **Property 94: Evidence_Record 식별자 고유성**
    - **검증: 요구사항 15.4**

  - [x] 12.4 Property 95 속성 테스트 작성: 저장 실패 원자성
    - **Property 95: 저장 실패 원자성**
    - **검증: 요구사항 15.5**

  - [x] 12.5 Property 96 속성 테스트 작성: 예측값 없는 판정 거부
    - **Property 96: 예측값 없는 판정 거부**
    - **검증: 요구사항 15.6**

  - [x] 12.6 Property 97 속성 테스트 작성: 응답 KPI 필드는 Evidence Record의 부분집합
    - **Property 97: 응답 KPI 필드는 Evidence_Record의 부분집합**
    - **검증: 요구사항 15.7**

  - [x] 12.7 Evidence Record MongoDB 통합 테스트 작성
    - 실제 Knowledge DB에 판정·충돌·시간 계획 근거를 저장하고 고유 ID로 완전한 문서를 다시 조회하는 흐름을 검증한다.
    - 저장 실패 주입 시 부분 문서가 없고 KPI Advisor가 Evidence ID 없는 성공 응답을 반환하지 않는지 검증한다.
    - _요구사항: 15.1~15.7_
- [x] 13. 예측 오차 기반 재학습과 모델 승격 구현
  - [x] 13.1 Retraining Controller와 원자적 서빙 버전 전환 구현
    - `smo/aimlfw/retraining_controller/`에 신규 결과 감지, 실제값-예측값 오차 기록, KPI별 재학습 트리거와 진행 중 작업 중복 억제를 구현한다.
    - 동일 평가 데이터셋의 신·구 버전 비교로 승격/거부를 결정하고 Model Registry 상태와 Inference Service의 내부 서빙 버전 전환을 연결하되, 결정 전과 실패/24시간 초과 시 기존 버전을 유지한다.
    - 임계값 기본화, 예측/실측 누락 건너뛰기와 모든 사유 기록을 구현한다.
    - _요구사항: 13.1~13.11_

  - [x] 13.2 Property 84 속성 테스트 작성: Prediction Error 계산
    - **Property 84: Prediction_Error 계산**
    - **검증: 요구사항 13.2**

  - [x] 13.3 Property 85 속성 테스트 작성: 재학습 트리거
    - **Property 85: 재학습 트리거**
    - **검증: 요구사항 13.3**

  - [x] 13.4 Property 86 속성 테스트 작성: 모델 승격/거부 조건
    - **Property 86: 모델 승격/거부 조건**
    - **검증: 요구사항 13.4, 13.5, 13.6**

  - [x] 13.5 Property 87 속성 테스트 작성: Retraining Threshold 검증과 기본값
    - **Property 87: Retraining_Threshold 검증과 기본값**
    - **검증: 요구사항 13.7**

  - [x] 13.6 Property 88 속성 테스트 작성: 재학습 중 서빙 버전 불변
    - **Property 88: 재학습 중 서빙 버전 불변**
    - **검증: 요구사항 13.8**

  - [x] 13.7 Property 89 속성 테스트 작성: 조회 실패 시 건너뛰기
    - **Property 89: 조회 실패 시 건너뛰기**
    - **검증: 요구사항 13.9**

  - [x] 13.8 Property 90 속성 테스트 작성: 중복 트리거 방지
    - **Property 90: 중복 트리거 방지**
    - **검증: 요구사항 13.10**

  - [x] 13.9 Property 91 속성 테스트 작성: 실패/시간초과 처리
    - **Property 91: 실패/시간초과 처리**
    - **검증: 요구사항 13.11**

  - [x] 13.10 신규 결과 감지·재학습·승격 통합 테스트 작성
    - 파일시스템 watcher가 60초 이내 신규 결과를 감지하고 KPI별 작업을 생성하는 흐름을 모킹된 시계와 실제 서비스 계약으로 검증한다.
    - 재학습 중 기존 버전 응답, 성공 승격, 품질 저하 거부, 중복 트리거 및 24시간 실패 처리까지 자동 검증한다.
    - _요구사항: 13.1~13.11_

- [ ] 14. SMO Dashboard 가시화 확장
  - [ ] 14.1 KPI Advisor 대시보드 API와 화면 구성 구현
    - 기존 `smo/agentic_ai/dashboard/server.py`와 `static/`에 프로빙 결과, 판정/충돌, 5개 시간 배정, 현재 모델, 정책 상태를 조회하는 프록시·캐시와 화면 컴포넌트를 추가한다.
    - 로딩 중 이전 데이터 유지, 전체/부분 `데이터 없음`, 마지막 성공 시각, `status_unavailable`, 빈 충돌 목록 및 Probe Variant 100개 절단 표시를 구현한다.
    - 신규 대시보드 서버를 추가하지 않고 기존 서버를 확장한다.
    - _요구사항: 14.1~14.11_

  - [ ] 14.2 Property 92 속성 테스트 작성: Probe Variant 표시 절단
    - **Property 92: Probe_Variant 표시 절단**
    - **검증: 요구사항 14.10**

  - [ ] 14.3 대시보드 렌더링·오류 상태 테스트 작성
    - 표/판정/충돌/시간 계획/모델/정책 영역의 스냅샷 및 DOM 테스트를 작성하고 0건, 부분 누락, 10초 타임아웃, 로딩 중 상태를 검증한다.
    - 프로빙 결과의 3초 표시 계약과 시간 계획 0~4 오름차순 5행, 초 단위 조회 시각을 자동 검증한다.
    - _요구사항: 14.1~14.11_

- [ ] 15. 전체 서비스 배선과 자동화된 종단간 검증
  - [x] 15.1 최소 로컬 AIMLFW 서비스 실행 구성과 전체 경로 배선
    - Data Extractor(8101), Feature Store(8102), Training Manager(8103), Model Registry(8104), Inference Service(8105), GNN MCP, KPI Advisor, Retraining Controller, 기존 Dashboard와 A1 Mediator(9000)의 설정·의존성 주입·health 확인 코드를 연결한다.
    - 데이터/학습/모델 관리/추론/Agent 판단/A1 제어 경계를 유지하면서 실제 ns-3 결과에서 Evidence ID와 정책 상태까지 이어지는 로컬 실행 구성을 구현한다.
    - _요구사항: 1.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.9, 9.11, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1_

  - [ ] 15.2 컴포넌트 계약 통합 테스트 스위트 작성
    - 각 FastAPI/MCP 경계의 스키마, 오류 코드, 모델 이름·버전 전파, 순서 보존과 부분 결과 금지를 자동 검증한다.
    - MongoDB의 Model Registry/planner manifest/Evidence Record와 로컬 Feature/Model Storage의 실제 읽기·쓰기 계약을 함께 검증한다.
    - _요구사항: 1.1~15.7_

  - [ ] 15.3 정상 흐름 종단간 테스트 작성
    - 실제 `cell_kpi.csv` 샘플 추출 → Feature Store → 학습/등록 → 배치 추론/MCP → KPI Advisor 프로빙·판정·간접 충돌 → Temporal Plan → 승인된 A1 정책 5건 → Evidence 조회 → Dashboard 표시를 한 자동화 테스트로 검증한다.
    - 각 결과가 동일 모델 버전과 동일 Parameter Set 근거를 추적하고 RAN 직접 변경 없이 A1 경로만 사용하는지 확인한다.
    - _요구사항: 1.1~12.11, 14.1~15.7_

  - [ ] 15.4 실패·재학습 종단간 테스트 작성
    - 임계 위반 승인 보류, MCP/A1/MongoDB 실패, 배치 버전 불일치, 무부분결과·저장 원자성과 Dashboard 데이터 없음 상태를 자동 검증한다.
    - 신규 ns-3 결과로 오차 초과 → KPI별 재학습 → 신·구 평가 → 승격 또는 유지 → Inference 서빙 버전 확인 흐름을 검증한다.
    - _요구사항: 5.10~5.12, 7.11, 7.12, 10.13, 11.9~11.15, 13.1~13.11, 14.8, 14.9, 15.4~15.7_

  - [ ] 15.5 성능·시간 제한 자동 검증 스위트 작성
    - 1,000,000행 추출 300초, 작업 생성 5초/조회 1초, 모델 로드 60초, 64개 추론 5초, 충돌 200쌍 2000ms, Advisor 60초, A1 상태 5초 등 요구된 SLA를 비감시 단발 테스트로 검증한다.
    - 환경 의존 SLA는 별도 마커로 분리하고 실패 시 측정값과 기준을 출력한다.
    - _요구사항: 1.1, 3.1, 3.3, 5.1, 5.6, 8.9, 9.11, 11.11, 12.3, 12.7, 13.1, 14.1_

- [ ] 16. 최종 체크포인트 - 전체 자동화 테스트 확인
  - 모든 테스트가 통과하는지 확인하고, 질문이 생기면 사용자에게 문의한다.
## Notes

- `*`가 붙은 하위 작업은 빠른 MVP에서 생략할 수 있는 선택적 자동화 테스트 작업이다.
- 모든 구현 작업은 Python을 사용하고 서비스 인터페이스는 설계의 FastAPI/MCP 계약을 따른다.
- 각 Property 작업은 설계의 해당 번호와 요구사항 조항을 태그 주석으로 포함하며 Hypothesis `max_examples=100` 이상을 사용한다.
- 병렬 충돌 방지를 위해 각 Property 및 통합 테스트 작업은 번호별 독립 테스트 모듈을 생성하고, 공통 생성기·픽스처 변경은 1.2에서만 수행한다.
- 체크포인트는 실행 그래프에 포함하지 않으며 각 웨이브 완료 후 자동화 테스트 상태를 확인한다.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "3.1", "7.1"] },
    { "id": 2, "tasks": ["2.2", "3.2"] },
    { "id": 3, "tasks": ["2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "2.13", "2.14", "2.15", "2.16", "3.3", "3.11", "3.12", "3.13", "3.14", "3.15", "3.16", "3.17"] },
    { "id": 4, "tasks": ["2.17", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "3.18"] },
    { "id": 5, "tasks": ["5.1"] },
    { "id": 6, "tasks": ["5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8", "5.9", "5.10", "6.1"] },
    { "id": 7, "tasks": ["6.2", "6.3", "6.4", "6.5", "6.6", "7.2"] },
    { "id": 8, "tasks": ["7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10", "8.1", "8.8"] },
    { "id": 9, "tasks": ["7.11", "7.12", "7.13", "7.14", "7.15", "7.16", "8.2", "8.3", "8.4", "8.5", "8.6", "8.7", "8.9", "8.10", "8.11", "8.12", "8.13", "8.14", "8.15", "8.16", "8.17", "8.18", "9.1"] },
    { "id": 10, "tasks": ["9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8", "9.9", "9.10", "9.11", "11.1", "12.1", "13.1", "14.1"] },
    { "id": 11, "tasks": ["11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8", "11.9", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7", "13.2", "13.3", "13.4", "13.5", "13.6", "13.7", "13.8", "13.9", "13.10", "14.2", "14.3"] },
    { "id": 12, "tasks": ["15.1"] },
    { "id": 13, "tasks": ["15.2", "15.4", "15.5"] },
    { "id": 14, "tasks": ["15.3"] }
  ]
}
```
