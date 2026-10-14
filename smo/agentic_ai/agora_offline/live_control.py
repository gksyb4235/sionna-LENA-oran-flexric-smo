"""Explicit operator-triggered bridge to the local real RIC gateway."""

import os
import httpx
from fastapi import HTTPException

BASE = os.environ.get("AGORA_RIC_GATEWAY_URL", "http://127.0.0.1:18082").rstrip("/")


A1_BASE = os.environ.get("AGORA_A1_MEDIATOR", "http://127.0.0.1:9000").rstrip("/")


async def request(method, path, body=None, base=None):
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            response = await client.request(method, (base or BASE) + path, json=body)
    except httpx.HTTPError as exc:
        raise HTTPException(
            503,
            "RIC 게이트웨이에 연결할 수 없습니다. python scripts/agora.py gateway로 시작하세요.",
        ) from exc
    try:
        data = response.json()
    except ValueError:
        raise HTTPException(502, "RIC 게이트웨이가 잘못된 응답을 반환했습니다.")
    if response.is_error:
        raise HTTPException(response.status_code, data.get("detail", "RIC 요청 실패"))
    return data


async def deploy(receipt, run_id):
    route = receipt["target_ric"]
    if route == "near_rt":
        method, path = "POST", "/api/near-rt/deployments"
    elif route == "non_rt":
        method, path = "PUT", f"/a1-p/policytypes/401/policies/{receipt['deployment_id']}"
    else:
        raise HTTPException(
            422,
            "실제 적용에서는 Near-RT 또는 Non-RT 한 경로를 선택하세요. 분리 배포는 오프라인 대기열에서 지원합니다.",
        )
    return await request(
        method,
        path,
        {"plan": receipt, "live_run_id": run_id},
        base=A1_BASE if route == "non_rt" else BASE,
    )


async def deployment_status(receipt):
    ident = receipt["deployment_id"]
    if receipt["target_ric"] == "non_rt":
        try:
            return await request(
                "GET", f"/a1-p/policytypes/401/policies/{ident}/status", base=A1_BASE
            )
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            # Read historical gateway receipts without bypassing new A1 submissions.
    return await request("GET", "/api/deployments/" + ident)
