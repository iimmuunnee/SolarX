"""
낙관 편향(optimism bias) 재현 스크립트.

같은 벤치마크를 두 번 돌린다. 한 번은 가상 가격(실측 SMP를 넣기 전의 가격
가정), 한 번은 KPX 실측 SMP. 각각 3사와 기준선(ESS 없음)의 누적 수익과
기준선 대비 개선율을 출력해, "가상에서 크게 나오던 개선율이 실측에서 얼마로
떨어지는가"를 재현 가능한 형태로 보여준다.

핵심 원칙: 계산 로직은 건드리지 않는다. SimulationService를 그대로 쓰고
가격 배열(self.real_prices)만 갈아끼운다. src/, main.py, backend/app/services/는
변경 없이 이 스크립트 하나만 추가된다.

실행:  python scripts/bias_check.py
"""
import os
import sys

# Windows 콘솔(cp949 등)에서도 한글·기호(—, →, ·)가 깨지거나 크래시하지 않도록
# 출력 인코딩을 UTF-8로 고정한다.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from backend.app.services.simulation_service import SimulationService  # noqa: E402
from backend.app.schemas.requests import BenchmarkRequest  # noqa: E402

# 화면에 표시할 벤더 순서(최종 수익 상위 → 하위)
VENDOR_ROW = [("samsung", "samsung"), ("lg", "lg"), ("tesla", "tesla")]

# fallbackResults.ts에서 직접 계산된 실측 기준 값. 실측 줄은 이 값과 일치해야
# 한다(앱이 화면에 보여주는 것과 같은 수치인지 검증하는 앵커).
EXPECTED_REAL = {
    "baseline": 56_883_301,
    "samsung": 59_039_576,
    "lg": 58_989_484,
    "tesla": 58_665_879,
}


def run_scenario(svc: SimulationService, prices: np.ndarray):
    """가격 배열만 갈아끼워 벤치마크를 돌리고 (기준선, {벤더: (수익, 개선율%)}) 반환."""
    svc.real_prices = prices
    resp = svc.run_benchmark(BenchmarkRequest())
    base = resp.baseline.revenue_krw
    by_vendor = {
        v.vendor_id: (v.revenue_krw, (v.revenue_krw - base) / base * 100.0)
        for v in resp.vendors
    }
    return base, by_vendor


def fmt_row(label: str, base: float, by_vendor: dict) -> str:
    cells = "   ".join(
        f"{name} {by_vendor[vid][1]:+6.2f}%" for vid, name in VENDOR_ROW
    )
    return f"{label:<30} {cells}   (baseline {int(base):,})"


def main() -> int:
    svc = SimulationService()
    svc.initialize()

    real_prices = np.asarray(svc.real_prices, dtype=float)
    # simulation_service.py:97의 가상 가격 가정: 낮(10~16시) 100원, 그 외 200원.
    # 실측 SMP를 넣기 전 코드에 남아 있는 가격 곡선이다.
    virtual_prices = np.array(
        [100 if 10 <= i % 24 <= 16 else 200 for i in range(len(svc.y_real_kw))],
        dtype=float,
    )

    n = len(real_prices)
    base_v, virt = run_scenario(svc, virtual_prices)
    base_r, real = run_scenario(svc, real_prices)
    svc.real_prices = real_prices  # 원복

    print("=" * 78)
    print("낙관 편향 재현 — 가상 가격 vs KPX 실측 SMP (동일 벤치마크, 가격만 교체)")
    print(f"평가 구간 {n}시간({n / 24:.0f}일) 테스트 분할 · 용량 2,280 kWh · allow_grid_charge=True")
    print("=" * 78)
    print(fmt_row("가상 가격 (낮 100 / 그 외 200)", base_v, virt))
    print(fmt_row(f"KPX 실측 SMP ({n}h 테스트셋)", base_r, real))
    print()

    # 상세 수익액
    print("[상세] 누적 수익 (KRW)")
    print(f"  {'':10} {'가상 가격':>16} {'실측 SMP':>16}")
    print(f"  {'baseline':10} {int(base_v):>16,} {int(base_r):>16,}")
    for vid, name in VENDOR_ROW:
        print(f"  {name:10} {int(virt[vid][0]):>16,} {int(real[vid][0]):>16,}")
    print()

    # 편향 요약(포트폴리오가 주장하는 쌍의 실제 값)
    print("[낙관 편향 쌍] samsung 개선율")
    print(f"  가상 가격: {virt['samsung'][1]:+.2f}%   →   실측 SMP: {real['samsung'][1]:+.2f}%")
    print()

    # 검증: 실측 줄이 fallbackResults.ts와 일치하는가
    print("[검증] 실측 수익이 fallbackResults.ts 값과 일치하는지")
    ok = True
    checks = [("baseline", base_r)] + [(vid, real[vid][0]) for vid, _ in VENDOR_ROW]
    for key, actual in checks:
        expected = EXPECTED_REAL[key]
        # 반올림/부동소수 오차 허용: 1,000원 이내면 일치로 본다
        match = abs(actual - expected) <= 1_000
        ok = ok and match
        mark = "OK" if match else "MISMATCH"
        print(f"  {key:10} 실측 {int(actual):>14,}  기대 {expected:>14,}  [{mark}]")
    print()
    if ok:
        print("결과: 실측 줄이 fallbackResults.ts와 일치합니다.")
        return 0
    print("결과: 불일치. 여기서 멈추고 원인을 확인해야 합니다.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
