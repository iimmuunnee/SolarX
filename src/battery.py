import numpy as np

class ESSBattery:
    def __init__(self, name, capacity, c_rate, eff, soc_range, eff_is_roundtrip=True):
        self.name = name
        self.capacity = capacity
        self.max_power = capacity * c_rate
        if eff_is_roundtrip:
            one_way = float(np.sqrt(eff))
            self.charge_eff = one_way
            self.discharge_eff = one_way
        else:
            self.charge_eff = eff
            self.discharge_eff = eff
        self.soc_min, self.soc_max = soc_range
        self.current_kwh = capacity * self.soc_min

    def update(self, action, amount_kw, dt_hours=1.0):
        """
        action: 1(충전), -1(방전), 0(대기)
        amount_kw: 요청한 충·방전 전력(kW)
        return: 그리드 거래 전력(+: 방전 판매, -: 충전 구매)
        """
        amount_kw = min(amount_kw, self.max_power)
        actual_trade = 0

        if action == 1:  # 충전
            max_storable = (self.capacity * self.soc_max) - self.current_kwh
            real_in_kwh = min(amount_kw * dt_hours, max_storable)
            self.current_kwh += real_in_kwh * self.charge_eff
            actual_trade = -(real_in_kwh / dt_hours)

        elif action == -1:  # 방전
            max_outable = self.current_kwh - (self.capacity * self.soc_min)
            real_out_kwh = min(max_outable, amount_kw * dt_hours)
            self.current_kwh -= real_out_kwh
            grid_out_kw = (real_out_kwh * self.discharge_eff) / dt_hours
            actual_trade = grid_out_kw

        return actual_trade


class SamsungSDI(ESSBattery):
    def __init__(self, capacity):
        super().__init__(
            name="Samsung SDI (NCA)",
            capacity=capacity,
            c_rate=1.8,
            eff=0.985,
            soc_range=(0.05, 0.95),
            eff_is_roundtrip=True,
        )


class LGEnergySolution(ESSBattery):
    def __init__(self, capacity):
        super().__init__(
            name="LG Energy Solution (NCM)",
            capacity=capacity,
            c_rate=2.0,
            eff=0.980,
            soc_range=(0.05, 0.95),
            eff_is_roundtrip=True,
        )


class TeslaBattery(ESSBattery):
    def __init__(self, capacity):
        super().__init__(
            name="Tesla In-house (4680)",
            capacity=capacity,
            c_rate=1.5,
            eff=0.970,
            soc_range=(0.10, 0.90),
            eff_is_roundtrip=True,
        )
