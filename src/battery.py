import numpy as np

class ESSBattery:
    def __init__(self, name, capacity, c_rate, eff, soc_range):
        self.name = name
        self.capacity = capacity
        self.max_power = capacity * c_rate
        self.eff = eff
        self.soc_min, self.soc_max = soc_range
        self.current_kwh = capacity * self.soc_min # 초기 상태

    def update(self, action, amount_kw):
        """
        action: 1(충전), -1(방전), 0(대기)
        amount_kw: 발전량 또는 필요 전력량
        return: 실제 그리드와 거래된 전력량 (+: 방전 판매, -: 충전 구매)
        """
        amount_kw = min(amount_kw, self.max_power) # 물리적 속도 제한
        actual_trade = 0 

        if action == 1: # 충전
            max_storable = (self.capacity * self.soc_max) - self.current_kwh
            real_in = min(amount_kw, max_storable)
            self.current_kwh += real_in * self.eff 
            actual_trade = -real_in 

        elif action == -1: # 방전
            max_outable = self.current_kwh - (self.capacity * self.soc_min)
            real_out = min(max_outable, self.max_power)
            self.current_kwh -= real_out
            grid_out = real_out * self.eff 
            actual_trade = grid_out 
            
        return actual_trade

# 자식 클래스들 (구체적인 스펙 정의)
class SamsungSDI(ESSBattery):
    def __init__(self, capacity):
        super().__init__(
            name="Samsung SDI (NCA)", 
            capacity=capacity, 
            c_rate=1.8, 
            eff=0.985, 
            soc_range=(0.05, 0.95)
        )

class LGEnergySolution(ESSBattery):
    def __init__(self, capacity):
        super().__init__(
            name="LG Energy Solution (NCM)", 
            capacity=capacity, 
            c_rate=2.0, 
            eff=0.980, 
            soc_range=(0.05, 0.95)
        )

class TeslaBattery(ESSBattery):
    def __init__(self, capacity):
        super().__init__(
            name="Tesla In-house (4680)", 
            capacity=capacity, 
            c_rate=1.5, 
            eff=0.970, 
            soc_range=(0.10, 0.90)
        )