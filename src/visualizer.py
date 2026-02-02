import matplotlib.pyplot as plt
import os
import numpy as np

class ReportGenerator:
    def __init__(self):
        if not os.path.exists('images'):
            os.makedirs('images')

    @staticmethod
    def plot_prediction(y_real, y_pred):
        """예측 정확도 그래프"""
        plt.figure(figsize=(15, 6))
        limit = min(300, len(y_real))
        
        plt.plot(y_real[:limit], label='Actual (Real)', color='blue', alpha=0.6)
        plt.plot(y_pred[:limit], label='AI Prediction', color='orange', linestyle='--', linewidth=2)
        
        plt.title('Solar Power Generation Prediction (Test Set)')
        plt.xlabel('Time (Hours)')
        plt.ylabel('Power Generation (kW)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('images/prediction_graph.png')
        plt.close()
        print("   📸 예측 그래프 저장 완료: images/colab_1_prediction.png")

    @staticmethod
    def plot_benchmark(results, baseline):
        """
        [수정] 전체 수익(위) + 순이익 차이(아래)를 하나의 이미지로 통합!
        """
        # 전체 캔버스 크기 설정 (세로로 길게)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        
        colors = ['red', 'blue', 'green']
        
        # -------------------------------------------------------
        # (위쪽) 그래프 1: 전체 누적 수익 (Total Cumulative Profit)
        # -------------------------------------------------------
        ax1.plot(baseline, label='No ESS (Grid Only)', color='black', linestyle='--', linewidth=2)
        
        idx = 0
        for name, history in results.items():
            c = colors[idx % len(colors)]
            ax1.plot(history, label=name, color=c, linewidth=1.5)
            idx += 1
            
        ax1.set_title('Global Battery Benchmark: Cumulative Profit (Total)', fontsize=14)
        ax1.set_ylabel('Total Profit (KRW)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # -------------------------------------------------------
        # (아래쪽) 그래프 2: 순이익 차이 (Net Profit Gain)
        # -------------------------------------------------------
        idx = 0
        for name, history in results.items():
            c = colors[idx % len(colors)]
            # 핵심: (배터리 수익 - 기준 수익)
            gain = np.array(history) - np.array(baseline)
            ax2.plot(gain, label=f"{name} (Net Gain)", color=c, linewidth=2)
            idx += 1
            
        ax2.set_title('Net Profit Gain (Difference View)', fontsize=14)
        ax2.set_xlabel('Time (Hours)', fontsize=12)
        ax2.set_ylabel('Additional Profit (KRW)', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 그래프 간격 조정 및 저장
        plt.tight_layout()
        save_path = 'images/benchmark_graph.png'
        plt.savefig(save_path)
        plt.close()
        print(f"   📸 통합 수익 그래프 저장 완료: {save_path}")