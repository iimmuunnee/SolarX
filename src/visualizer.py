import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

class ReportGenerator:
    @staticmethod
    def save_plots(y_real, y_pred, results, baseline):
        if not os.path.exists('./images'):
            os.makedirs('./images')

        plt.rcParams['font.family'] = 'Malgun Gothic' 
        plt.rcParams['axes.unicode_minus'] = False

        # --- [1] 발전량 예측 그래프 ---
        plt.figure(figsize=(14, 6))
        plt.plot(y_real[:200], label='Actual', color='grey', alpha=0.5, linewidth=2)
        plt.plot(y_pred[:200], label='AI Prediction', color='red', linestyle='--', linewidth=2)
        plt.title('Solar Generation Prediction (Zoom-in)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('./images/prediction_graph.png')
        plt.close()

        # --- [2] 수익성 비교 그래프 (Main + Sub) ---
        # 위아래로 2단 분리
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
        
        # [상단] 전체 누적 수익 (Cumulative Profit)
        ax1.plot(baseline, label='Baseline (No ESS)', color='black', linestyle='--', linewidth=2)
        
        colors = {'LG Energy Solution': 'red', 'Samsung SDI': 'blue', 'Tesla In-house': 'green'}
        for name, history in results.items():
            color = next((v for k, v in colors.items() if k in name), None)
            ax1.plot(history, label=name, color=color, linewidth=1.5)
            
        ax1.set_title('Cumulative Profit (Overall)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Profit (KRW)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # [하단] 순수익 차이 확대 (Net Gain Analysis)
        # Baseline을 0으로 잡고, 얼마나 더 벌었는지(Gain)만 시각화
        ax2.axhline(0, color='black', linestyle='--', linewidth=1) # 기준선
        
        for name, history in results.items():
            # (배터리 수익 - 기준 수익) 계산
            gain = [h - b for h, b in zip(history, baseline)]
            color = next((v for k, v in colors.items() if k in name), None)
            ax2.plot(gain, label=f"{name} (Gain)", color=color, linewidth=2)
            
            # 마지막 값에 텍스트 표시
            final_gain = gain[-1]
            ax2.text(len(gain)-1, final_gain, f"+{int(final_gain):,} Won", 
                     fontsize=10, color=color, fontweight='bold', ha='left')

        ax2.set_title('Net Profit Gain (Baseline Removed)', fontsize=14, fontweight='bold', color='darkred')
        ax2.set_xlabel('Time (Hour)')
        ax2.set_ylabel('Additional Profit (KRW)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./images/benchmark_graph.png')
        plt.close()
        
        print("💰 수익 그래프 저장 완료 (Subplot 포함): ./images/benchmark_graph.png")
    @staticmethod
    def save_scalability_plot(scenarios_data):
        plt.figure(figsize=(12, 6))
        
        styles = {
            'Donghae (Base)': {'color': 'blue', 'style': '-'},
            'Jeju (High Solar)': {'color': 'red', 'style': '-'},
            'Seattle (Low Solar)': {'color': 'grey', 'style': '-'}
        }
        
        for name, profit_history in scenarios_data.items():
            style = styles.get(name, {'color': 'black', 'style': '-'})
            plt.plot(profit_history, label=name, color=style['color'], linestyle=style['style'], linewidth=2)
            
            # 최종 수익 표시
            final_val = profit_history[-1]
            plt.text(len(profit_history)-1, final_val, f"{int(final_val):,} Won", 
                    color=style['color'], fontweight='bold', ha='left')

        plt.title('Scalability Test: Robustness Across Locations', fontsize=15, fontweight='bold')
        plt.xlabel('Time (Hour)')
        plt.ylabel('Cumulative Profit (KRW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = './images/scalability_graph.png'
        plt.savefig(save_path)
        plt.close()
        print(f"🌍 확장성 그래프 저장 완료: {save_path}")