import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm


class ReportGenerator:
    @staticmethod
    def save_plots(y_real, y_pred, results, baseline):
        if not os.path.exists("./images"):
            os.makedirs("./images")

        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

        # [1] 예측 확대 그래프 (Zoom-in)
        plt.figure(figsize=(14, 6))
        plt.plot(y_real[:200], label="Actual", color="grey", alpha=0.5, linewidth=2)
        plt.plot(y_pred[:200], label="AI Prediction", color="red", linestyle="--", linewidth=2)
        plt.title("Solar Generation Prediction (확대)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("./images/prediction_graph.png")
        plt.close()

        # [2] 누적 수익 + 순이익 (Cumulative + Net)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1]})

        ax1.plot(baseline, label="Baseline (ESS 없음)", color="black", linestyle="--", linewidth=2)

        colors = {"LG Energy Solution": "red", "Samsung SDI": "blue", "Tesla In-house": "green"}
        for name, history in results.items():
            color = next((v for k, v in colors.items() if k in name), None)
            ax1.plot(history, label=name, color=color, linewidth=1.5)

        ax1.set_title("Cumulative Profit (전체)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Profit (KRW)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2.axhline(0, color="black", linestyle="--", linewidth=1)

        gains_by_name = {}
        for name, history in results.items():
            gain = [h - b for h, b in zip(history, baseline)]
            gains_by_name[name] = gain[-1] if gain else 0
            color = next((v for k, v in colors.items() if k in name), None)
            ax2.plot(gain, label=f"{name} (Gain)", color=color, linewidth=2)

        ax2.set_title("Net Profit Gain (기준선 제거)", fontsize=14, fontweight="bold", color="darkred")
        ax2.set_xlabel("Time (Hour)")
        ax2.set_ylabel("Additional Profit (KRW)")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        # 오른쪽 하단에 요약 수치 표시 (선 옆 텍스트 대신)
        profit_lines = [f"Baseline: {int(baseline[-1]):,} KRW"]
        for name, history in results.items():
            profit_lines.append(f"{name}: {int(history[-1]):,} KRW")
        profit_text = "\n".join(profit_lines)
        ax1.text(
            0.98,
            0.02,
            profit_text,
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
        )

        # 하단 우측 요약: 선 색상과 동일하게 표시, 부호 포함
        y_start = 0.12
        y_step = 0.05
        ax2.text(
            0.98,
            0.02,
            "",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
        )
        sorted_gains = sorted(gains_by_name.items(), key=lambda x: x[1], reverse=False)
        for idx, (name, gain) in enumerate(sorted_gains):
            color = next((v for k, v in colors.items() if k in name), "black")
            ax2.text(
                0.98,
                y_start + (y_step * idx),
                f"{name}: {int(gain):+,.0f} KRW",
                transform=ax2.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color=color,
            )

        plt.tight_layout()
        plt.savefig("./images/benchmark_graph.png")
        plt.close()

        print("Profit plot 저장 완료: ./images/benchmark_graph.png")

    @staticmethod
    def save_scalability_plot(scenarios_data):
        plt.figure(figsize=(12, 6))

        styles = {
            "Donghae (Base)": {"color": "blue", "style": "-"},
            "Jeju (High Solar)": {"color": "red", "style": "-"},
            "Seattle (Low Solar)": {"color": "grey", "style": "-"},
        }

        for name, profit_history in scenarios_data.items():
            style = styles.get(name, {"color": "black", "style": "-"})
            plt.plot(
                profit_history,
                label=name,
                color=style["color"],
                linestyle=style["style"],
                linewidth=2,
            )

            final_val = profit_history[-1]
            plt.text(
                len(profit_history) - 1,
                final_val,
                f"{int(final_val):,} Won",
                color=style["color"],
                fontweight="bold",
                ha="left",
            )

        plt.title("Scalability Test: 지역별 안정성", fontsize=15, fontweight="bold")
        plt.xlabel("Time (Hour)")
        plt.ylabel("Cumulative Profit (KRW)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = "./images/scalability_graph.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Scalability plot 저장 완료: {save_path}")
