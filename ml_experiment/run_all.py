"""ML 레벨 실험 두 개를 순서대로 실행하고 결과 그래프/요약을 생성한다."""
import exp_image_text
import exp_text_text
import plot_results


def main():
    exp_image_text.main()
    print()
    exp_text_text.main()
    print()
    plot_results.main()


if __name__ == "__main__":
    main()
