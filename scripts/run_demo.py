"""Entry point for the Financial Research Assistant demo."""

from src.pipeline import run_pipeline

SAMPLE_QUESTION = (
    "What are the key risks for US technology equities given interest rates and AI growth trends?"
)


def main() -> None:
    response = run_pipeline(SAMPLE_QUESTION)
    print(response)


if __name__ == "__main__":
    main()
