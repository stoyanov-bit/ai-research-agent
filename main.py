from research_agent.agent import run_agent

from research_agent.logging_config import (
    setup_logging,
)

setup_logging()


def main():
    user_input = input("Question: ")

    answer = run_agent(user_input)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()