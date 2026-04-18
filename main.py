from agents import bull_agent, bear_agent, judge_agent


def main():
    # This is the starting point of the program.
    print("Multi-Agent Market Debate System")

    # Ask the user for one simple question.
    question = input("Enter a market question: ")

    # Call the three simple agents.
    bull_output = bull_agent(question)
    bear_output = bear_agent(question)
    judge_output = judge_agent(bull_output, bear_output)

    # Print the answers so we can see the debate.
    print(bull_output)
    print(bear_output)
    print(judge_output)


if __name__ == "__main__":
    main()
