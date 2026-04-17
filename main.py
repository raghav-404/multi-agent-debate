from agents import bull, bear, judge
from graph import build_graph
from memory import load_memory, save_memory
from tools import ask_ollama


def main():
    # This is the starting point of the program.
    print("Multi-Agent Market Debate System")

    # Ask the user for one simple question.
    question = input("Enter a market question: ")

    # Call small placeholder parts of the project.
    build_graph()
    load_memory()
    ask_ollama("test prompt")

    # Show the three agent roles.
    print(bull(question))
    print(bear(question))
    print(judge(question))

    # Save a tiny placeholder memory item.
    save_memory([question])


if __name__ == "__main__":
    main()
