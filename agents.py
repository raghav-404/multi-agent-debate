def bull_agent(question):
    # Bull always gives a positive view.
    return f"Bull: This looks strong for '{question}'. Buy could work."


def bear_agent(question):
    # Bear always gives a negative view.
    return f"Bear: This has risk for '{question}'. Selling may be safer."


def judge_agent(bull_output, bear_output):
    # Judge picks one simple final answer.
    if len(bull_output) > len(bear_output):
        return "Judge: BUY"
    if len(bear_output) > len(bull_output):
        return "Judge: SELL"
    return "Judge: HOLD"
