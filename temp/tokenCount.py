from transformers import GPT2Tokenizer


def count_tokens(query):
    """
    Count the number of tokens in the query
    :param query: the query
    :return: token count
    """
    # Initialize GPT-3.5 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Tokenize the query
    tokens = tokenizer.encode(query, add_special_tokens=False)

    # Count the tokens
    token_count = len(tokens)

    return token_count


def main():
    print("Token counter")
    # Example query
    # query = "How many tokens are in this query?"

    # Request text from user, it should allow for multiple lines, and special characters
    query = input("Enter query: ")

    # Count tokens
    print("Token count:", count_tokens(query))


if __name__ == "__main__":
    main()
