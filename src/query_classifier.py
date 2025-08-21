
import logging

# Configure logging in the query classifier
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def classify_query(query):       #function to classify user queries
    """
    Classifies a user query into one of several categories based on keywords.

    Categories:
    - Factual: For direct questions seeking facts.
    - Comparative: For questions that compare two or more entities.
    - Analytical: For questions that require reasoning or finding patterns.
    - Non-Sport: For questions outside the sports domain.
    """
    query_lower = query.lower()     # Convert query to lowercase for uniformity

    # Keywords for Comparative analysis
    comparative_keywords = ['vs', 'versus', 'compare', 'better than', 'more than']
    if any(keyword in query_lower for keyword in comparative_keywords):
        logging.info(f"Query classified as: Comparative")
        return "Comparative"

    # Keywords for Analytical/Reasoning questions
    analytical_keywords = ['why', 'how', 'what is the strategy', 'explain the tactic']
    if any(keyword in query_lower for keyword in analytical_keywords):
        logging.info(f"Query classified as: Analytical")
        return "Analytical"
        
    # Keywords for Factual questions (often start with "wh-")
    # This is a general catch-all for direct questions after the more specific ones.
    factual_keywords = ['who', 'what', 'when', 'where', 'list', 'define']
    if any(query_lower.startswith(keyword) for keyword in factual_keywords):
        logging.info(f"Query classified as: Factual")
        return "Factual"
        
    # Keywords to detect non-sports queries
    # This list should be expanded for better accuracy.
    non_sport_keywords = ['movie', 'politics', 'stock market', 'cooking', 'music']
    if any(keyword in query_lower for keyword in non_sport_keywords):
        logging.info(f"Query classified as: Non-Sport")
        return "Non-Sport"

    # If no other category fits, default to Factual
    # This is a safe assumption for a specialized chatbot.
    logging.info(f"Query could not be specifically classified, defaulting to: Factual")
    return "Factual"

if __name__ == '__main__':
    # This is for testing the script directly
    print("--- Testing Query Classifier ---")

    queries_to_test = [     # List of queries to test the classifier
        "What is the offside rule in football?",
        "Compare Messi vs Ronaldo career statistics",
        "Why do teams use a 4-4-2 formation?",
        "Who has won the most World Cups?",
        "Tell me about the stock market."
    ]

    for q in queries_to_test:      # Iterate over each query
        category = classify_query(q)
        print(f"Query: '{q}'\n  -> Category: {category}\n")    # Print the classified category