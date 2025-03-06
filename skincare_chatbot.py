# -*- coding: utf-8 -*-
"""Final term_SkincareProductChatbot - Team05.py"""

# Step 1: Install necessary libraries
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import gradio as gr
import re
import os

# Step 2: Load the dataset from cloud bucket
file_path = 'https://team5-skincare-chatbot.s3.us-east-1.amazonaws.com/personal_project_2024_db.csv'  

df = pd.read_csv(file_path)

# Step 3: Prepare dataset for embeddings
df["context"] = df.apply(lambda row: f"{row['product_name']} by {row['brand']} is a {row['product_type']}. "
                                     f"Price: ${row['price']}. Ingredients: {row['ingredients']}. "
                                     f"Volume: {row['volume']} {row['volume_unit']}.", axis=1)

# Step 4: Load Pre-trained Sentence Transformer Model
print("Loading sentence transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings for all product descriptions
print("Computing embeddings for products...")
df["embeddings"] = df["context"].apply(lambda x: embedder.encode(x, convert_to_numpy=True))
print("Embeddings computed successfully!")

# Step 5: Define the chatbot function
def chatbot(question):
    # Convert question to embedding
    question_embedding = embedder.encode(question, convert_to_numpy=True)

    # Enhanced Keyword Extraction
    keywords = re.findall(r'\b[A-Za-z]+\b', question)  # Extract words
    keywords = [word.lower() for word in keywords]
    
    # Handle potential NaN values in brand column
    brand_keywords = [word for word in keywords if word in df['brand'].fillna('').str.lower().unique()]
    product_keywords = [word for word in keywords if word in df['product_name'].fillna('').str.lower().unique()]

    # Prioritize Brand and Product Name Matches
    filtered_df = df.copy()  # Start with all products
    
    if brand_keywords or product_keywords:
        filtered_df = df[
            df['brand'].fillna('').str.lower().apply(lambda x: any(word in x for word in brand_keywords)) |
            df['product_name'].fillna('').str.lower().apply(lambda x: any(word in x for word in product_keywords))
        ]

    # Price Range Filtering
    price_match = re.search(r"\$(\d+)-(\d+)", question)  # Search for price range (e.g., $10-20)
    if price_match:
        min_price, max_price = int(price_match.group(1)), int(price_match.group(2))
        filtered_df = filtered_df[(filtered_df['price'] >= min_price) & (filtered_df['price'] <= max_price)]

    if filtered_df.empty:
        # Fallback to ingredient and type matching if no brand/product match
        ingredient_keywords = keywords
        product_type_keywords = ['moisturiser', 'mask', 'peel', 'exfoliator', 'body wash', 'toner', 'serum', 'mist']
        product_type_match = any(word.lower() in question.lower() for word in product_type_keywords)

        filtered_df = df[df['ingredients'].fillna('').str.lower().apply(lambda ing: any(word in ing for word in ingredient_keywords))]

        if product_type_match:
            filtered_df = filtered_df[filtered_df['product_type'].str.contains('|'.join(product_type_keywords), case=False)]

    if filtered_df.empty:
        return "Sorry, I couldn't find any products matching your query. Try another one!"

    # Compute similarity scores for filtered products
    similarities = [util.cos_sim(question_embedding, emb)[0][0] for emb in filtered_df["embeddings"]]

    # Get top 3 similar products
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
    top_products = filtered_df.iloc[top_indices]

    # Generate response with multiple products
    response = f"Okay, here are some products that match your query:\n\n"
    for i, (_, product) in enumerate(top_products.iterrows()):
        response += (
            f"{i + 1}. **{product['product_name']}** by {product['brand']}\n"
            f"   - Type: {product['product_type']}\n"
            f"   - Price: ${product['price']}\n"
            f"   - Ingredients: {product['ingredients']}\n"
            f"   - Volume: {product['volume']} {product['volume_unit']}\n"
            f"   - [Product Link]({product['product_url']})\n\n"
        )

    response += "I hope this helps! Let me know if you have any other questions. 😊"

    return response

# Step 6: Create an interactive UI with Gradio
def main():
    print("Starting Skincare Product Chatbot...")
    iface = gr.Interface(
        fn=chatbot,
        inputs="text",
        outputs="markdown",
        title="Skincare Product Chatbot",
        description="Ask about any skincare product, and I'll provide details"
    )

    # Step 8: Launch the chatbot
    iface.launch(share=True)

# Evaluation function
def evaluate_chatbot():
    print("Evaluating chatbot performance...")
    # Define a test dataset with expected keywords in answers
    test_queries = [
        "Tell me about CeraVe Moisturizer",
        "What is the price of La Roche-Posay sunscreen?",
        "Which product contains Hyaluronic Acid?",
        "List all facial cleansers",
        "Does any product have Retinol?",
    ]

    expected_keywords = [
        ["cerave", "moisturizer", "cream"],
        ["la roche-posay", "sunscreen", "price"],
        ["hyaluronic acid"],
        ["facial", "cleanser"],
        ["retinol"]
    ]

    # Get actual responses
    actual_responses = [chatbot(query) for query in test_queries]

    # Check if expected keywords are in responses
    scores = []
    for response, keywords in zip(actual_responses, expected_keywords):
        response_lower = response.lower()
        score = sum(keyword in response_lower for keyword in keywords) / len(keywords)
        scores.append(score)

    # Display Evaluation Metrics
    avg_score = np.mean(scores)
    print(f"Average Keyword Match Score: {avg_score:.4f}")

    # Print individual scores
    for i, (query, score) in enumerate(zip(test_queries, scores)):
        print(f"Query {i+1}: '{query}' - Score: {score:.2f}")

if __name__ == "__main__":
    # Uncomment the function you want to run
    main()  # Run the chatbot interface
    # evaluate_chatbot()  # Run evaluation
