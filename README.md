# Skincare Product Chatbot

A chatbot that provides information about skincare products based on user queries.

## Setup

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
3. Install dependencies: `sh install.sh`
4. Run the chatbot: `python3.11 skincare_chatbot.py`

## Features

- Search for products by brand, name, or ingredients
- Filter by price range
- Get detailed product information


## Key Changes Made:
Added proper package installation handling - install.sh

Created an install_package function that checks if packages are installed and installs them if needed
This is more appropriate for a standalone Python script

Updated SentenceTransformer model: Changed from "distilbert-base-nli-mean-tokens" (which may not be available) to "all-MiniLM-L6-v2" which is a standard model.

Fixed embedding handling:

Changed to convert_to_numpy=True instead of convert_to_tensor=True for better storage in the DataFrame
Updated similarity calculation to work with numpy arrays
Added error handling for NaN values: Used fillna('') to prevent errors when searching in text fields.

Fixed the evaluation section:

Implemented a keyword-based evaluation that checks if expected keywords appear in responses
Added proper scoring and reporting
Fixed indexing in response generation: Changed from using absolute DataFrame indices to enumeration.

Improved case handling: Made keyword matching case-insensitive throughout.

Added main() and evaluate_chatbot() functions
Added if __name__ == "__main__": block to make the script executable
Added progress messages:

Added print statements to show progress during embedding computation and startup
Made evaluation optional:

Separated the evaluation into its own function that can be called separately
This structure is much more appropriate for a standalone Python script rather than a Jupyter notebook. The user can now run the script directly with python skincare_chatbot.py and it will launch the Gradio interface.