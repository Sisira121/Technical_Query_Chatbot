# Technical Query Chatbot

## Overview
Technical Query Chatbot is an AI-powered chatbot that leverages Retrieval-Augmented Generation (RAG) to provide responses to technical queries. It integrates OpenAI's GPT-3.5-turbo model with FAISS vector search and Streamlit for an interactive UI. The chatbot fetches academic resources, processes PDF documents, and generates responses based on relevant retrieved content.

## Features
- **Retrieval-Augmented Generation (RAG)**: Combines FAISS-based vector search with GPT-3.5 to enhance response accuracy.
- **FAISS Vector Search**: Uses FAISS for efficient similarity search and document retrieval.
- **PDF Text Extraction**: Supports document-based queries by extracting text from uploaded PDFs.
- **Sentence Embeddings**: Utilizes `all-MiniLM-L6-v2` model from SentenceTransformers.
- **Chat History**: Maintains session-based chat history in Streamlit.
- **Model Evaluation**: Implements accuracy, precision, recall, and F1-score metrics for model assessment.
- **Fine-Tuning**: Dynamically optimizes responses based on user queries.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/technical-query-chatbot.git
   cd technical-query-chatbot
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file and add your OpenAI API key:
     ```sh
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage
Run the Streamlit application:
```sh
streamlit run app.py
```

## How It Works
1. **Query Processing**: User inputs a query in the Streamlit UI.
2. **Document Retrieval**: FAISS searches the most relevant academic resources.
3. **Query Enhancement**: The retrieved document is used as context for GPT-3.5.
4. **Response Generation**: GPT-3.5 generates a contextual response.
5. **Chat History Management**: Previous queries and responses are stored.
6. **Evaluation & Fine-Tuning**: The model can be optimized based on feedback.


## Dependencies
- `requests`
- `faiss`
- `numpy`
- `streamlit`
- `sentence-transformers`
- `langchain`
- `PyMuPDF`
- `scikit-learn`

## Contributing
Feel free to fork the repository and submit pull requests with improvements!

## License
This project is licensed under the MIT License.

## Contact
For inquiries, reach out via GitHub issues or email at `sisiras325@gmail.com`. 🚀


