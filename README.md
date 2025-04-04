# 📚 Trading Knowledge RAG System 

A Retrieval-Augmented Generation (RAG) system for trading knowledge, leveraging Supabase vector store and Claude AI to provide accurate answers from your trading books.

## 🌟 Features

- 📖 Automated processing of EPUB trading books
- 🔪 Configurable text chunking with overlap control
- 🧠 State-of-the-art embeddings generation (BAAI/bge-large-en-v1.5)
- 🗄️ Vector search using Supabase PostgreSQL + pgvector
- 💬 AI-powered responses using Claude 3 Haiku
- 🌐 Interactive web interface built with Streamlit

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │    │             │
│  EPUB Books │──▶│  Chunking   │───▶│  Embedding  │──▶│  Supabase   │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                               │
                                                               ▼
                                                        ┌─────────────┐
                                                        │             │
                                                        │ Streamlit   │
                                                        │    App      │
                                                        │             │
                                                        └──────┬──────┘
                                                               │
                                                               ▼
                                                        ┌─────────────┐
                                                        │             │
                                                        │  Claude AI  │
                                                        │             │
                                                        └─────────────┘
```

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Vector Database**: Supabase (PostgreSQL + pgvector)
- **Embedding Model**: BAAI/bge-large-en-v1.5 (1024 dimensions)
- **LLM**: Anthropic Claude 3 Haiku
- **Data Processing**: PyTorch, Transformers, EbookLib

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-knowledge-rag.git
   cd trading-knowledge-rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Supabase:
   - Create a Supabase project at [supabase.com](https://supabase.com)
   - Enable the pgvector extension
   - Set up the necessary tables (use the `--setup` flag with `v2_push.py`)

4. Configure the Anthropic API:
   - Get an API key from [Anthropic](https://anthropic.com)
   - Replace the key in `v2_app.py` or use environment variables

## 📋 Usage

### 1️⃣ Process EPUB Books

The system processes EPUB books by extracting text, chunking it, and generating embeddings:

```bash
python v2_chunking.py --epub-dir ./epubbooks --output-dir ./embedded_chunks --chunk-size 4000 --overlap 200
```

Parameters:
- `--epub-dir`: Directory containing EPUB files (default: "epubbooks")
- `--output-dir`: Directory to save the output JSON files (default: "embedded_chunks")
- `--model`: Hugging Face model to use for embeddings (default: "BAAI/bge-large-en-v1.5")
- `--chunk-size`: Approximate size of each chunk in characters (default: 4000)
- `--overlap`: Number of characters to overlap between chunks (default: 200)
- `--batch-size`: Batch size for embedding generation (default: 8)
- `--combined`: Combine all outputs into a single file

### 2️⃣ Upload to Supabase

Upload the generated chunks to Supabase:

```bash
python v2_push.py --input-dir ./embedded_chunks --config 4000_200 --setup
```

Parameters:
- `--input-dir`: Directory containing embedded chunk JSON files (default: "embedded_chunks_4000_200")
- `--config`: Configuration to use (determines which table to upload to):
  - `4000_200`: Larger chunks with 4000 characters and 200 overlap
  - `1200_150`: Smaller chunks with 1200 characters and 150 overlap
- `--setup`: Set up tables in Supabase before uploading
- `--list-configs`: List available configurations

### 3️⃣ Run the Streamlit App

Start the web interface:

```bash
streamlit run v2_app.py
```

The app provides:
- A chat interface for asking trading-related questions
- Responses based solely on the content of your trading books
- Visibility into the source documents used for each response
- Adjustable settings for retrieval parameters

## 🔍 Technical Details

### Chunking Strategy

The system uses a paragraph-based chunking strategy with overlaps to maintain context between chunks:

```python
if len(current_chunk) + len(para) > chunk_size and current_chunk:
    # Save current chunk
    chunks.append({...})
    
    # Start new chunk with overlap from previous chunk
    overlap_point = max(0, len(current_chunk) - overlap)
    current_chunk = current_chunk[overlap_point:] + "\n" + para
```

### Embedding Generation

Embeddings are generated using mean pooling of token embeddings:

```python
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
```

### Vector Search

The system currently uses basic text search for retrieval, prioritizing matches based on keyword frequency and exact phrase matches:

```python
# Score is the number of query terms found in the content
score = sum(1 for term in query_terms if term in content)
# Boost score for exact phrase matches
if query.lower() in content:
    score += 5
```

### Claude Integration

The system uses Claude 3 Haiku with a specialized system prompt to ensure responses are grounded in the retrieved context:

```python
SYSTEM_PROMPT = """You are a trading knowledge assistant that ONLY answers questions based on the provided context from trading books.
...
"""
```

## 📊 Performance Considerations

- **Memory Usage**: The embedding model (BAAI/bge-large-en-v1.5) requires a GPU with at least 4GB VRAM for optimal performance
- **Batch Size**: Adjust the batch size based on your available memory
- **Chunk Size**: 
  - Larger chunks (4000 chars) provide more context but may reduce precision
  - Smaller chunks (1200 chars) improve precision but may lose broader context
- **Database Query Time**: Performance depends on the size of your vector database and Supabase tier

## 🔐 Security Notes

This example code contains hardcoded API keys and Supabase credentials. In a production environment:

1. Use environment variables for sensitive credentials
2. Implement proper authentication for the Streamlit app
3. Configure role-based access controls in Supabase

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
