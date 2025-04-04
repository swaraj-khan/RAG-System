import os
import re
import json
import argparse
import uuid
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
from ebooklib import epub
from bs4 import BeautifulSoup

# Default parameters
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_OVERLAP = 200
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"  # 1024 dimensions

def extract_text_from_epub(epub_path):
    """Extract text content from an EPUB file."""
    book = epub.read_epub(epub_path)
    chapters = []
    
    # Get book title if available
    book_title = "Unknown"
    for item in book.get_metadata('DC', 'title'):
        if item:
            book_title = item[0]
            break
    
    # Process all items in the EPUB
    for item in book.get_items():
        # Check if the item is HTML content
        if item.media_type == "application/xhtml+xml":
            try:
                # Parse HTML content
                soup = BeautifulSoup(item.content, 'html.parser')
                
                # Extract title if available
                title_tag = soup.find('title')
                chapter_title = title_tag.text if title_tag else "Chapter"
                
                # Extract text content
                text = soup.get_text(separator='\n')
                
                # Clean text (remove multiple newlines, etc.)
                text = re.sub(r'\n+', '\n', text)
                text = text.strip()
                
                if text:  # Only add non-empty chapters
                    chapters.append({
                        "title": chapter_title,
                        "content": text,
                        "book_title": book_title
                    })
            except Exception as e:
                print(f"Error processing item: {str(e)}")
    
    return chapters, book_title

def create_chunks(chapters, chunk_size=4000, overlap=200):
    """Split chapters into chunks of specified size with overlap."""
    chunks = []
    
    for chapter in chapters:
        chapter_title = chapter["title"]
        content = chapter["content"]
        book_title = chapter["book_title"]
        
        # If content is smaller than chunk_size, keep it as one chunk
        if len(content) <= chunk_size:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": content,
                "metadata": {
                    "source": book_title,
                    "chapter": chapter_title
                }
            })
            continue
        
        # Split content into paragraphs
        paragraphs = content.split('\n')
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk and start new one
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": current_chunk,
                    "metadata": {
                        "source": book_title,
                        "chapter": chapter_title
                    }
                })
                
                # Start new chunk with overlap from previous chunk
                overlap_point = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_point:] + "\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n"
                current_chunk += para
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": current_chunk,
                "metadata": {
                    "source": book_title,
                    "chapter": chapter_title
                }
            })
    
    return chunks

def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling to generate sentence embeddings"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def create_embeddings(chunks, model_name=DEFAULT_MODEL, batch_size=8, device=None):
    """Generate embeddings for text chunks using the specified model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Process in batches
    embeddings = []
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        # Apply mean pooling
        batch_embeddings = mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
        
        # Normalize embeddings
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        
        # Convert to list and append
        embeddings.extend(batch_embeddings.cpu().numpy().tolist())
    
    # Add embeddings to chunks
    for i, embedding in enumerate(embeddings):
        chunks[i]["embedding"] = embedding
    
    return chunks

def process_all_epubs(epub_dir, output_dir, model_name=DEFAULT_MODEL, chunk_size=DEFAULT_CHUNK_SIZE, 
                    overlap=DEFAULT_OVERLAP, batch_size=8, combined_output=False):
    """Process all EPUB files in a directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all EPUB files in the directory
    epub_files = [f for f in os.listdir(epub_dir) if f.endswith('.epub')]
    
    if not epub_files:
        print(f"No EPUB files found in {epub_dir}")
        return
    
    print(f"Found {len(epub_files)} EPUB files to process")
    
    all_chunks_with_embeddings = []
    
    # Process each EPUB file
    for epub_file in tqdm(epub_files, desc="Processing EPUB files"):
        epub_path = os.path.join(epub_dir, epub_file)
        
        # Extract text from EPUB
        print(f"\nExtracting text from {epub_file}...")
        chapters, book_title = extract_text_from_epub(epub_path)
        print(f"Extracted {len(chapters)} chapters from {book_title}")
        
        # Create chunks
        print(f"Creating chunks of size {chunk_size} with {overlap} overlap...")
        chunks = create_chunks(chapters, chunk_size, overlap)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        chunks_with_embeddings = create_embeddings(chunks, model_name, batch_size)
        
        # Save individual file results
        if not combined_output:
            # Create a clean filename
            clean_name = re.sub(r'[^\w\-_\. ]', '_', book_title)
            output_file = os.path.join(output_dir, f"{clean_name}_embedded.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(chunks_with_embeddings)} embedded chunks to {output_file}")
        else:
            # Add to combined results
            all_chunks_with_embeddings.extend(chunks_with_embeddings)
    
    # Save combined results if requested
    if combined_output and all_chunks_with_embeddings:
        combined_file = os.path.join(output_dir, "all_books_embedded.json")
        
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks_with_embeddings, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved {len(all_chunks_with_embeddings)} combined embedded chunks to {combined_file}")
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Model used: {model_name}")
    if all_chunks_with_embeddings:
        print(f"Embedding dimensions: {len(all_chunks_with_embeddings[0]['embedding'])}")

def main():
    parser = argparse.ArgumentParser(description="Process EPUB files to chunks with embeddings")
    parser.add_argument("--epub-dir", default="epubbooks", help="Directory containing EPUB files")
    parser.add_argument("--output-dir", default="embedded_chunks", help="Directory to save the output")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model to use for embeddings")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Approximate size of each chunk")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Number of characters to overlap between chunks")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding generation")
    parser.add_argument("--combined", action="store_true", help="Combine all outputs into a single file")
    
    args = parser.parse_args()
    
    process_all_epubs(
        args.epub_dir, 
        args.output_dir, 
        args.model, 
        args.chunk_size, 
        args.overlap, 
        args.batch_size,
        args.combined
    )

if __name__ == "__main__":
    main()