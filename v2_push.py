import os
import json
import argparse
from supabase import create_client
from tqdm import tqdm
import requests

# Supabase credentials (hardcoded as requested)
SUPABASE_URL = ""
SUPABASE_KEY = ""

# Define available configurations
CONFIGURATIONS = {
    "4000_200": {
        "table_name": "chunks_4000_200",
        "chunk_size": 4000,
        "overlap": 200,
        "description": "Larger chunks with 4000 characters and 200 overlap"
    },
    "1200_150": {
        "table_name": "chunks_1200_150",
        "chunk_size": 1200,
        "overlap": 150,
        "description": "Smaller chunks with 1200 characters and 150 overlap"
    }
}

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def create_sql_and_print_instructions():
    """Create SQL setup script and print instructions for manual setup."""
    sql_script = """
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for 4000/200 configuration
CREATE TABLE IF NOT EXISTS chunks_4000_200 (
    id UUID PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1024),
    metadata JSONB
);

-- Create vector index for 4000/200 configuration
CREATE INDEX IF NOT EXISTS chunks_4000_200_embedding_idx 
ON chunks_4000_200 
USING ivfflat (embedding vector_cosine_ops);

-- Create table for 1200/150 configuration
CREATE TABLE IF NOT EXISTS chunks_1200_150 (
    id UUID PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1024),
    metadata JSONB
);

-- Create vector index for 1200/150 configuration
CREATE INDEX IF NOT EXISTS chunks_1200_150_embedding_idx 
ON chunks_1200_150 
USING ivfflat (embedding vector_cosine_ops);
"""
    
    print("\n==== MANUAL SETUP INSTRUCTIONS ====")
    print("Since this is a new project, you need to set up the tables manually:")
    print("1. Go to your Supabase dashboard: https://app.supabase.com/")
    print("2. Select your project")
    print("3. Go to the SQL Editor (left sidebar)")
    print("4. Create a new query")
    print("5. Paste the following SQL:")
    print("\n" + sql_script)
    print("\n6. Run the query")
    print("7. After running the SQL successfully, come back and run this script again without the --setup flag")
    print("==================================\n")
    
    # Also save the SQL to a file for convenience
    with open("supabase_setup.sql", "w") as f:
        f.write(sql_script)
    print("The SQL script has also been saved to 'supabase_setup.sql' for your convenience.")
    
    return sql_script

def try_direct_sql_execution(sql_script):
    """Try to execute SQL directly with the REST API."""
    print("Attempting to execute SQL directly via REST API...")
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }
    
    # Splitting into separate statements to improve chances of success
    statements = [s.strip() for s in sql_script.split(';') if s.strip()]
    
    for i, statement in enumerate(statements):
        if not statement.strip():
            continue
            
        print(f"Executing statement {i+1}/{len(statements)}...")
        try:
            response = requests.post(
                f"{SUPABASE_URL}/rest/v1/rpc/exec",
                headers=headers,
                json={"query": statement}
            )
            
            if response.status_code == 200:
                print(f"Statement {i+1} executed successfully")
            else:
                print(f"Error executing statement {i+1}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Exception while executing statement {i+1}: {str(e)}")
            return False
    
    return True

def setup_tables():
    """Set up the necessary tables and extensions in Supabase."""
    print("Setting up Supabase tables...")
    
    # Generate SQL script
    sql_script = create_sql_and_print_instructions()
    
    # Try direct execution
    success = try_direct_sql_execution(sql_script)
    
    if not success:
        print("\nAutomatic setup failed. Please follow the manual setup instructions above.")
        return False
    
    print("\nTables created successfully!")
    return True

def check_table_exists(table_name):
    """Check if a specific table exists in the database."""
    try:
        # Try to query the table
        response = supabase.table(table_name).select("id").limit(1).execute()
        return True
    except Exception as e:
        return False

def upload_chunks(chunks, table_name):
    """Upload chunks to Supabase table in batches."""
    print(f"Uploading {len(chunks)} chunks to {table_name}...")
    batch_size = 5  # Very small batch size to avoid timeouts
    success_count = 0
    
    for i in tqdm(range(0, len(chunks), batch_size), desc=f"Uploading to {table_name}"):
        batch = chunks[i:i + batch_size]
        records = []
        
        for chunk in batch:
            # Extract data
            record = {
                "id": chunk["id"],
                "content": chunk["text"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"]
            }
            records.append(record)
        
        # Insert records
        try:
            response = supabase.table(table_name).insert(records).execute()
            success_count += len(records)
        except Exception as e:
            print(f"Error uploading batch {i//batch_size}: {str(e)}")
            # Try to upload one-by-one if batch fails
            for record in records:
                try:
                    response = supabase.table(table_name).insert([record]).execute()
                    success_count += 1
                except Exception as inner_e:
                    print(f"Error uploading individual record: {str(inner_e)}")
    
    print(f"Upload complete for {table_name}. Successfully uploaded {success_count} chunks.")
    return success_count

def process_folder(input_dir, table_name):
    """Process all JSON files in a directory and upload to Supabase."""
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0
    
    print(f"Found {len(json_files)} JSON files to process")
    total_uploaded = 0
    
    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(input_dir, json_file)
        
        print(f"\nProcessing {json_file}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Validate the chunks
            valid_chunks = []
            for chunk in chunks:
                if "id" in chunk and "text" in chunk and "embedding" in chunk and "metadata" in chunk:
                    valid_chunks.append(chunk)
                else:
                    print(f"Skipping invalid chunk: missing required fields")
            
            print(f"Found {len(valid_chunks)} valid chunks in file")
            
            # Upload chunks
            uploaded_count = upload_chunks(valid_chunks, table_name)
            total_uploaded += uploaded_count
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    return total_uploaded

def main():
    parser = argparse.ArgumentParser(description="Upload embedded chunks to Supabase")
    parser.add_argument("--input-dir", default="embedded_chunks_4000_200", help="Directory containing embedded chunk JSON files")
    parser.add_argument("--config", choices=list(CONFIGURATIONS.keys()), default="4000_200", 
                        help="Configuration to use (determines which table to upload to)")
    parser.add_argument("--setup", action="store_true", help="Set up tables in Supabase before uploading")
    parser.add_argument("--list-configs", action="store_true", help="List available configurations")
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        print("Available configurations:")
        for config_name, config in CONFIGURATIONS.items():
            print(f"  {config_name}: {config['description']}")
            print(f"    - Table: {config['table_name']}")
            print(f"    - Chunk size: {config['chunk_size']}")
            print(f"    - Overlap: {config['overlap']}")
        return
    
    # Set up tables if requested
    if args.setup:
        setup_success = setup_tables()
        if not setup_success:
            print("Please run the SQL commands manually as instructed, then run this script again without the --setup flag.")
            return
    
    # Get table name from configuration
    config = CONFIGURATIONS[args.config]
    table_name = config["table_name"]
    
    # Check if table exists
    table_exists = check_table_exists(table_name)
    if not table_exists:
        print(f"The table '{table_name}' doesn't exist in the database.")
        print("Please run this script with the --setup flag first or create the table manually.")
        return
    
    # Process and upload chunks
    total_uploaded = process_folder(args.input_dir, table_name)
    
    # Print summary
    print("\nUpload process complete!")
    print(f"Configuration: {args.config} ({config['description']})")
    print(f"Total chunks uploaded: {total_uploaded}")
    print(f"Table: {table_name}")
    print(f"Input directory: {args.input_dir}")
    print(f"Supabase project: {SUPABASE_URL}")

if __name__ == "__main__":
    main()