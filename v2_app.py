import streamlit as st
import anthropic
from supabase import create_client

# Hardcoded credentials
SUPABASE_URL = ""
SUPABASE_KEY = ""
ANTHROPIC_API_KEY = ""

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# System prompt template
SYSTEM_PROMPT = """You are a trading knowledge assistant that ONLY answers questions based on the provided context from trading books.

IMPORTANT RULES:
1. ONLY use information from the provided context to answer questions.
2. If the answer cannot be found in the context, respond with EXACTLY: "Answer not present in books."
3. Do not use any external knowledge or information not provided in the context.
4. Be concise and directly answer the question based on the provided context.
5. If only partial information is available, provide what you can from the context and clarify what aspects are not covered.
6. Do not make up or infer information that isn't explicitly stated in the context.

Context:
{context}
"""

def get_relevant_chunks_by_text(query, top_k=5):
    """Retrieve relevant chunks from Supabase using text search."""
    try:
        # Clean up the query for search
        query_terms = query.strip().lower().split()
        
        # Make a broader initial search to get a pool of potential matches
        # We'll retrieve more than we need and then filter them
        initial_pool_size = min(100, top_k * 10)  # Get 10x what we need or max 100
        
        # Get a pool of documents to rank
        response = supabase.table('general_concepts').select(
            'content, metadata, id'
        ).limit(initial_pool_size).execute()
        
        if not hasattr(response, 'data') or not response.data:
            return []
        
        # Rank documents by keyword matching
        scored_docs = []
        for doc in response.data:
            content = doc.get('content', '').lower()
            # Score is the number of query terms found in the content
            score = sum(1 for term in query_terms if term in content)
            # Boost score for exact phrase matches
            if query.lower() in content:
                score += 5
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and take top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:top_k]]
        
    except Exception as e:
        st.error(f"Error retrieving from database: {str(e)}")
        return []

def get_claude_response(query, context_chunks):
    """Get response from Claude based on the query and context chunks."""
    # Combine chunks into context text
    if not context_chunks:
        return "Answer not present in books."
    
    context_text = "\n\n".join([f"CHUNK {i+1}:\nTitle: {chunk.get('metadata', {}).get('source', 'Unknown')}\nChapter: {chunk.get('metadata', {}).get('chapter', 'Unknown')}\nContent: {chunk.get('content', '')}" 
                            for i, chunk in enumerate(context_chunks)])
    
    # Create the prompt with context
    system_prompt = SYSTEM_PROMPT.format(context=context_text)
    
    try:
        # Get response from Claude
        response = claude.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        return response.content[0].text
    
    except Exception as e:
        st.error(f"Error getting response from Claude: {str(e)}")
        return "Sorry, I encountered an error when generating a response."

# Set up Streamlit UI
st.title("Trading Knowledge Chatbot")
st.markdown("Ask questions about trading concepts from your uploaded books!")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This chatbot answers questions using knowledge from trading books loaded into your Supabase vector database.")
    st.write("It only answers based on information present in the books.")
    
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
    st.info("This version uses text search instead of vector search for better compatibility.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
user_query = st.chat_input("Ask a question about trading...")

# Process the query
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Searching for relevant information...")
        
        # Get relevant chunks using text search
        relevant_chunks = get_relevant_chunks_by_text(user_query, top_k=top_k)
        
        if relevant_chunks:
            message_placeholder.markdown("⚙️ Generating response based on found knowledge...")
            # Display sources if chunks found
            with st.expander("View Source Documents"):
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"**Source {i+1}:** {chunk.get('metadata', {}).get('source', 'Unknown')}")
                    st.markdown(f"**Chapter:** {chunk.get('metadata', {}).get('chapter', 'Unknown')}")
                    st.markdown(f"**Content:** {chunk.get('content', '')}")
                    st.markdown("---")
        else:
            message_placeholder.markdown("No relevant information found in the books.")
        
        # Get response from Claude
        full_response = get_claude_response(user_query, relevant_chunks)
        
        # Update the message with the full response
        message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})