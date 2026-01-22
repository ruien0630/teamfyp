from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_docling import DoclingLoader
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores.utils import filter_complex_metadata

import os

def setup_qa_chain(
    local_vector_store_path=None,
    vector_object=None,
    use_local_path=True,
    model_name="llama3.1",
    temperature=0.1,
    embbedings_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    search_type="mmr",
    k=4,
    fetch_k=20,
    lambda_mult=0.5,
    filter_dict=None
):
    """
    Set up a complete RAG chain with Llama 3.1 via Ollama (local).
    Compatible with content-aware splitting chunked vector stores.

    Args:
    
        local_vector_store_path (str, optional): The file path to a local Chroma DB.
                                                 Required if use_local_path is True.
        vector_object (object, optional): A pre-initialized vector store object.
                                          Required if use_local_path is False.
        use_local_path (bool): If True, the function will load the vector store from
                               the local path. If False, it will use the provided
                               vector_object.
        model_name (str): The Ollama model to use (default: "llama3.1").
                         Other options: "llama3.1:70b", "llama3.1:405b"
        temperature (float): Controls randomness (0-1). Lower = more focused.
        embbedings_model_name (str): HuggingFace embedding model name.
        search_type (str): Retrieval algorithm - "similarity" or "mmr" (Maximal Marginal Relevance).
                          MMR balances relevance with diversity. Default: "mmr".
        k (int): Number of documents to return (default: 4).
        fetch_k (int): Number of documents to fetch before MMR reranking (default: 20).
                      Only used when search_type="mmr".
        lambda_mult (float): Balance between relevance (1.0) and diversity (0.0) for MMR.
                            Default: 0.5 (balanced). Only used when search_type="mmr".
        filter_dict (dict): Optional metadata filter for retrieval.
                           Example: {"source": "report.pdf"} to only retrieve from specific source.

    Returns:
        dict: Dictionary with 'retriever' and 'qa_chain' (compatible wrapper).

    Raises:
        ValueError: If required parameters are missing for the chosen retrieval method.
    """
    # Step 1: Configure Llama 3.1 via Ollama (local)
    print(f"Initializing Llama 3.1 model via Ollama: {model_name}")
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
    )

    # Step 2: Create custom prompt template
    prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question. If you cannot find the answer in the context, say "I cannot find this information in the provided documents."

Context: {context}

Question: {question}

Answer: """)
    
    # Step 3: Set up retrieval chain based on the chosen method
    if use_local_path:
        if not local_vector_store_path:
            raise ValueError("`local_vector_store_path` must be provided when `use_local_path` is True.")
        print(f"Loading vector store from local path: {local_vector_store_path}")
        embeddings = HuggingFaceEmbeddings(model_name=embbedings_model_name)
        retriever_source = Chroma(persist_directory=local_vector_store_path,
            embedding_function=embeddings)
    else:
        if vector_object is None:
            raise ValueError("`vector_object` must be provided when `use_local_path` is False.")
        print("Using provided vector store object.")
        retriever_source = vector_object

    # Step 4: Create modern LCEL chain with MMR and filter support
    try:
        # Configure search parameters based on search type
        search_kwargs = {"k": k}
        
        if search_type == "mmr":
            search_kwargs["fetch_k"] = fetch_k
            search_kwargs["lambda_mult"] = lambda_mult
            print(f"Using MMR retrieval: k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
        else:
            print(f"Using similarity search: k={k}")
        
        # Add metadata filter if provided
        if filter_dict:
            search_kwargs["filter"] = filter_dict
            print(f"Applying metadata filter: {filter_dict}")
        
        retriever = retriever_source.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        # Helper function to format docs
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create the LCEL chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✓ QA chain successfully created with Llama 3.1!")
        
        # Return dict with retriever and chain for compatibility
        return {"retriever": retriever, "qa_chain": rag_chain}
    except Exception as e:
        raise Exception(f"Failed to create the RAG chain. Check your vector store and LLM configurations. Details: {e}")

# Example query function
def ask_question(qa_chain_dict, question):
    """
    Query the system and return answer with sources
    
    Args:
        qa_chain_dict: Dictionary with 'retriever' and 'qa_chain'
        question: The question to ask
    """
    # Get chain and retriever from dict
    chain = qa_chain_dict["qa_chain"]
    retriever = qa_chain_dict["retriever"]
    
    # Get answer from chain
    answer = chain.invoke(question)
    
    # Get source documents from retriever
    sources = retriever.invoke(question)
    
    # Format response with citations
    response = {
        "answer": answer,
        "sources": [doc.metadata.get("source", "Unknown") for doc in sources],
        "confidence": len(sources)  # Simple confidence metric
    }
    
    return response

def create_chroma_vectordb_with_content_aware_chunking(
    file_paths: list,
    chroma_db_folder: str = "./chroma_db_content_aware",
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Create a Chroma vector database using Content-Aware (Recursive Character) Splitting.
    
    This is a more intelligent chunking strategy that respects document structure by
    recursively trying different separators (paragraphs, sentences, words) to split text.
    It prioritizes keeping semantically related content together.
    
    Args:
        file_paths (list): List of paths to documents to be processed
        chroma_db_folder (str): Directory to store the Chroma vector database
        model_name (str): HuggingFace model for embeddings
        chunk_size (int): Maximum size of each chunk in characters (default: 1000)
        chunk_overlap (int): Number of overlapping characters between chunks (default: 200)
    
    Returns:
        Chroma: The persistent Chroma vector store with content-aware chunked documents
        
    Pros:
        ✅ Respects document structure - preserves paragraphs and sentences
        ✅ Semantic awareness - keeps related content together
        ✅ Better context preservation - chunks are more coherent
        ✅ Flexible - adapts to different text structures
        ✅ Language-friendly - doesn't split words or sentences mid-way
        
    Cons:
        ❌ Variable chunk sizes - harder to predict memory usage
        ❌ Slightly slower than fixed-size chunking
        ❌ May create very small chunks with short documents
        ❌ Relies on consistent formatting (newlines, punctuation)
    
    Use Cases:
        ✓ Well-structured documents (reports, articles, documentation)
        ✓ When semantic coherence is important
        ✓ Multi-paragraph content
        ✓ Production RAG systems where quality matters
        ✓ Documents with clear section/paragraph boundaries
    
    How it works:
        1. Tries to split on double newlines (paragraphs) first
        2. If chunks too large, splits on single newlines
        3. Then tries sentences (periods, question marks)
        4. Finally splits on spaces (words) if needed
        5. Maintains overlap between chunks for context continuity
    """
    try:
        # Step 1: Load documents using Docling
        print("Loading documents...")
        converter = DocumentConverter()
        loader = DoclingLoader(file_paths, converter=converter)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        
        # Step 2: Initialize embeddings
        print(f"Initializing embeddings with model: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Step 3: Create recursive character text splitter (content-aware)
        print(f"Creating content-aware splitter (chunk_size={chunk_size}, overlap={chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Step 4: Split documents using content-aware method
        print("Splitting documents using content-aware recursive splitting...")
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} content-aware chunks")
        
        # Step 5: Create Chroma vector store
        print(f"Creating vector store at {chroma_db_folder}...")
        if not os.path.exists(chroma_db_folder):
            os.makedirs(chroma_db_folder)
        
        vectorstore = Chroma.from_documents(
            filter_complex_metadata(chunks),
            embeddings,
            persist_directory=chroma_db_folder
        )
        
        print("✓ Content-aware chunking completed successfully!")
        return vectorstore
        
    except Exception as e:
        raise Exception(f"Failed to create vector database with content-aware chunking: {e}")

def main():
    """
    Main function to demonstrate the usage of setup_qa_chain.
    """
    print("--- Example 1: Using a local file path ---")
    try:
        # Note: This will fail if the directory does not exist or is not a valid Chroma DB.
        # Ensure you have a populated './chroma_db' directory.
        qa_chain_from_path = setup_qa_chain(local_vector_store_path="./chroma_db")
        print("Successfully created QA chain from local path.")
        # Example call (requires a real vector store)
        # response = qa_chain_from_path.invoke("What is the main topic?")
        # print(response)
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n--- Example 2: Using a pre-initialized vector store object ---")
    try:
        # This is a placeholder, as you would normally have a real vector store object.
        # This will work as long as it has an as_retriever() method.
        class DummyVectorStore:
            def as_retriever(self, **kwargs):
                return self
        
        qa_chain_from_object = setup_qa_chain(
            vector_object=DummyVectorStore(),
            use_local_path=False
        )
        print("Successfully created QA chain from provided object.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n--- Example 3: Testing with different Ollama model ---")
    try:
        qa_chain_llama = setup_qa_chain(
            local_vector_store_path="./chroma_db",
            model_name="llama3.1"
        )
        print("Successfully created QA chain with Llama 3.1.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    print("\n--- Example 4: Handling missing parameters ---")
    try:
        setup_qa_chain(use_local_path=True, local_vector_store_path=None)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\n--- Example 5: Content-Aware Chunking with Ollama ---")
    try:
        # Create vector DB with content-aware chunking
        file_paths = ["./output_md/example.md"]  # Replace with actual file paths
        vectorstore = create_chroma_vectordb_with_content_aware_chunking(
            file_paths=file_paths,
            chroma_db_folder="./chroma_db_content_aware",
            chunk_size=1000,
            chunk_overlap=200
        )
        print("Successfully created vector store with content-aware chunking!")
        
        # Test with QA chain
        qa_chain = setup_qa_chain(
            local_vector_store_path="./chroma_db_content_aware",
            model_name="llama3.1"
        )
        print("QA chain ready with content-aware chunked data!")
    except Exception as e:
        print(f"Content-aware chunking example error: {e}")

if __name__ == "__main__":
    main()
