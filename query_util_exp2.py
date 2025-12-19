from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_docling import DoclingLoader
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores.utils import filter_complex_metadata

import os

def setup_qa_chain(
    local_vector_store_path=None,
    vector_object=None,
    use_local_path=True,
    model_id="ibm/granite-3-8b-instruct",
    embbedings_model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Set up a complete RAG chain with a Chroma vector store.

    Args:
        local_vector_store_path (str, optional): The file path to a local Chroma DB.
                                                 Required if use_local_path is True.
        vector_object (object, optional): A pre-initialized vector store object.
                                          Required if use_local_path is False.
        use_local_path (bool): If True, the function will load the vector store from
                               the local path. If False, it will use the provided
                               vector_object.
        model_id (str): The ID of the language model to use. Currently, only
                        'ibm/granite-3.1-8b-instruct' is supported.

    Returns:
        RetrievalQA: The configured RetrievalQA chain.

    Raises:
        ValueError: If an unsupported model_id is provided, or if required
                    parameters are missing for the chosen retrieval method.
    """
    # Handle exceptions for unsupported models
    if model_id == "ibm/granite-3-8b-instruct":
        # Step 1: Configure IBM Granite model
        llm = WatsonxLLM(
            url="https://us-south.ml.cloud.ibm.com",
            apikey=os.environ.get("WATSONX_APIKEY"),
            project_id=os.environ.get("WATSONX_PROJECT_ID"),
            model_id=model_id,
            params={
                "temperature": 0.1,
                "max_new_tokens": 512,
                "repetition_penalty": 1.1
            }
        )
    else:
        raise ValueError("Only 'ibm/granite-3-8b-instruct' is currently supported.")

    # Step 2: Create custom prompt template
    prompt_template = """
    Use the following context to answer the question. If you cannot find the answer in the context, say "I cannot find this information in the provided documents."

    Context: {context}

    Question: {question}

    Answer: """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
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

    # Step 4: Add error handling for the RetrievalQA chain creation
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_source.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        raise Exception(f"Failed to create the RetrievalQA chain. Check your vector store and LLM configurations. Details: {e}")

# Example query function
def ask_question(qa_chain, question):
    """
    Query the system and return answer with sources
    """
    result = qa_chain({"query": question})
    
    answer = result["result"]
    sources = result["source_documents"]
    
    # Format response with citations
    response = {
        "answer": answer,
        "sources": [doc.metadata.get("source", "Unknown") for doc in sources],
        "confidence": len(sources)  # Simple confidence metric
    }
    
    return response

def create_chroma_vectordb_with_semantic_chunking(
    file_paths: list,
    chroma_db_folder: str = "./chroma_db_semantic",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    breakpoint_threshold_type: str = "percentile"
):
    """
    Create a Chroma vector database using semantic chunking.
    
    Semantic chunking splits documents based on semantic similarity rather than 
    character count or structure. This keeps topic coherence intact and is ideal 
    for deep analysis like summarization or classification.
    
    Args:
        file_paths (list): List of paths to documents to be processed
        chroma_db_folder (str): Directory to store the Chroma vector database
        model_name (str): HuggingFace model for embeddings
        breakpoint_threshold_type (str): Method for determining chunk boundaries.
            Options: "percentile", "standard_deviation", "interquartile"
            Default: "percentile"
    
    Returns:
        Chroma: The persistent Chroma vector store with semantically chunked documents
        
    Pros:
        ✅ Keeps topic coherence intact
        ✅ Ideal for deep analysis like sentiment or topic modeling
        ✅ Better context preservation than character-based splitting
        
    Cons:
        ❌ Computationally expensive — requires embedding models
        ❌ Slower than simple character-based splitting
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
        
        # Step 3: Create semantic chunker
        print(f"Creating semantic chunker with {breakpoint_threshold_type} threshold...")
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type
        )
        
        # Step 4: Split documents semantically
        print("Splitting documents using semantic chunking (this may take a while)...")
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} semantic chunks")
        
        # Step 5: Create Chroma vector store
        print(f"Creating vector store at {chroma_db_folder}...")
        if not os.path.exists(chroma_db_folder):
            os.makedirs(chroma_db_folder)
        
        vectorstore = Chroma.from_documents(
            filter_complex_metadata(chunks),
            embeddings,
            persist_directory=chroma_db_folder
        )
        
        print("✓ Semantic chunking completed successfully!")
        return vectorstore
        
    except Exception as e:
        raise Exception(f"Failed to create vector database with semantic chunking: {e}")

def main():
    """
    Main function to demonstrate the usage of setup_qa_chain with semantic chunking.
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

    print("\n--- Example 3: Handling an unsupported model ID ---")
    try:
        setup_qa_chain(model_id="unsupported_model")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\n--- Example 4: Handling missing parameters ---")
    try:
        setup_qa_chain(use_local_path=True, local_vector_store_path=None)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\n--- Example 5: Semantic Chunking ---")
    try:
        # Create vector DB with semantic chunking
        file_paths = ["./output_md/example.md"]  # Replace with actual file paths
        vectorstore = create_chroma_vectordb_with_semantic_chunking(
            file_paths=file_paths,
            chroma_db_folder="./chroma_db_semantic"
        )
        print("Successfully created vector store with semantic chunking!")
    except Exception as e:
        print(f"Semantic chunking example error: {e}")

if __name__ == "__main__":
    main()
