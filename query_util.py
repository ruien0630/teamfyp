from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv

import os

# Load environment variables from .env file
load_dotenv(override=True)

def setup_qa_chain(
    local_vector_store_path=None,
    vector_object=None,
    use_local_path=True,
    model_id="gemini-2.0-flash",
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
        model_id (str): The ID of the language model to use. Supported models:
                        'gemini-1.5-flash' (default, higher quota), 'gemini-1.5-pro', 
                        or 'gemini-2.0-flash-exp'.

    Returns:
        RetrievalQA: The configured RetrievalQA chain.

    Raises:
        ValueError: If an unsupported model_id is provided, or if required
                    parameters are missing for the chosen retrieval method.
    """
    # Handle exceptions for unsupported models
    if model_id in ["gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]:
        # Step 1: Configure Google Gemini model
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel(model_id)
    elif model_id == "ibm/granite-3-8b-instruct":
        # Fallback to IBM Watson (requires langchain-ibm)
        raise ValueError("IBM Watson integration requires langchain-ibm which has dependency conflicts. Please use 'gemini-2.0-flash' instead.")
    else:
        raise ValueError("Only Gemini models (gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp) are currently supported.")

    # Step 2: Create custom prompt template
    prompt_template = """Use the following context to answer the question. If you cannot find the answer in the context, say "I cannot find this information in the provided documents."

Context: {context}

Question: {question}

Answer:"""
    
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

    # Step 4: Create a custom QA chain object
    try:
        retriever = retriever_source.as_retriever(search_kwargs={"k": 4})
        
        # Create a simple class to wrap the retriever and model
        class SimpleQAChain:
            def __init__(self, llm, retriever, prompt_template):
                self.llm = llm
                self.retriever = retriever
                self.prompt_template = prompt_template
            
            def invoke(self, question):
                # Get relevant documents
                docs = self.retriever.invoke(question)
                context = "\n\n".join(doc.page_content for doc in docs)
                
                # Format prompt
                prompt = self.prompt_template.format(context=context, question=question)
                
                # Generate response
                response = self.llm.generate_content(prompt)
                return {
                    "answer": response.text,
                    "source_documents": docs
                }
        
        qa_chain = SimpleQAChain(llm, retriever, prompt_template)
        return qa_chain
    except Exception as e:
        raise Exception(f"Failed to create the QA chain. Check your vector store and LLM configurations. Details: {e}")

# Example query function
def ask_question(qa_chain, question):
    """
    Query the system and return answer with sources
    """
    # Get the answer from the chain
    result = qa_chain.invoke(question)
    
    # Format response with citations
    response = {
        "answer": result["answer"],
        "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]],
        "confidence": len(result["source_documents"])  # Simple confidence metric
    }
    
    return response

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

if __name__ == "__main__":
    main()
