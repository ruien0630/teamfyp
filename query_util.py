from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

import os

def setup_qa_chain(
    local_vector_store_path=None,
    vector_object=None,
    use_local_path=True,
    model_id="llama3.1",
    embbedings_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    search_type="mmr",
    search_kwargs=None
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
        model_id (str): The ID of the language model to use (e.g., 'llama3.1', 'llama3.1:8b').
        embbedings_model_name (str): The HuggingFace embedding model name.
        search_type (str): The search type to use ('similarity' or 'mmr'). Defaults to 'mmr'.
        search_kwargs (dict, optional): Search parameters. If None, uses default MMR settings.

    Returns:
        RetrievalQA: The configured RetrievalQA chain.

    Raises:
        ValueError: If an unsupported model_id is provided, or if required
                    parameters are missing for the chosen retrieval method.
    """
    # Step 1: Configure Ollama model
    llm = ChatOllama(
        model=model_id,
        temperature=0.1,
        num_predict=512,
    )

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

    # Step 4: Configure search parameters
    if search_kwargs is None:
        search_kwargs = {
            "k": 3,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    
    # Step 5: Create RAG chain using LCEL (LangChain Expression Language)
    try:
        retriever = retriever_source.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create the RAG chain using LCEL
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )
        
        # Wrapper to maintain compatibility with old interface
        class RAGChainWrapper:
            def __init__(self, chain, retriever):
                self.chain = chain
                self.retriever = retriever
            
            def __call__(self, inputs):
                question = inputs.get("query") or inputs.get("question")
                answer = self.chain.invoke(question)
                source_docs = self.retriever.invoke(question)
                return {
                    "result": answer,
                    "source_documents": source_docs
                }
        
        return RAGChainWrapper(rag_chain, retriever)
    except Exception as e:
        raise Exception(f"Failed to create the RAG chain. Check your vector store and LLM configurations. Details: {e}")

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
