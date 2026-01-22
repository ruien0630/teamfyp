from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA #from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate#from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

def setup_qa_chain(
    local_vector_store_path=None,
    vector_object=None,
    use_local_path=True,
    model_id= None,
    embbedings_model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_ollama= True,  # Set to True to use local Ollama LLM
    ollama_model="llama3.1",
    temperature=0.1
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
        model_id (str): The ID of the IBM Watsonx model to use.
        use_ollama (bool): If True, use local Ollama instead of IBM Watsonx.
        ollama_model (str): The name of the Ollama model (default: "llama3.1").
        temperature (float): Temperature setting for the LLM.

    Returns:
        RetrievalQA: The configured RetrievalQA chain.

    Raises:
        ValueError: If an unsupported model_id is provided, or if required
                    parameters are missing for the chosen retrieval method.
    """
    # Step 1: Configure LLM
    if use_ollama:
        try:
            llm = ChatOllama(
                model=ollama_model,
                temperature=temperature,
            )
            print(f"✓ Using Ollama with model: {ollama_model}")
        except Exception as e:
            raise Exception(f"Failed to initialize Ollama. Run 'ollama run {ollama_model}' first. Error: {e}")
    else:
        # Use IBM Watsonx
        llm = WatsonxLLM(
            url=os.environ.get("WATSONX_URL", "https://jp-tok.ml.cloud.ibm.com"),
            project_id=os.environ.get("IBM_PROJECT_ID"),
            model_id=model_id,
            params={
                "temperature": temperature,
                "max_new_tokens": 512,
                "repetition_penalty": 1.1
            }
        )

    # Step 2: Create custom prompt template
    prompt_template = """
You are a professional Financial Research Assistant specialising in analysing company annual reports.

Your task is to answer the question as accurately and completely as possible using the context provided.

INSTRUCTIONS:
1. Grounding:
   Answer the question using ONLY the provided context. Do not use external knowledge.

2. Supported Interpretation:
   You MAY perform light interpretation when the answer is not stated verbatim, provided it is clearly supported by:
   - Job titles and role descriptions
   - Section headings (e.g. Chairman’s Statement, Risk Management, Operations Review)
   - Tables, charts, and financial summaries
   - Consistent statements across multiple parts of the context

3. Answer Completeness:
   If relevant information is available across multiple sentences or sections, combine them into a single coherent answer.

4. Tone and Style:
   - Write in full, professional sentences
   - Do NOT include reasoning steps, explanations, or meta-commentary
   - Do NOT mention phrases such as “the document states” or “based on the context”

5. Uncertainty Handling:
   If the information is genuinely not available or cannot be reasonably inferred from the context, respond ONLY with:
   "The information is not stated in the provided document."
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
