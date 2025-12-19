from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.schema import Document
from langchain_docling import DoclingLoader
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores.utils import filter_complex_metadata
import re

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

class SentenceParagraphSplitter:
    """
    Custom text splitter that respects natural language boundaries (sentences and paragraphs).
    
    This splitter prioritizes maintaining semantic coherence by:
    1. First splitting on paragraph boundaries (double newlines)
    2. Then splitting on sentence boundaries (periods, question marks, exclamation marks)
    3. Combining sentences into chunks that don't exceed max size
    """
    
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the sentence/paragraph splitter.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> list:
        """Split text into chunks based on sentences and paragraphs."""
        # First split by paragraphs (double newlines or more)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Split paragraph into sentences
            # Match sentences ending with . ! ? followed by space or end of string
            sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # If adding this sentence would exceed max size, start a new chunk
                if current_chunk and len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                    chunks.append(current_chunk.strip())
                    # Add overlap from the end of previous chunk
                    if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                        current_chunk = current_chunk[-self.chunk_overlap:] + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            # Add paragraph break if not at max size
            if current_chunk and len(current_chunk) < self.max_chunk_size - 10:
                current_chunk += "\n\n"
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_documents(self, documents: list) -> list:
        """Split a list of Document objects."""
        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                # Create new document with metadata
                metadata = doc.metadata.copy()
                metadata['chunk_index'] = i
                split_docs.append(Document(page_content=chunk, metadata=metadata))
        return split_docs

def create_chroma_vectordb_with_sentence_splitting(
    file_paths: list,
    chroma_db_folder: str = "./chroma_db_sentence",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_chunk_size: int = 1000,
    chunk_overlap: int = 100
):
    """
    Create a Chroma vector database using sentence/paragraph-aware splitting.
    
    This approach splits text using natural language boundaries (sentences and paragraphs)
    rather than arbitrary character counts. This preserves semantic meaning and context,
    making it ideal for tasks requiring understanding of complete thoughts and ideas.
    
    Args:
        file_paths (list): List of paths to documents to be processed
        chroma_db_folder (str): Directory to store the Chroma vector database
        model_name (str): HuggingFace model for embeddings
        max_chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Overlap between chunks to maintain context
    
    Returns:
        Chroma: The persistent Chroma vector store with sentence-aware chunked documents
        
    Pros:
        ✅ Keeps sentences intact, preserving meaning
        ✅ Respects paragraph boundaries for better context
        ✅ Easy and intuitive setup
        ✅ Works well with structured text (articles, papers, reports)
        ✅ Maintains semantic flow for topic modeling and sentiment analysis
        
    Cons:
        ❌ May fail on poorly formatted text without proper punctuation
        ❌ Less effective on unstructured or noisy text
        ❌ Chunk sizes may vary significantly based on sentence length
    
    Best Use Cases:
        - Topic modeling and sentiment analysis where semantic flow matters
        - Highly structured text like articles or research papers
        - Annual reports, technical documentation, academic papers
        - Content where complete sentences are essential for understanding
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
        
        # Step 3: Create sentence/paragraph splitter
        print(f"Creating sentence/paragraph splitter (max_size={max_chunk_size}, overlap={chunk_overlap})...")
        text_splitter = SentenceParagraphSplitter(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Step 4: Split documents using sentence-aware splitting
        print("Splitting documents using sentence/paragraph boundaries...")
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} sentence-aware chunks")
        
        # Print sample chunk info
        if chunks:
            print(f"\nSample chunk preview (first 3 chunks):")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n  Chunk {i+1}:")
                print(f"    Length: {len(chunk.page_content)} characters")
                print(f"    Preview: {chunk.page_content[:150]}...")
                # Count sentences in chunk
                sentences = re.split(r'(?<=[.!?])\s+', chunk.page_content)
                print(f"    Sentences: {len([s for s in sentences if s.strip()])}")
        
        # Step 5: Create Chroma vector store
        print(f"\nCreating vector store at {chroma_db_folder}...")
        if not os.path.exists(chroma_db_folder):
            os.makedirs(chroma_db_folder)
        
        vectorstore = Chroma.from_documents(
            filter_complex_metadata(chunks),
            embeddings,
            persist_directory=chroma_db_folder
        )
        
        print("✓ Sentence/paragraph splitting completed successfully!")
        return vectorstore
        
    except Exception as e:
        raise Exception(f"Failed to create vector database with sentence splitting: {e}")

def demonstrate_sentence_splitting():
    """
    Demonstrate sentence/paragraph splitting with examples.
    """
    print("\n" + "="*80)
    print("SENTENCE/PARAGRAPH SPLITTING DEMONSTRATION")
    print("="*80)
    
    # Example 1: Well-formatted text
    example_text1 = """LangChain simplifies AI processing. It uses embeddings for efficient retrieval.
This enables smarter question-answering systems.

RAG (Retrieval-Augmented Generation) combines retrieval with generation. This approach improves accuracy significantly. It's widely used in modern AI applications.

Vector databases store embeddings efficiently. They enable fast similarity search. ChromaDB is a popular choice for this purpose."""
    
    print("\nExample 1: Well-formatted text with paragraphs")
    print("-" * 80)
    print(example_text1)
    print("-" * 80)
    
    splitter = SentenceParagraphSplitter(max_chunk_size=150, chunk_overlap=20)
    chunks1 = splitter.split_text(example_text1)
    
    print(f"\n✓ Split into {len(chunks1)} chunks:\n")
    for i, chunk in enumerate(chunks1, 1):
        print(f"Chunk {i}:")
        print(f"  {chunk}")
        print(f"  Length: {len(chunk)} characters")
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        print(f"  Sentences: {len([s for s in sentences if s.strip()])}\n")
    
    # Example 2: Comparison with character-based splitting
    print("\n" + "="*80)
    print("COMPARISON: Sentence Splitting vs Character Splitting")
    print("="*80)
    
    example_text2 = """The company reported strong financial performance. Revenue increased by 25% year-over-year. Net profit margins improved significantly. The CEO expressed optimism about future growth."""
    
    print("\nOriginal text:")
    print(example_text2)
    
    # Sentence splitting
    sentence_splitter = SentenceParagraphSplitter(max_chunk_size=80, chunk_overlap=0)
    sentence_chunks = sentence_splitter.split_text(example_text2)
    
    print(f"\n📝 Sentence Splitting ({len(sentence_chunks)} chunks):")
    for i, chunk in enumerate(sentence_chunks, 1):
        print(f"  {i}. {chunk}")
    
    # Character splitting (simple)
    char_chunks = [example_text2[i:i+80] for i in range(0, len(example_text2), 80)]
    
    print(f"\n📄 Character Splitting ({len(char_chunks)} chunks):")
    for i, chunk in enumerate(char_chunks, 1):
        print(f"  {i}. {chunk}")
    
    print("\n" + "="*80)
    print("Notice how sentence splitting:")
    print("  ✅ Preserves complete sentences")
    print("  ✅ Maintains semantic meaning")
    print("  ✅ Respects natural language boundaries")
    print("  ✅ Better for comprehension and analysis")
    print("\nWhile character splitting:")
    print("  ❌ Breaks sentences mid-word")
    print("  ❌ Loses semantic context")
    print("  ❌ Harder to understand individual chunks")

def main():
    """
    Main function to demonstrate the usage of setup_qa_chain with sentence splitting.
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
    
    print("\n--- Example 5: Sentence/Paragraph Splitting Demonstration ---")
    try:
        demonstrate_sentence_splitting()
    except Exception as e:
        print(f"Demonstration error: {e}")
    
    print("\n--- Example 6: Create Vector DB with Sentence Splitting ---")
    try:
        # Example with markdown files
        file_paths = ["./output_md/example.md"]  # Replace with actual file paths
        vectorstore = create_chroma_vectordb_with_sentence_splitting(
            file_paths=file_paths,
            chroma_db_folder="./chroma_db_sentence",
            max_chunk_size=1000,
            chunk_overlap=100
        )
        print("Successfully created vector store with sentence splitting!")
    except Exception as e:
        print(f"Sentence splitting example error: {e}")

if __name__ == "__main__":
    main()
