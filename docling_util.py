import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from pathlib import Path
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ImageRefMode
from tqdm import tqdm
import gemini_vision
import re

from langchain_chroma import Chroma
from langchain_docling import DoclingLoader
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

# Load environment variables
load_dotenv()

def convert_pdf_with_image_annotation(input_doc_path: str, ai_service_provider: str = "google") -> DocumentConverter:
    """
    Convert and annotate images in a PDF using specified AI service.
    
    Args:
        input_doc_path (str): Path to the PDF file
        ai_service_provider (str): AI service to use ('google' or 'openai')
        
    Returns:
        DocumentConverter: Converted document with annotations
    """
    pipeline_options = PdfPipelineOptions(
        enable_remote_services=True,
        generate_picture_images=True,
        images_scale=2
    )
    
    if ai_service_provider.lower() in ["google", "openai"]:
        if ai_service_provider.lower() == "google":
            api_key = os.environ.get("GOOGLE_API_KEY")
            model = "gemini-2.0-flash"  # Updated model name
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            headers = {"x-goog-api-key": api_key}
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            model = "gpt-4-vision-preview"
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}"}

        picture_desc_api_option = PictureDescriptionApiOptions(
            url=url,
            prompt="Describe this image in a single paragraph. The entire response must be in Markdown.",
            params=dict(model=model),
            headers=headers,
            timeout=60,
        )
        
        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = picture_desc_api_option

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    return converter.convert(source=input_doc_path)

def process_documents_to_md(doc_dir: str, ai_service_provider: str = "google"):
    """
    Process documents from a directory based on their file type.
    
    Args:
        doc_dir (str): Directory containing documents
        ai_service_provider (str): AI service to use for image annotation
    """
    output_dir = Path("output_md")
    output_dir.mkdir(exist_ok=True)
    
    # Define supported formats and their processors
    SUPPORTED_FORMATS = {
        '.pdf': convert_pdf_with_image_annotation,
        # Add more formats and their processors here
        # '.docx': convert_docx_with_annotation,
        # '.txt': convert_txt_with_annotation,
    }

    # Get all files with supported extensions
    files = [
        f for f in Path(doc_dir).glob("*") 
        if f.suffix.lower() in SUPPORTED_FORMATS
    ]

    # Print found files by type
    for ext in SUPPORTED_FORMATS:
        ext_files = [f for f in files if f.suffix.lower() == ext]
        if ext_files:
            print(f"\n{ext} files found:")
            for f in ext_files:
                print(f"  - {f.name}")

    print(f"\nTotal files to process: {len(files)}")

    # Process each file with appropriate converter
    processed_files = []
    for file_path in tqdm(files, desc="Processing documents"):
        try:
            # Get the appropriate converter for the file type
            converter = SUPPORTED_FORMATS[file_path.suffix.lower()]
            
            # Process the file
            result = converter(str(file_path), ai_service_provider)
            
            # Save output
            md_filename=output_dir / f"{file_path.stem}.md"
            result.document.save_as_markdown(
                md_filename,
                image_mode=ImageRefMode.REFERENCED,
                include_annotations=True
            )
            processed_files.append(file_path)
            print(f"\n✓ Processed: {file_path}")
            
        except Exception as e:
            print(f"\n✗ Error processing {file_path.name}: {str(e)}")
    return processed_files

def process_markdown_folder(folder_path):
    """
    Processes all markdown files within a given folder by calling
    describe_images_and_update on each one.

    Args:
        folder_path (str): The path to the folder containing markdown files.
    """
    try:
        # Check if the provided path is a directory
        if not os.path.isdir(folder_path):
            print(f"Error: '{folder_path}' is not a valid directory.")
            return

        print(f"Processing markdown files in folder: {folder_path}")
        
        # Iterate over all entries in the folder
        for filename in os.listdir(folder_path):
            # Check if the file has a .md extension (case-insensitive)
            if filename.lower().endswith('.md'):
                # Construct the full path to the markdown file
                markdown_file = os.path.join(folder_path, filename)
                print(f"\nProcessing md file: {markdown_file}")
                
                # Call the function to update the markdown file
                describe_images_and_update(markdown_file)

    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def describe_images_and_update(markdown_file):
    """
    Reads a markdown file, extracts the filename from each image tag,
    analyzes the image, and writes the analysis result back into the
    markdown file using markdown formatting.

    Args:
        markdown_file (str): The path to the markdown file.
    """
    image_regex = r"!\[.*?\]\((.*?\.(?:jpg|jpeg|png|gif|bmp|svg|webp))\)"
    
    parent_profile = os.path.dirname(markdown_file)
    
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            match = re.search(image_regex, line, re.IGNORECASE)
            
            new_lines.append(line)
            
            if match:
                filename = match.group(1)
                
                if not os.path.isabs(filename):
                    filename = os.path.join(parent_profile, filename)
                
                print(f"Analyzing image filename: {filename}")
                
                analysis_result = gemini_vision.analyze_image(filename)
                
                # MODIFICATION: Format the analysis result in Markdown
                # The result is wrapped in a bolded prefix and followed by two newlines
                analysis_line = f"**Description**: {analysis_result}\n\n"
                new_lines.append(analysis_line)
                
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"Successfully updated '{markdown_file}' with formatted image descriptions.")

    except FileNotFoundError:
        print(f"Error: The file '{markdown_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_chroma_vectordb(
    file_paths: list, 
    chroma_db_folder: str = "./chroma_db",
    text_splitter_choice: str = "CharacterTextSplitter",
    splitter_para: dict = {},
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Complete document processing pipeline with enhanced parameter flexibility and error handling.

    Args:
        file_paths (list): A list of paths to the documents to be processed.
        chroma_db_folder (str): The directory to permanently store the Chroma vector database.
                                Defaults to "./chroma_db".
        text_splitter_choice (str): The type of text splitter to use. Supported choices are
                                    "CharacterTextSplitter" and "TokenTextSplitter".
                                    Defaults to "CharacterTextSplitter".
        splitter_para (dict): A dictionary of parameters for the text splitter.
                              Defaults to an empty dictionary, with chunk_size=1000 and 
                              chunk_overlap=200 as fallback.
        model_name (str): The name of the HuggingFace model for generating embeddings.
                          Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        Chroma: The persistent Chroma vector store.
    """
    try:
        # Step 1: Load documents using Docling
        converter = DocumentConverter()
        loader = DoclingLoader(file_paths, converter=converter)
        documents = loader.load()

        # Step 2: Initialize and configure the text splitter
        chunk_size = splitter_para.get('chunk_size', 1000)
        chunk_overlap = splitter_para.get('chunk_overlap', 200)

        if text_splitter_choice == "CharacterTextSplitter":
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
        elif text_splitter_choice == "TokenTextSplitter":
            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError(
                f"Unsupported text splitter choice: {text_splitter_choice}. "
                "Supported splitters are 'CharacterTextSplitter' and 'TokenTextSplitter'."
            )        
        
        chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Ensure the directory for Chroma DB exists
        if not os.path.exists(chroma_db_folder):
            os.makedirs(chroma_db_folder)
        
        vectorstore = Chroma.from_documents(
            filter_complex_metadata(chunks), 
            embeddings,
            persist_directory=chroma_db_folder
        )

        print(f"Vectorstore saved permanently to: {os.path.abspath(chroma_db_folder)}")
        
        return vectorstore

    except Exception as e:
        print(f"An error occurred during document processing: {e}")
        return None
    
def main():
    """Example usage of the convert_pdf_with_image_annotation function"""
    
    # Example 1: Using Google Gemini (default)
    try:
        result = convert_pdf_with_image_annotation("input_docs/sample.pdf")
        print("Google Gemini Result:")
        print(result.document.export_to_markdown(mark_annotations=True, include_annotations=True))
    except Exception as e:
        print(f"Google conversion failed: {e}")

    # Example 2: Using OpenAI
    try:
        result = convert_pdf_with_image_annotation("input_docs/sample.pdf", ai_service_provider="openai")
        print("\nOpenAI Result:")
        print(result.document.export_to_markdown(mark_annotations=True, include_annotations=True))
    except Exception as e:
        print(f"OpenAI conversion failed: {e}")

    # Example 3: Basic conversion without image annotation
    try:
        result = convert_pdf_with_image_annotation("input_docs/sample.pdf", ai_service_provider="none")
        print("\nBasic Conversion Result:")
        print(result.document.export_to_markdown())
    except Exception as e:
        print(f"Basic conversion failed: {e}")

    """Example usage of document processing"""
    doc_dir = "input_docs"
    
    # Process with Google Gemini (default)
    print (process_documents_to_md(doc_dir))
    
    # Or process with OpenAI
    # print process_documents(doc_dir, ai_service_provider="openai")

    # Analyze images in markdown file
    file_to_process = "annual-report-sats-ltd-2023-2024_p8_only.md"
    describe_images_and_update(file_to_process)

    # Example Usage\
    process_markdown_folder('my_docs')

    """Example usage of create_chroma_vectordb function"""
    # Note: Replace these with actual paths to your documents for the code to run successfully.
    # For demonstration, we use placeholder paths.
    
    # Example 1: Basic call with all default parameters
    print("--- Example 1: Basic Call with Defaults ---")
    try:
        # Assuming you have documents at these paths
        my_documents = ["./data/doc1.txt", "./data/doc2.pdf"] 
        db_default = create_chroma_vectordb(file_paths=my_documents)
        if db_default:
            print("Successfully created a vector DB with default settings.")
    except Exception as e:
        print(f"Example 1 failed: {e}")
        
    print("\n" + "-"*50 + "\n")

    # Example 2: Custom folder and text splitter parameters
    print("--- Example 2: Custom Folder and Splitter Parameters ---")
    try:
        my_documents = ["./data/report.docx"]
        db_custom_splitter = create_chroma_vectordb(
            file_paths=my_documents,
            chroma_db_folder="./my_custom_db",
            splitter_para={'chunk_size': 500, 'chunk_overlap': 50}
        )
        if db_custom_splitter:
            print("Successfully created a vector DB with a custom folder and chunking parameters.")
    except Exception as e:
        print(f"Example 2 failed: {e}")
        
    print("\n" + "-"*50 + "\n")

    # Example 3: Use a different text splitter and embedding model
    print("--- Example 3: Different Splitter and Embedding Model ---")
    try:
        my_documents = ["./data/article.html"]
        db_token_splitter = create_chroma_vectordb(
            file_paths=my_documents,
            text_splitter_choice="TokenTextSplitter",
            model_name="BAAI/bge-small-en-v1.5" # A different popular embedding model
        )
        if db_token_splitter:
            print("Successfully created a vector DB using TokenTextSplitter and a different model.")
    except Exception as e:
        print(f"Example 3 failed: {e}")
        
    print("\n" + "-"*50 + "\n")

    # Example 4: Test unsupported text splitter (will trigger exception handling)
    print("--- Example 4: Testing Unsupported Splitter (Expected Failure) ---")
    try:
        my_documents = ["./data/some_file.txt"]
        db_unsupported = create_chroma_vectordb(
            file_paths=my_documents,
            text_splitter_choice="RecursiveSplitter"
        )
        if db_unsupported is None:
            print("Handled the unsupported splitter exception gracefully.")
    except Exception as e:
        print(f"Example 4 failed unexpectedly: {e}")

if __name__ == "__main__":
    main()