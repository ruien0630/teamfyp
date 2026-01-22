# Using Local LLM with Ollama

## Overview
Your RAG system now supports **both cloud-based (IBM Watsonx) and local LLM (Ollama)** options. Simply change one parameter to switch between them!

## Setup Instructions

### Step 1: Install Ollama
1. Download Ollama from: https://ollama.com/download
2. Install the application (Windows/Mac/Linux)
3. Ollama will run as a background service

### Step 2: Download Llama 3.1 Model
Open your terminal and run:
```bash
ollama run llama3.1
```

This will:
- Download the Llama 3.1 8B model (~4.7GB)
- Start an interactive chat (you can exit with `/bye`)
- Keep the model cached for future use

### Step 3: Verify Installation
```bash
ollama list
```
You should see `llama3.1` in the list of downloaded models.

## Usage in Your Code

### Modified File: `query_util_exp2.py`
The file now supports both Watsonx and Ollama with minimal changes:

**Using Ollama (Local):**
```python
import query_util_exp2 as query_util

qa_chain = query_util.setup_qa_chain(
    local_vector_store_path="./chroma_db",
    use_ollama=True,           # 👈 Set to True for local LLM
    ollama_model="llama3.1",
    temperature=0
)
```

**Using IBM Watsonx (Cloud):**
```python
import query_util_exp2 as query_util

qa_chain = query_util.setup_qa_chain(
    local_vector_store_path="./chroma_db",
    use_ollama=False,          # 👈 Set to False for Watsonx
    model_id="ibm/granite-4-h-small"
)
```

## In the Notebook

See the new cells in `rag_annual_report.ipynb`:
- **Cell with Watsonx setup** - Option 1 (cloud-based)
- **Cell with Ollama setup** - Option 2 (local)
- **Test cell** - Try a sample question with Ollama

## Troubleshooting

### "ollama not found" error
- Make sure Ollama is installed and running
- Restart your terminal after installation
- Check with: `ollama --version`

### "Failed to initialize Ollama" error
- Ensure Ollama service is running
- Download the model first: `ollama run llama3.1`
- Check available models: `ollama list`

### Slow responses
- First query may be slower (model loading)
- Performance depends on your hardware
- Consider using a smaller model if needed: `ollama run llama3.1:8b`

## Comparison: Ollama vs Watsonx

| Feature | Ollama (Local) | IBM Watsonx (Cloud) |
|---------|----------------|---------------------|
| **Cost** | Free | Requires IBM Cloud account |
| **Privacy** | Runs offline | Data sent to cloud |
| **Speed** | Depends on hardware | Consistent cloud speed |
| **Setup** | 2-step install | API keys needed |
| **Model** | Llama 3.1 8B | Granite 4 |

## Available Ollama Models

You can use other models too:
```bash
ollama run llama3.1        # Llama 3.1 8B (default)
ollama run mistral         # Mistral 7B
ollama run codellama       # Code-specialized
ollama run llama3.1:70b    # Larger model (better quality, slower)
```

Then just change the `ollama_model` parameter:
```python
qa_chain = query_util.setup_qa_chain(
    local_vector_store_path="./chroma_db",
    use_ollama=True,
    ollama_model="mistral"  # 👈 Change model here
)
```

## What Changed?

### 1. Modified `query_util_exp2.py`
- ✅ Added `ChatOllama` import
- ✅ Added `use_ollama` parameter (default: False)
- ✅ Added `ollama_model` parameter (default: "llama3.1")
- ✅ Modified LLM initialization to support both options
- ✅ No breaking changes - existing Watsonx code still works!

### 2. Created `query_util_ollama.py`
- Standalone version using only Ollama
- Alternative if you want a dedicated Ollama-only file

### 3. Updated Notebook
- Added Ollama setup cells
- Added comparison documentation
- Test cells for both approaches

## Next Steps

1. ✅ Install Ollama
2. ✅ Download Llama 3.1 model
3. ✅ Run the Ollama setup cell in your notebook
4. ✅ Test with your golden dataset questions
5. ✅ Compare results between Watsonx and Ollama
