# AI Agent Backend

This AI Agent Backend is the backend to my AI Agent Frontend. You can talk and screenshare to the AI, which makes it very efficient to solve any kind of computer problems. It uses WebSocket communication for low latency and streaming. I uses gemma-3 (LLM), Whisper (STT) and Kokoro (TTS). Everything is deployed fully local and can run for free.

## Prerequisites
- Python 3.12 or newer
- pip (package installer for Python)
- CUDA GPU with at least 12 GByte VRAM
- PyTorch with CUDA
- AI Agent Frontend (see other project)
- huggingface.co account to download gemma-3 and Whisper

## Setup
1. Clone the repository    
    ```
    git clone https://github.com/feifel/AiAgentBackend.git
    cd AiAgentBackend
    ```    
3. Run the following command to create a virtual environment:     
    ```powershell
    python -m venv venv
    ```    
4. Activate the virtual environment:    
    ```csharp
    venv\Scripts\Activate.ps1
    ```    
5. Install the dependencies:     
    ```powershell
    pip install -r requirements.txt
    ```    
6. Start the backend:     
    ```powershell
    python main.py
    ```    
    The first time when you run this it will download the models:    
    - model.safetensors: 1.62 GByte
    - model-00001-of-00002.safetensors:  4.96 GByte
    - model-00002-of-00002.safetensors:  3.03 GByte
    
    And you will see this error:     
    ```powershell
    2025-04-11 11:12:19,922 - INFO - Loading google/gemma-3-4b-it...
    The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
    2025-04-11 11:12:20,159 - ERROR - Server error: You are trying to access a gated repo.
    Make sure to have access to it at https://huggingface.co/google/gemma-3-4b-it.
    401 Client Error. (Request ID: Root=1-67f8dcf3-365946af0a94d9fd54548d43;2937c910-c1bc-42e8-bd24-ada603908e1d)
    ```    
    The Gemma models from Google require acceptance of their license terms and authentication with Hugging Face to use them. Here's how to fix the access issue:    
    1. First, create a Hugging Face account if you don't have one at huggingface.co
    2. Accept the Gemma model license:
        - Visit google/gemma-3-4b-it
        - Click "Accept" on the model license agreement
    3. Get your Hugging Face token:
        - Go to your Hugging Face profile settings
        - Create a new access token
        - Copy the token
    4. Login using the token in your terminal:         
        ```powershell
        huggingface-cli login --token YOUR_TOKEN_HERE
        ```        
    5. After that you can try to start the server again: `python main.py`        
        It should show the port of the WebSocket that it is listening:         
        ```powershell
        2025-04-11 14:07:01,348 - INFO - WebSocket server running on 0.0.0.0:9073
        ```        
        → You will need to configure this port on the AiAgentFrontend project (App.svelte).

## Credits
This project was heavily inspired by https://github.com/yeyu2/Youtube_demos/tree/main/Multimodal-server-gemma3