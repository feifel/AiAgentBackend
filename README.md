# AI Agent Backend

This AI Agent Backend is the backend to my AI Agent Frontend. You can talk and screenshare to the AI, which makes it very efficient to solve any kind of computer problems. It uses WebSocket communication for low latency and streaming. I uses aratan/gemma3u (LLM), Whisper (STT) and Kokoro (TTS). Everything is deployed fully local and can run for free.

## Prerequisites
- Python 3.12 or newer
- pip (package installer for Python)
- CUDA GPU with at least 12 GByte VRAM
- AI Agent Frontend (see other project)
- Ollama installed (for running aratan/gemma3u)
- huggingface.co account to download Whisper
- PyTorch with CUDA, see here [Start Locally | PyTorch](https://pytorch.org/get-started/locally/)
    ```powershell
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    This will download a 2.7 GByte file.

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
6. Install Gemma3 model in Ollama:
    ```powershell
    ollama pull gemma3
    ```
7. Start the backend:     
    ```powershell
    python main.py
    ```    
    The first time when you run this it will download the remaining models:    
    - openai--whisper-large-v3-turbo: 1.51 GByte
    - hexgrad--Kokoro-82M: 0.31 GByte
    
    It should show the port of the WebSocket that it is listening:         
    ```powershell
    2025-04-11 14:07:01,348 - INFO - WebSocket server running on 0.0.0.0:9073
    ```        
    → You will need to configure this port on the AiAgentFrontend project (App.svelte).

## Credits
This project was heavily inspired by https://github.com/yeyu2/Youtube_demos/tree/main/Multimodal-server-gemma3

## Roadmap
1. Support German:
    - Replace Kokoro with gTTS to support TTS in German
    - Use gemma3 from ollama (this supports German)
        -> Unfortunately screensharing got broken with ollama
        -> Check if gemma3 from Hugging faces also works
2. Replace gTTS with OpenVoice since gTTS requires online connection
3. Send also LLM Text besides Audio to updated ChatHistory
4. Implement Talking Avatar using Wav2Lip
5. Add MCP support for tooling

## Similar Projects
1. https://github.com/HumanAIGC-Engineering/OpenAvatarChat
