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
    ```bash
    python3 -m venv venv
    ```    
4. Activate the virtual environment:    
    ```bash
    source ./venv/bin/activate
    ```    
5. Install the dependencies:     
    ```bash
    pip install -r requirements.txt
    ```    
6. Install Gemma3 model in Ollama:
    ```bash
    ollama pull gemma3
    ```
    If you havn't Ollama installed yet, you can install it by running this as normal user:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    sudo systemctl enable --now ollama
    ```
7. Install kani-tts:
    ```bash
    cd /home/roman/Projects
    git clone https://github.com/feifel/kani-tts.git
    cd kani-tts
    docker build -t kani-tts .
    docker run --gpus all -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/cache:/app/cache \
    -v $(pwd)/output:/app/output \
    kani-tts
    ```
    Read this for more details:
    https://github.com/feifel/kani-tts/blob/main/DOCKER.md
8. Start the backend:     
    ```bash
    python3 main.py
    ```    
    The first time when you run this it will download the remaining models:    
    - openai--whisper-large-v3-turbo: 1.51 GByte
    - hexgrad--Kokoro-82M: 0.31 GByte
    
    It should show the port of the WebSocket that it is listening:         
    ```bash
    2025-04-11 14:07:01,348 - INFO - WebSocket server running on 0.0.0.0:9073
    ```        
    → You will need to configure this port on the AiAgentFrontend project (App.svelte).

## Credits
This project was heavily inspired by https://github.com/yeyu2/Youtube_demos/tree/main/Multimodal-server-gemma3

## Roadmap
1. Use Configuration provide by Configuration messages (see line 590 and 826)
1. Use https://ollama.readthedocs.io/en/api/#generate-a-chat-completion instead of https://ollama.readthedocs.io/en/api/#generate-a-completion
    1. Maintain chat history messages
2. Implement visualization of an Avatar with Lipsync by using https://github.com/Rudrabha/Wav2Lip or better https://github.com/antgroup/ditto-talkinghead
3. Add support to select different tools retrieved from an MCP server
    1. Implement MCP server: 
        1. Implement Mem0: https://www.youtube.com/watch?v=lbyPJqCI-tw
        2. Implement LightRAG: https://www.youtube.com/watch?v=Fx3J8k--U3E
    2. Integrate n8n (via MCP server): https://www.youtube.com/watch?v=WcIRVk8kyJk
5. Replace gTTS with kani-tts, which runs locally with 2 GByte VRAM, supports German and streaming
   -> See docker setup here: https://github.com/feifel/kani-tts 

## Similar Projects
1. https://github.com/HumanAIGC-Engineering/OpenAvatarChat
