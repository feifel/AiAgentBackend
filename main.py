import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
from gtts import gTTS
import re
import librosa
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels"""
    
    def __init__(self, 
                 sample_rate=16000,
                 energy_threshold=0.015,
                 silence_duration=0.8,
                 min_speech_duration=0.8,
                 max_speech_duration=15): 
        
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        
        # Internal state
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.lock = asyncio.Lock()
        self.segment_queue = asyncio.Queue()
        
        # Counters
        self.segments_detected = 0
        
        # Add TTS playback and generation control
        self.tts_playing = False
        self.tts_lock = asyncio.Lock()
        self.current_generation_task = None
        self.current_tts_task = None
        self.task_lock = asyncio.Lock()
    
    async def set_tts_playing(self, is_playing):
        """Set TTS playback state"""
        async with self.tts_lock:
            self.tts_playing = is_playing
    
    async def cancel_current_tasks(self):
        """Cancel any ongoing generation and TTS tasks"""
        async with self.task_lock:
            if self.current_generation_task and not self.current_generation_task.done():
                self.current_generation_task.cancel()
                try:
                    await self.current_generation_task
                except asyncio.CancelledError:
                    pass
                self.current_generation_task = None
            
            if self.current_tts_task and not self.current_tts_task.done():
                self.current_tts_task.cancel()
                try:
                    await self.current_tts_task
                except asyncio.CancelledError:
                    pass
                self.current_tts_task = None
            
            # Clear TTS playing state
            await self.set_tts_playing(False)
    
    async def set_current_tasks(self, generation_task=None, tts_task=None):
        """Set current generation and TTS tasks"""
        async with self.task_lock:
            self.current_generation_task = generation_task
            self.current_tts_task = tts_task
    
    async def add_audio(self, audio_bytes):
        """Add audio data to the buffer and check for speech segments"""
        async with self.lock:
            # Add new audio to buffer regardless of TTS state
            self.audio_buffer.extend(audio_bytes)
            
            # Convert recent audio to numpy for energy analysis
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate audio energy (root mean square)
            if len(audio_array) > 0:
                energy = np.sqrt(np.mean(audio_array**2))
                
                # Speech detection logic
                if not self.is_speech_active and energy > self.energy_threshold:
                    # Speech start detected
                    self.is_speech_active = True
                    self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                    self.silence_counter = 0
                    logger.info(f"Speech start detected (energy: {energy:.6f})")
                    
                elif self.is_speech_active:
                    if energy > self.energy_threshold:
                        # Continued speech
                        self.silence_counter = 0
                    else:
                        # Potential end of speech
                        self.silence_counter += len(audio_array)
                        
                        # Check if enough silence to end speech segment
                        if self.silence_counter >= self.silence_samples:
                            speech_end_idx = len(self.audio_buffer) - self.silence_counter
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                            
                            # Reset for next speech detection
                            self.is_speech_active = False
                            self.silence_counter = 0
                            
                            # Trim buffer to keep only recent audio
                            self.audio_buffer = self.audio_buffer[speech_end_idx:]
                            
                            # Only process if speech segment is long enough
                            if len(speech_segment) >= self.min_speech_samples * 2:  # × 2 for 16-bit
                                self.segments_detected += 1
                                logger.info(f"Speech segment detected: {len(speech_segment)/2/self.sample_rate:.2f}s")
                                
                                # If TTS is playing or generation is ongoing, cancel them
                                async with self.tts_lock:
                                    if self.tts_playing:
                                        await self.cancel_current_tasks()
                                
                                # Add to queue
                                await self.segment_queue.put(speech_segment)
                                return speech_segment
                            
                        # Check if speech segment exceeds maximum duration
                        elif (len(self.audio_buffer) - self.speech_start_idx) > self.max_speech_samples * 2:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:
                                                             self.speech_start_idx + self.max_speech_samples * 2])
                            # Update start index for next segment
                            self.speech_start_idx += self.max_speech_samples * 2
                            self.segments_detected += 1
                            logger.info(f"Max duration speech segment: {len(speech_segment)/2/self.sample_rate:.2f}s")
                            
                            # If TTS is playing or generation is ongoing, cancel them
                            async with self.tts_lock:
                                if self.tts_playing:
                                    await self.cancel_current_tasks()
                            
                            # Add to queue
                            await self.segment_queue.put(speech_segment)
                            return speech_segment
            
            return None
    
    async def get_next_segment(self):
        """Get the next available speech segment"""
        try:
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

class WhisperTranscriber:
    """Handles speech transcription using Whisper large-v3 model with pipeline"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # Use GPU for transcription
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Set torch dtype based on device
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        # Load model and processor
        model_id = "openai/whisper-large-v3-turbo"
        logger.info(f"Loading {model_id}...")
        
        # Load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        logger.info("Whisper model ready for transcription")
        
        # Counter
        self.transcription_count = 0
    
    async def transcribe(self, audio_bytes, sample_rate=16000):
        """Transcribe audio bytes to text using the pipeline"""
        try:
            # Convert PCM bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Check for valid audio
            if len(audio_array) < 1000:  # Too short
                return ""
            
            # Use the pipeline to transcribe
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipe(
                    {"array": audio_array, "sampling_rate": sample_rate},
                    generate_kwargs={
                        "task": "transcribe",
                        "language": "german",  # Changed from "english" to "german"
                        "temperature": 0.0
                    }
                )
            )
            
            # Extract the text from the result
            text = result.get("text", "").strip()
            
            self.transcription_count += 1
            logger.info(f"Transcription result: '{text}'")
            
            return text
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

class Gemma3Processor:
    """Handles text generation using Gemma3 model via Ollama"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        logger.info("Initializing Gemma3 processor...")
        
        # Message history management
        self.message_history = []
        self.max_history_messages = 4  # Keep last 4 exchanges
        
        # Cache for most recent image
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()
        
        # Ollama API endpoint
        self.ollama_api = "http://localhost:11434/api/generate"
        
        # Counter
        self.generation_count = 0
        
        logger.info("Gemma3 processor initialized")
    
    async def set_image(self, image_data):
        """Cache the most recent image received"""
        async with self.lock:
            try:
                # Always ensure data URL prefix
                if image_data.startswith('data:image/'):
                    self.last_image = image_data
                else:
                    self.last_image = f"data:image/jpeg;base64,{image_data}"
                self.last_image_timestamp = time.time()
                self.message_history = []
                logger.info(f"Image cached, length: {len(self.last_image)}")
                return True
            except Exception as e:
                logger.error(f"Error caching image: {e}")
                return False

    def _build_prompt(self, text):
        """Build messages with history for the model"""
        # Format system prompt
        system_prompt = """Du bist ein hilfreicher Assistent, der gesprochene Antworten über Bilder gibt und natürliche Gespräche führt. Halte deine Antworten prägnant, flüssig und gesprächig. Verwende natürliche gesprochene Sprache, die leicht anzuhören ist.

Wenn sich die Frage auf ein Bild bezieht, beschreibe bitte genau was du im Bild siehst. Konzentriere dich auf die wichtigsten Details."""

        # Build conversation without system prompt first
        conversation = ""
        
        # Add conversation history
        for msg in self.message_history[-4:]:  # Only keep last 4 messages
            role_prefix = "Benutzer: " if msg["role"] == "user" else "Assistent: "
            conversation += role_prefix + msg["content"] + "\n"
        
        # Add current message
        conversation += f"Benutzer: {text}\nAssistent:"

        # Combine with the system prompt at the beginning
        final_prompt = f"{system_prompt}\n\n{conversation}"
        
        return final_prompt

    async def _generate_with_ollama(self, prompt):
        """Generate text using Ollama API"""
        try:
            payload = {
                "model": "gemma3:latest",
                "prompt": prompt,
                "stream": True
            }
            if self.last_image:
                img = self.last_image
                # If it starts with data URL, extract only the base64 part
                if img.startswith('data:'):
                    img = img.split(',', 1)[1]
                payload["images"] = [img]
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.ollama_api,
                    json=payload,
                    timeout=30.0
                )
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return response
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return None

    async def generate_streaming(self, text, initial_chunks=3):
        """Generate a response using Gemma3 with streaming"""
        async with self.lock:
            try:
                # Build messages with history
                messages = self._build_prompt(text)
                
                # Get streaming response from Ollama
                response = await self._generate_with_ollama(messages)
                
                if response:
                    # Initialize response text
                    initial_text = ""
                    min_chars = 50
                    sentence_end_pattern = re.compile(r'[.!?]')
                    has_sentence_end = False
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    chunk = data["response"]
                                    initial_text += chunk
                                    
                                    # Check for sentence end
                                    if sentence_end_pattern.search(chunk):
                                        has_sentence_end = True
                                        if len(initial_text) >= min_chars / 2:
                                            break
                                    
                                    # If we have enough content, break
                                    if len(initial_text) >= min_chars and (has_sentence_end or "," in initial_text):
                                        break
                                    
                                    # Safety check
                                    if len(initial_text) >= min_chars * 2:
                                        break
                                        
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    self.generation_count += 1
                    logger.info(f"Gemma3 initial generation: '{initial_text}' ({len(initial_text)} chars)")
                    
                    # Store for history update
                    self.pending_user_message = text
                    self.pending_response = initial_text
                    
                    return response, initial_text
                    
                return None, "Error generating response"
                
            except Exception as e:
                logger.error(f"Gemma3 streaming generation error: {e}")
                return None, f"Error processing: {text}"

    def _update_history_with_complete_response(self, user_text, initial_response, remaining_text=None):
        """Update message history with complete response"""
        complete_response = initial_response
        if remaining_text:
            complete_response = initial_response + remaining_text
        
        # Add exchange to history
        self.message_history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": complete_response}
        ])
        
        # Trim history
        if len(self.message_history) > self.max_history_messages:
            self.message_history = self.message_history[-self.max_history_messages:]
        
        logger.info(f"Updated message history with complete response ({len(complete_response)} chars)")

class GoogleTTSProcessor:
    """Handles text-to-speech conversion using Google TTS (gTTS)"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        logger.info("Initializing Google TTS processor...")
        self.synthesis_count = 0
        self.target_sr = 24000  # Target sample rate for better quality
        logger.info("Google TTS processor initialized successfully")
    
    def _mp3_to_wav(self, mp3_data):
        """Convert MP3 data to WAV numpy array using librosa"""
        try:
            # Load MP3 data using librosa with original sample rate
            y, sr = librosa.load(io.BytesIO(mp3_data), sr=None)  # Use None to keep original sample rate
            
            # Resample to target sample rate if needed
            if sr != self.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            
            # Audio is already normalized by librosa to [-1, 1], just return it
            return y
            
        except Exception as e:
            logger.error(f"Error converting MP3 to WAV: {e}")
            return None

    def _generate_audio(self, text, lang='de'):
        """Generate audio using Google TTS"""
        try:
            # Create an in-memory bytes buffer
            mp3_fp = io.BytesIO()
            
            # Generate MP3 audio
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            # Convert MP3 to numpy array
            audio_data = self._mp3_to_wav(mp3_fp.getvalue())
            
            if audio_data is not None:
                return audio_data
            return None
                
        except Exception as e:
            logger.error(f"Audio generation error: {e}")
            return None

    async def synthesize_initial_speech(self, text):
        """Convert initial text to speech using Google TTS"""
        if not text:
            return None
        
        try:
            logger.info(f"Synthesizing initial speech for text: '{text}'")
            
            # Run TTS in a thread pool to avoid blocking
            audio = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._generate_audio(text, 'de')
            )
            
            if audio is not None:
                self.synthesis_count += 1
                logger.info(f"Initial speech synthesis complete: {len(audio)} samples")
                return audio
            return None
            
        except Exception as e:
            logger.error(f"Initial speech synthesis error: {e}")
            return None
    
    async def synthesize_remaining_speech(self, text):
        """Convert remaining text to speech using Google TTS"""
        if not text:
            return None
        
        try:
            logger.info(f"Synthesizing remaining speech for text: '{text[:50]}...' if len(text) > 50 else text")
            
            # Split text into sentences for better processing
            sentences = re.split(r'([.!?]+)', text)
            audio_segments = []
            
            for i in range(0, len(sentences)-1, 2):
                sentence = sentences[i].strip() + (sentences[i+1] if i+1 < len(sentences) else "")
                if sentence.strip():
                    # Generate audio for each sentence
                    audio = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._generate_audio(sentence, 'de')
                    )
                    if audio is not None:
                        audio_segments.append(audio)
            
            # Combine all audio segments
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                logger.info(f"Remaining speech synthesis complete: {len(combined_audio)} samples")
                return combined_audio
            return None
            
        except Exception as e:
            logger.error(f"Remaining speech synthesis error: {e}")
            return None
    
    async def synthesize_speech(self, text):
        """Convert text to speech using Google TTS (legacy method)"""
        if not text:
            return None
        
        try:
            logger.info(f"Synthesizing speech for text: '{text[:50]}...' if len(text) > 50 else text")
            
            # Split text into sentences for better processing
            sentences = re.split(r'([.!?]+)', text)
            audio_segments = []
            
            for i in range(0, len(sentences)-1, 2):
                sentence = sentences[i].strip() + (sentences[i+1] if i+1 < len(sentences) else "")
                if sentence.strip():
                    # Generate audio for each sentence
                    audio = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._generate_audio(sentence, 'de')
                    )
                    if audio is not None:
                        audio_segments.append(audio)
            
            # Combine all audio segments
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                logger.info(f"Speech synthesis complete: {len(combined_audio)} samples")
                return combined_audio
            return None
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            return None

async def handle_client(websocket):
    """Handles WebSocket client connection"""
    try:
        # Receive initial configuration
        await websocket.recv()
        logger.info("Client connected")
        
        # Initialize speech detection and get instance of processors
        detector = AudioSegmentDetector()
        transcriber = WhisperTranscriber.get_instance()
        gemma3_processor = Gemma3Processor.get_instance()  # Updated to use Gemma3
        tts_processor = GoogleTTSProcessor.get_instance()  # Updated to use Google TTS
        
        # Add keepalive task
        async def send_keepalive():
            while True:
                try:
                    await websocket.ping()
                    await asyncio.sleep(10)  # Send ping every 10 seconds
                except Exception:
                    break
        
        async def detect_speech_segments():
            while True:
                try:
                    # Get next segment from queue
                    speech_segment = await detector.get_next_segment()
                    if speech_segment:
                        # Transcribe directly
                        transcription = await transcriber.transcribe(speech_segment)
                        
                        # Filter out pure punctuation or empty transcriptions
                        if transcription:
                            # Remove extra whitespace
                            transcription = transcription.strip()
                            
                            # Check if transcription contains any alphanumeric characters
                            if not any(c.isalnum() for c in transcription):
                                logger.info(f"Skipping pure punctuation transcription: '{transcription}'")
                                continue
                            
                            # Filter out single-word utterances and common filler sounds
                            words = [w for w in transcription.split() if any(c.isalnum() for c in w)]
                            if len(words) <= 1:
                                logger.info(f"Skipping single-word transcription: '{transcription}'")
                                continue
                                
                            # Filter out common filler sounds and very short responses
                            filler_patterns = [
                                r'^(um+|uh+|ah+|oh+|hm+|mhm+|hmm+)$',
                                r'^(okay|yes|no|yeah|nah)$',
                                r'^bye+$'
                            ]
                            if any(re.match(pattern, transcription.lower()) for pattern in filler_patterns):
                                logger.info(f"Skipping filler sound: '{transcription}'")
                                continue
                            
                            # Send interrupt signal before starting new generation
                            logger.info("Sending interrupt signal for new speech detection")
                            interrupt_message = json.dumps({"interrupt": True})
                            logger.info(f"Interrupt message: {interrupt_message}")
                            await websocket.send(interrupt_message)
                            
                            # Set TTS playing flag and start new generation workflow
                            await detector.set_tts_playing(True)
                            
                            try:
                                # Create generation task
                                generation_task = asyncio.create_task(
                                    gemma3_processor.generate_streaming(transcription, initial_chunks=3)
                                )
                                
                                # Store the generation task
                                await detector.set_current_tasks(generation_task=generation_task)
                                
                                # Wait for initial generation
                                try:
                                    streamer, initial_text = await generation_task
                                except asyncio.CancelledError:
                                    logger.info("Generation cancelled - new speech detected")
                                    continue
                                
                                if initial_text:
                                    # Create TTS task for initial speech
                                    tts_task = asyncio.create_task(
                                        tts_processor.synthesize_initial_speech(initial_text)
                                    )
                                    
                                    # Store the TTS task
                                    await detector.set_current_tasks(tts_task=tts_task)
                                    
                                    try:
                                        # Wait for initial audio synthesis
                                        initial_audio = await tts_task
                                        
                                        if initial_audio is not None:
                                            # Convert to base64 and send to client
                                            # Audio is already in [-1, 1] range from librosa
                                            audio_bytes = (initial_audio * 32768).astype(np.int16).tobytes()
                                            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                                            
                                            # Send the initial audio to the client
                                            await websocket.send(json.dumps({
                                                "audio": base64_audio
                                            }))
                                            
                                            # Now start collecting remaining text in parallel
                                            remaining_text_task = asyncio.create_task(
                                                collect_remaining_text(streamer, initial_text)
                                            )
                                            
                                            # Store the remaining text task
                                            await detector.set_current_tasks(generation_task=remaining_text_task)
                                            
                                            try:
                                                # Wait for remaining text collection
                                                remaining_text = await remaining_text_task
                                                
                                                # Update message history with complete response
                                                gemma3_processor._update_history_with_complete_response(
                                                    transcription, initial_text, remaining_text
                                                )
                                                
                                                if remaining_text:
                                                    # Create TTS task for remaining text
                                                    remaining_tts_task = asyncio.create_task(
                                                        tts_processor.synthesize_remaining_speech(remaining_text)
                                                    )
                                                    
                                                    # Store the TTS task
                                                    await detector.set_current_tasks(tts_task=remaining_tts_task)
                                                    
                                                    try:
                                                        # Wait for remaining audio synthesis
                                                        remaining_audio = await remaining_tts_task
                                                        
                                                        if remaining_audio is not None:
                                                            # Convert to base64 and send to client
                                                            # Audio is already in [-1, 1] range from librosa
                                                            audio_bytes = (remaining_audio * 32768).astype(np.int16).tobytes()
                                                            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                                                            
                                                            # Send the remaining audio to the client
                                                            await websocket.send(json.dumps({
                                                                "audio": base64_audio
                                                            }))
                                                    
                                                    except asyncio.CancelledError:
                                                        # Even if TTS is cancelled, keep the message history
                                                        logger.info("Remaining TTS cancelled - new speech detected")
                                                        continue
                                            
                                            except asyncio.CancelledError:
                                                # If text collection is cancelled, update history with what we have
                                                gemma3_processor._update_history_with_complete_response(
                                                    transcription, initial_text
                                                )
                                                logger.info("Remaining text collection cancelled - new speech detected")
                                                continue
                                    
                                    except asyncio.CancelledError:
                                        # If initial TTS is cancelled, still update history
                                        gemma3_processor._update_history_with_complete_response(
                                            transcription, initial_text
                                        )
                                        logger.info("Initial TTS cancelled - new speech detected")
                                        continue
                            
                            except websockets.exceptions.ConnectionClosed:
                                break
                            except Exception as e:
                                logger.error(f"Error in speech processing: {e}")
                            finally:
                                # Clear TTS playing flag and tasks
                                await detector.set_tts_playing(False)
                                await detector.set_current_tasks()
                                
                    await asyncio.sleep(0.01)
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Error detecting speech: {e}")
                    await detector.set_tts_playing(False)
                    await detector.set_current_tasks()
        
        async def collect_remaining_text(streamer, initial_text):
            """Collect remaining text from the streamer"""
            collected_text = ""
            
            if streamer:
                try:
                    async for line in streamer.aiter_lines():
                        data = json.loads(line)
                        if "response" in data:
                            chunk = data["response"]
                            collected_text += chunk
                except asyncio.CancelledError:
                    raise
            # remove the initial_text from the remaining text
            return collected_text[len(initial_text):]
        
        async def receive_audio_and_images():
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    # Handle audio data
                    if "realtime_input" in data:
                        for chunk in data["realtime_input"]["media_chunks"]:
                            if chunk["mime_type"] == "audio/pcm":
                                audio_data = base64.b64decode(chunk["data"])
                                await detector.add_audio(audio_data)
                            # Only process image if TTS is not playing
                            elif chunk["mime_type"].startswith("image/") and not detector.tts_playing:
                                # Pass the image data through exactly as received
                                await gemma3_processor.set_image(chunk["data"])
                    
                    # Only process standalone image if TTS is not playing
                    if "image" in data and not detector.tts_playing:
                        # Pass the image data through exactly as received
                        await gemma3_processor.set_image(data["image"])
                        
                except Exception as e:
                    logger.error(f"Error receiving data: {e}")
                    if 'data' in locals():
                        logger.error(f"Message type: {type(data)}")
                        if isinstance(data, dict):
                            logger.error(f"Keys in message: {list(data.keys())}")
                    raise
        
        # Run tasks concurrently
        await asyncio.gather(
            receive_audio_and_images(),
            detect_speech_segments(),
            send_keepalive(),
            return_exceptions=True
        )
        
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed")
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        # Ensure TTS playing flag is cleared when connection ends
        await detector.set_tts_playing(False)

async def main():
    """Main function to start the WebSocket server"""
    try:
        # Initialize all processors ahead of time to load models
        transcriber = WhisperTranscriber.get_instance()
        gemma3_processor = Gemma3Processor.get_instance()  # Updated to use Gemma3
        tts_processor = GoogleTTSProcessor.get_instance()
        
        logger.info("Starting WebSocket server on 0.0.0.0:9073")
        # Add ping_interval and ping_timeout parameters
        async with websockets.serve(
            handle_client, 
            "0.0.0.0", 
            9073,
            ping_interval=20,    # Send ping every 20 seconds
            ping_timeout=60,     # Wait up to 60 seconds for pong response
            close_timeout=10     # Wait up to 10 seconds for close handshake
        ):
            logger.info("WebSocket server running on 0.0.0.0:9073")
            await asyncio.Future()  # Run forever
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
