import os
import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import pipeline
import anthropic
from pydub import AudioSegment
from gtts import gTTS
import warnings

class LanguageBarrierlessSystem:
    def __init__(self, source_lang, target_lang, claude_api_key):
        """
        Initialize the language barrier-less system.
        
        Args:
            source_lang (str): Source language code (e.g., 'en', 'es', 'fr')
            target_lang (str): Target language code
            claude_api_key (str): API key for Claude
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.reference_audio = "reference_sample.wav"  # Default reference audio
        
        # Initialize Claude client
        self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
        
        # Initialize Whisper for STT
        print("Initializing Whisper speech recognition model...")
        self.whisper = pipeline("automatic-speech-recognition", 
                                model="openai/whisper-large-v3-turbo",
                                device="cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"System initialized for translation from {source_lang} to {target_lang}")
    
    def extract_voice_characteristics(self, audio_file):
        """
        Extract voice characteristics from the input audio.
        
        Args:
            audio_file (str): Path to the input audio file
            
        Returns:
            dict: Voice characteristics (pitch, energy, etc.)
        """
        print(f"Extracting voice characteristics from {audio_file}...")
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extract pitch (F0 contour)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                    fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'),
                                                    sr=sr)
        
        # Extract energy/volume
        energy = np.array([sum(abs(y[i:i+512])) for i in range(0, len(y), 512)])
        
        # Extract speaking rate (approximation)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        print("Voice characteristics extracted successfully")
        
        return {
            "f0": f0,
            "energy": energy,
            "tempo": tempo,
            "sample_rate": sr,
            "duration": librosa.get_duration(y=y, sr=sr)
        }
    
    def speech_to_text(self, audio_file):
        """
        Convert speech to text using Whisper.
        
        Args:
            audio_file (str): Path to the input audio file
            
        Returns:
            str: Transcribed text
        """
        print(f"Transcribing audio from {audio_file}...")
        
        # Transcribe audio using Whisper
        result = self.whisper(audio_file, language=self.source_lang)
        transcription = result["text"]
        
        print(f"Transcription: {transcription}")
        return transcription
    
    def translate_and_correct(self, text):
        """
        Translate and correct text using Claude API.
        
        Args:
            text (str): Text to translate
            
        Returns:
            str: Translated and corrected text
        """
        print(f"Translating text from {self.source_lang} to {self.target_lang}...")
        
        prompt = f"""
        Translate the following text from {self.source_lang} to {self.target_lang}. 
        Preserve the tone, sentiment, and style of the original message.
        Make sure the translation sounds natural and fluent in {self.target_lang}.
        
        Original text: {text}
        
        Translation:
        """
        
        response = self.claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        translation = response.content[0].text
        print(f"Translation: {translation}")
        return translation
    
    def text_to_speech(self, text, voice_characteristics, output_file):
        """
        Convert text to speech using gTTS.
        
        Args:
            text (str): Text to convert to speech
            voice_characteristics (dict): Voice characteristics to preserve
            output_file (str): Path to save the output audio
            
        Returns:
            str: Path to the output audio file
        """
        print(f"Converting text to speech using gTTS...")
        
        # gTTS only outputs MP3 format, so we'll need to convert to WAV if needed
        is_wav = output_file.lower().endswith('.wav')
        mp3_path = output_file.replace('.wav', '.mp3') if is_wav else output_file
        
        # Generate speech with gTTS
        tts = gTTS(text=text, lang=self.target_lang[:2])
        tts.save(mp3_path)
        print(f"Generated speech saved to {mp3_path}")
        
        # Convert MP3 to WAV if the output file should be WAV
        if is_wav:
            try:
                print(f"Converting MP3 to WAV format...")
                sound = AudioSegment.from_mp3(mp3_path)
                sound.export(output_file, format="wav")
                print(f"Converted to WAV: {output_file}")
                
                # Apply voice characteristics to WAV file
                self._apply_voice_characteristics(output_file, voice_characteristics, output_file)
                
                # Optionally remove the intermediate MP3 file
                # os.remove(mp3_path)
            except Exception as e:
                print(f"Error converting MP3 to WAV: {str(e)}")
                print(f"Using MP3 output instead: {mp3_path}")
                return mp3_path
            
        return output_file
    
    def _apply_voice_characteristics(self, audio_file, voice_chars, output_file):
        """
        Apply voice characteristics to the generated speech.
        
        Args:
            audio_file (str): Path to the input audio file
            voice_chars (dict): Voice characteristics to apply
            output_file (str): Path to save the modified audio
        """
        print("Applying voice characteristics...")
        
        try:
            # Load the generated audio
            y, sr = librosa.load(audio_file, sr=None)
            
            # Here we would apply more sophisticated voice conversion
            # For now, we'll implement a simple pitch shift based on the original voice
            
            # Calculate average pitch of original voice (excluding NaN values)
            original_f0 = voice_chars["f0"]
            valid_f0 = original_f0[~np.isnan(original_f0)]
            if len(valid_f0) > 0:
                avg_pitch = np.mean(valid_f0)
                print(f"Original average pitch: {avg_pitch:.2f} Hz")
                
                # Simple pitch shifting (for demonstration purposes)
                # In a real implementation, you would use more advanced techniques
                # This is just to show the concept
                try:
                    from librosa.effects import pitch_shift
                    
                    # First, analyze the generated audio's pitch
                    generated_f0, _, _ = librosa.pyin(y, 
                                                    fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'),
                                                    sr=sr)
                    valid_generated_f0 = generated_f0[~np.isnan(generated_f0)]
                    
                    if len(valid_generated_f0) > 0:
                        avg_generated_pitch = np.mean(valid_generated_f0)
                        print(f"Generated average pitch: {avg_generated_pitch:.2f} Hz")
                        
                        # Calculate number of semitones to shift
                        if avg_generated_pitch > 0 and avg_pitch > 0:
                            semitones = 12 * np.log2(avg_pitch / avg_generated_pitch)
                            print(f"Shifting pitch by {semitones:.2f} semitones")
                            
                            # Apply pitch shift
                            y_shifted = pitch_shift(y, sr=sr, n_steps=semitones)
                            
                            # Save shifted audio
                            sf.write(output_file, y_shifted, sr)
                            print(f"Applied pitch shifting and saved to {output_file}")
                            return
                except Exception as e:
                    print(f"Error in pitch shifting: {str(e)}")
        except Exception as e:
            print(f"Error applying voice characteristics: {str(e)}")
        
        # If we get here, just save the original file
        print("Using original audio without modifications")
        sf.write(output_file, y, sr)
        
        print(f"Voice characteristics processing completed for {output_file}")
    
    def process_conversation(self, input_audio, output_audio):
        """
        Process a complete conversation turn.
        
        Args:
            input_audio (str): Path to the input audio file
            output_audio (str): Path to save the output audio
            
        Returns:
            str: Path to the output audio file
        """
        print(f"\n*** Processing conversation from {input_audio} ***\n")
        
        # Extract voice characteristics
        voice_chars = self.extract_voice_characteristics(input_audio)
        
        # Convert speech to text
        text = self.speech_to_text(input_audio)
        
        # Translate and correct the text
        translated_text = self.translate_and_correct(text)
        
        # Convert text to speech while preserving voice characteristics
        result_audio = self.text_to_speech(translated_text, voice_chars, output_audio)
        
        print(f"\n*** Conversation processing complete ***")
        print(f"Output saved to: {result_audio}")
        
        return result_audio

# Example usage
if __name__ == "__main__":
    # Check if input file exists
    input_file = "/Users/krishna/Downloads/PersonalProjects/BarrierLessConnection/input_english.wav"
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please check the file path and try again.")
        exit(1)
    
    # Get Claude API key from environment variable
    claude_api_key = os.environ.get("CLAUDE_API_KEY")
    if not claude_api_key:
        print("Error: CLAUDE_API_KEY environment variable not set")
        print("Please set it using: export CLAUDE_API_KEY='your_api_key_here'")
        exit(1)
    
    # Example with English to Spanish
    system = LanguageBarrierlessSystem(
        source_lang="en", 
        target_lang="es",
        claude_api_key=claude_api_key
    )
    
    # Process a sample conversation
    result = system.process_conversation(
        input_audio=input_file,
        output_audio="output_spanish.wav"
    )