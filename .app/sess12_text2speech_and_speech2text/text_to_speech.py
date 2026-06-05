"""
=================================================================================================
Python script to demonstrate Text-to-Speech using scikit-learn-package
=================================================================================================
This program demonstrates Text-to-Speech using pyttsx3 package

Flow:
    1. Read in a text file
    2. Process the text in a Python program
    3. Text-to-Speach Engine generates audio
    4. Output an audio file (travel_output.wav)

Input / Ouput file locations:
    - input: files/passage.txt
    - output: filestravel_output.wav

Requirements:
    !pip install pyttsx3
"""

# -----------------------------------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------------------------------
import logging  # Optional to show what is happending at what time
import pyttsx3
import sys

from pathlib import Path

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------------
# 1. Optional Logging Configuration
# -----------------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------
# 2. Constants
# -----------------------------------------------------------------------------------------------
TEXT_FILE: Path = Path("../files/passage.txt")
OUTPUT_FILE: Path = Path("../files/travel_output.wav")
SPEAKING_RATE: int = 500  # Words per minute (normal conversation pace)
VOLUME: float = 1.0  # Full volume (range: 0.0 - 1.0)


# -----------------------------------------------------------------------------------------------
# 3. Functions
# -----------------------------------------------------------------------------------------------
def load_text(file_path: Path) -> str:
    # Optional loggin
    logger.info(f"Loading text from {file_path}...")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}."
                                f"\nPlease ensure the file location is valid")

    text: str = file_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Text file is empty."
                         f"\nAdd some text into the file.")

    # Optional logging
    logger.info(f"Successfully loaded text from {file_path}...")
    return text


def initialise_engine() -> pyttsx3.Engine:
    # Optional logging
    logger.info(f"Initialising text-to-speech engine...")

    try:
        engine: pyttsx3.Engine = pyttsx3.init()
    except Exception as e:
        raise RuntimeError(f"Failed to initialise text-to-speech engine: {e}") from exc

    # Set a moderate speaking speaking rate (WPM)
    engine.setProperty("rate", SPEAKING_RATE)

    # Set the volume to max
    engine.setProperty("volume", VOLUME)

    # Optional logging
    logger.info(f"Engine ready | Rate: {SPEAKING_RATE} | Volume: {VOLUME:.1f}")

    return engine


def save_audio(engine: pyttsx3.Engine, text: str, output_path: Path) -> None:
    # Optional logging
    logger.info(f"Saving audio to {output_path}...")

    # Ensure the output director exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # pyttsx3 saves audio by queuing the text and then running the entire engine
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()

    if not output_path.exists():
        raise IOError(f"Audio file was not created at: {output_path}."
                      f"\nPlease ensure that pyttsx3 has permission to write to the output file directory")

    # Optional logging
    logger.info(f"Audio file was saved succesfully at: {output_path}.")

def speak_text(engine: pyttsx3.Engine, text: str) -> None:
    # Optional logging
    logger.info(f"Playing speech through speakers...")

    engine.say(text)
    engine.runAndWait()

    # Optional logging
    logger.info(f"Audio playback complete")

# -----------------------------------------------------------------------------------------------
# 4. Main Execution Function
# -----------------------------------------------------------------------------------------------
def main() -> None:
    print()
    print("-" * 50)
    print("     TEXT TO SPEECH DEMONSTRATION     ")
    print("-" * 50)
    print()

    # -----------------------------------------------------------------------------------------------
    # Step I. Load the text
    # -----------------------------------------------------------------------------------------------
    print("Loading text...")
    try:
        text = load_text(TEXT_FILE)
    except (FileNotFoundError, ValueError) as e:
        # Optional logging
        logger.info(f"Could not load text: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------------------------------
    # Step II. Display the text on the screen
    # -----------------------------------------------------------------------------------------------
    print()
    print("-" * 50)
    print("     PASSAGE      ")
    print("-" * 50)
    print(text)
    print("-" * 50)
    print()

    # -----------------------------------------------------------------------------------------------
    # Step III. Initialise the TTS engine
    # -----------------------------------------------------------------------------------------------
    try:
        engine: pyttsx3.Engine = initialise_engine()
    except RuntimeError as e:
        # Optional logging
        logger.info(f"Engine initialisation failed: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------------------------------
    # Step IV. Save the audio to a .wav file
    # -----------------------------------------------------------------------------------------------
    print("Generating speech from text...")
    try:
        save_audio(engine, text, OUTPUT_FILE)
    except IOError as e:
        # Optional logging
        logger.info(f"Could not save audio to: {e}")
        sys.exit(1)

    print(f"Audio saved to: {OUTPUT_FILE}")

    # -----------------------------------------------------------------------------------------------
    # Step V. Speak the text aloud
    # -----------------------------------------------------------------------------------------------
    print("-" * 50)
    print("Speaking text now - please ensure your speakers are on...")
    print("-" * 50)
    try:
        speak_text(engine, text)
    except Exception as e:
        # Optional logging
        logger.info(f"Could not speak text: {e}")
        sys.exit(1)

    print("\nEND OF DEMONSTRATION")

# -----------------------------------------------------------------------------------------------
# 5. Run the script by invoking its main() function
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()