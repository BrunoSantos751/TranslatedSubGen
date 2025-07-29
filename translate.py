import os
import sys
from dotenv import load_dotenv
from pydub import AudioSegment
import deepl
import whisper
import moviepy.config
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from pysrt import open as opensrt

# Load environment variables from .env file
load_dotenv()

# Configuration parameters from .env
WHISPER_MODEL_SIZE = "small"  # Available sizes: 'tiny', 'base', 'small', 'medium', 'large'
DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")
SOURCE_LANG = os.getenv("SOURCE_LANG", "KO")
TARGET_LANG = os.getenv("TARGET_LANG", "PT-BR")

# Subtitle tuning parameters
MIN_SUB_DURATION_MS = 1500   # Minimum subtitle duration
MAX_SUB_DURATION_MS = 7000   # Maximum subtitle duration
MERGE_GAP_THRESHOLD_MS = 500 # Maximum silence gap to allow merging
MAX_CHARS_PER_LINE = 60      # Max characters per subtitle line
MAX_CHARS_PER_SUBTITLE = 120 # Max total characters per subtitle

# Replace this path with the ACTUAL path to your magick.exe
moviepy.config.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"


def format_timestamp(milliseconds):
    """Converts milliseconds to SRT timestamp format (HH:MM:SS,ms)."""
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    seconds = milliseconds // 1000
    milliseconds %= 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"


def translate_with_deepl(text, translator_obj, source_lang=SOURCE_LANG, target_lang=TARGET_LANG):
    """Translate text using the DeepL API."""
    try:
        translated_text = translator_obj.translate_text(
            text,
            source_lang=source_lang.upper(),
            target_lang=target_lang.upper()
        ).text
        return translated_text
    except deepl.exceptions.DeepLException as e:
        print(f"DeepL translation error: {e}")
        return f"[TRANSLATION ERROR]: {text}"
    except Exception as e:
        print(f"Unexpected translation error: {e}")
        return f"[UNEXPECTED ERROR]: {text}"


def wrap_text(text, max_chars_per_line):
    """Wraps text into multiple lines based on character limit per line."""
    words = text.split()
    lines = []
    current_line = []
    current_line_length = 0

    for word in words:
        if current_line_length + len(word) + (1 if current_line else 0) <= max_chars_per_line:
            current_line.append(word)
            current_line_length += len(word) + (1 if current_line else 0)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_line_length = len(word)
    lines.append(" ".join(current_line))
    return "\n".join(lines)


def post_process_segments(segments, min_sub_duration, max_sub_duration, merge_gap_threshold, max_chars_per_subtitle):
    """
    Optimizes transcription segments for subtitle display:
    - Merges short and nearby segments
    - Ensures minimum and maximum duration
    - Limits subtitle character length
    """
    processed_subs = []
    if not segments:
        return []

    current_sub = {
        "text": segments[0]["text"].strip(),
        "start_ms": int(segments[0]["start"] * 1000),
        "end_ms": int(segments[0]["end"] * 1000)
    }

    for i in range(1, len(segments)):
        segment = segments[i]
        next_text = segment["text"].strip()
        next_start_ms = int(segment["start"] * 1000)
        next_end_ms = int(segment["end"] * 1000)

        gap_to_next = next_start_ms - current_sub["end_ms"]
        merged_text_length = len(current_sub["text"] + " " + next_text)
        merged_duration = next_end_ms - current_sub["start_ms"]

        if (gap_to_next < merge_gap_threshold and
            merged_text_length <= max_chars_per_subtitle and
            merged_duration <= max_sub_duration):
            current_sub["text"] += " " + next_text
            current_sub["end_ms"] = next_end_ms
        else:
            if (current_sub["end_ms"] - current_sub["start_ms"]) < min_sub_duration:
                current_sub["end_ms"] = current_sub["start_ms"] + min_sub_duration
            processed_subs.append(current_sub)
            current_sub = {
                "text": next_text,
                "start_ms": next_start_ms,
                "end_ms": next_end_ms
            }

    if current_sub:
        if (current_sub["end_ms"] - current_sub["start_ms"]) < min_sub_duration:
            current_sub["end_ms"] = current_sub["start_ms"] + min_sub_duration
        processed_subs.append(current_sub)

    return processed_subs


def generate_srt_subtitles_with_whisper_deepl(video_path, output_srt_path="legendas_deepl.srt",
                                              audio_language_code=SOURCE_LANG,
                                              subtitle_language_code=TARGET_LANG):
    """
    Generates translated and synced subtitles from a video using Whisper + DeepL.
    Output is saved in .srt format.
    """
    print("Extracting audio from video...")
    video_clip = VideoFileClip(video_path)
    audio_temp_path = "temp_audio.wav"
    video_clip.audio.write_audiofile(audio_temp_path)
    print(f"Audio saved to: {audio_temp_path}")

    print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
    model = whisper.load_model(WHISPER_MODEL_SIZE)

    print("Transcribing audio...")
    result = model.transcribe(audio_temp_path, language=audio_language_code)
    transcribed_segments = result["segments"]
    print(f"{len(transcribed_segments)} segments transcribed.")

    print("Post-processing segments...")
    processed = post_process_segments(
        transcribed_segments,
        MIN_SUB_DURATION_MS,
        MAX_SUB_DURATION_MS,
        MERGE_GAP_THRESHOLD_MS,
        MAX_CHARS_PER_SUBTITLE
    )

    print("Connecting to DeepL...")
    try:
        translator = deepl.Translator(DEEPL_AUTH_KEY)
        _ = translator.get_usage()
        print("DeepL authenticated successfully.")
    except Exception as e:
        print(f"DeepL authentication error: {e}")
        sys.exit(1)

    print("Translating and writing subtitles...")
    with open(output_srt_path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(processed):
            original_text = sub["text"]
            translated = translate_with_deepl(original_text, translator, audio_language_code, subtitle_language_code)
            wrapped = wrap_text(translated, MAX_CHARS_PER_LINE)
            start = format_timestamp(sub["start_ms"])
            end = format_timestamp(sub["end_ms"])

            f.write(f"{i+1}\n{start} --> {end}\n{wrapped}\n\n")
            print(f"[{i+1}] {start} --> {end} | {wrapped.replace(chr(10), ' ')}")

    os.remove(audio_temp_path)
    print(f"SRT file generated: {output_srt_path}")


if __name__ == "__main__":
    video_input = "target_video.mp4"
    output_srt_file = "target_video_subtittle.srt"

    if os.path.exists(video_input):
        generate_srt_subtitles_with_whisper_deepl(video_input, output_srt_file)
    else:
        print(f"Video file '{video_input}' not found. Please check the path.")
