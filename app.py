import streamlit as st
import openai
import tempfile
import os
from pytube import YouTube
from moviepy.editor import AudioFileClip
import whisper

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="YouTube Quiz Generator", layout="centered")
st.title("🎥 YouTube Video to Quiz (with OpenAI)")

# User inputs
yt_url = st.text_input("Paste YouTube video URL:")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
num_qs = st.number_input("How many quiz questions do you want?", min_value=3, max_value=30, value=10)

if yt_url and openai_api_key:
    if st.button("Generate Quiz"):
        with st.spinner("Downloading video..."):
            yt = YouTube(yt_url)
            stream = yt.streams.filter(only_audio=True).first()
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, "audio.mp4")
            stream.download(output_path=temp_dir, filename="audio.mp4")

        with st.spinner("Extracting audio and transcribing..."):
            audio_clip = AudioFileClip(audio_path)
            audio_clip.write_audiofile(os.path.join(temp_dir, "audio.wav"))
            audio_clip.close()

            model = whisper.load_model("base")  # Change to "small"/"medium" if GPU available
            result = model.transcribe(os.path.join(temp_dir, "audio.wav"))
            transcript = result["text"]

        st.subheader("📜 Transcript Preview")
        st.write(transcript[:1000] + ("..." if len(transcript) > 1000 else ""))

        # Download transcript
        st.download_button("📥 Download Full Transcript", transcript, file_name="transcript.txt")

        with st.spinner("Generating Quiz with OpenAI..."):
            openai.api_key = openai_api_key

            prompt = f"""
            You are a teacher creating a quiz based on the following transcript.
            Focus on key ideas, important details, and deeper understanding, not trivial facts.
            Make the questions challenging and thought-provoking.
            Create exactly {num_qs} quiz questions with their answers.

            Transcript:
            {transcript}
            """

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert quiz maker."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            quiz = response["choices"][0]["message"]["content"]

        st.subheader("📝 Generated Quiz")
        st.write(quiz)

        st.success("Done! Copy your quiz or download transcript.")
