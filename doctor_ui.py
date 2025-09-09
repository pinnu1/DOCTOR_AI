# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#VoiceBot UI with Gradio
import os
import gradio as gr

from brain_doctor import encode_image, analyze_image_with_query
from voice_patient import record_audio, transcribe_with_groq
from voice_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

load_dotenv()

system_prompt = """
You are roleplaying as a highly experienced medical doctor for an educational scenario. 
Your task is to carefully analyze the given image and provide a concise medical opinion 
as if you are speaking directly to a patient in a clinical setting. 
Do not mention being an AI, do not include disclaimers, and do not use markdown formatting, 
numbers, bullet points, or special characters. 
Begin your response immediately with your assessment using natural, empathetic, 
and professional medical language, starting with phrases like 
'With what I see, I think you have...'. 
Keep your response to a single cohesive paragraph with no more than five sentences. 
If offering a differential or possible remedies, include them naturally in your response 
without explicitly listing them. 
Your tone should mimic that of a caring, confident doctor speaking to a real person.
"""


def process_inputs(audio_filepath, image_filepath):
    speech_to_text_output = transcribe_with_groq(GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                                                 audio_filepath=audio_filepath,
                                                 stt_model="whisper-large-v3")

    # Handle the image input
    if image_filepath:
        doctor_response = analyze_image_with_query(query=system_prompt+speech_to_text_output, encoded_image=encode_image(image_filepath), model="meta-llama/llama-4-scout-17b-16e-instruct") #model="meta-llama/llama-4-maverick-17b-128e-instruct") 
    else:
        doctor_response = "No image provided for me to analyze"

    voice_of_doctor = text_to_speech_with_gtts(input_text=doctor_response, output_filepath="final.mp3") 

    return speech_to_text_output, doctor_response, voice_of_doctor


#  interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio("Temp.mp3")
    ],
    title="Transformer based early diagnosis system"
    
)

iface.launch(debug=True)