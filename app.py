import gradio as gr
import json
from difflib import Differ
import ffmpeg
import os
from pathlib import Path
import time
import aiohttp
import asyncio


# Set true if you're using huggingface inference API API https://huggingface.co/inference-api
API_BACKEND = True
# MODEL = 'facebook/wav2vec2-large-960h-lv60-self'
# MODEL  = "facebook/wav2vec2-large-960h"
MODEL = "facebook/wav2vec2-base-960h"
# MODEL = "patrickvonplaten/wav2vec2-large-960h-lv60-self-4-gram"
if API_BACKEND:
    from dotenv import load_dotenv
    import base64
    import asyncio
    load_dotenv(Path(".env"))

    HF_TOKEN = os.environ["HF_TOKEN"]
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    API_URL = f'https://api-inference.huggingface.co/models/{MODEL}'

else:
    import torch
    from transformers import pipeline

    # is cuda available?
    cuda = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = 0 if torch.cuda.is_available() else -1
    speech_recognizer = pipeline(
        task="automatic-speech-recognition",
        model=f'{MODEL}',
        tokenizer=f'{MODEL}',
        framework="pt",
        device=device,
    )

videos_out_path = Path("./videos_out")
videos_out_path.mkdir(parents=True, exist_ok=True)

samples_data = sorted(Path('examples').glob('*.json'))
SAMPLES = []
for file in samples_data:
    with open(file) as f:
        sample = json.load(f)
    SAMPLES.append(sample)
VIDEOS = list(map(lambda x: [x['video']], SAMPLES))

total_inferences_since_reboot = 415
total_cuts_since_reboot = 1539


async def speech_to_text(video_file_path):
    """
    Takes a video path to convert to audio, transcribe audio channel to text and char timestamps

    Using https://huggingface.co/tasks/automatic-speech-recognition pipeline
    """
    global total_inferences_since_reboot
    if (video_file_path == None):
        raise ValueError("Error no video input")

    video_path = Path(video_file_path)
    try:
        # convert video to audio 16k using PIPE to audio_memory
        audio_memory, _ = ffmpeg.input(video_path).output(
            '-', format="wav", ac=1, ar='16k').overwrite_output().global_args('-loglevel', 'quiet').run(capture_stdout=True)
    except Exception as e:
        raise RuntimeError("Error converting video to audio")

    ping("speech_to_text")
    last_time = time.time()
    if API_BACKEND:
        # Using Inference API https://huggingface.co/inference-api
        # try twice, because the model must be loaded
        for i in range(10):
            for tries in range(4):
                print(f'Transcribing from API attempt {tries}')
                try:
                    inference_reponse = await query_api(audio_memory)
                    print(inference_reponse)
                    transcription = inference_reponse["text"].lower()
                    timestamps = [[chunk["text"].lower(), chunk["timestamp"][0], chunk["timestamp"][1]]
                                  for chunk in inference_reponse['chunks']]

                    total_inferences_since_reboot += 1
                    print("\n\ntotal_inferences_since_reboot: ",
                          total_inferences_since_reboot, "\n\n")
                    return (transcription, transcription, timestamps)
                except Exception as e:
                    print(e)
                    if 'error' in inference_reponse and 'estimated_time' in inference_reponse:
                        wait_time = inference_reponse['estimated_time']
                        print("Waiting for model to load....", wait_time)
                        # wait for loading model
                        # 5 seconds plus for certanty
                        await asyncio.sleep(wait_time + 5.0)
                    elif 'error' in inference_reponse:
                        raise RuntimeError("Error Fetching API",
                                           inference_reponse['error'])
                    else:
                        break
            else:
                raise RuntimeError(inference_reponse, "Error Fetching API")
    else:

        try:
            print(f'Transcribing via local model')
            output = speech_recognizer(
                audio_memory, return_timestamps="char",  chunk_length_s=10, stride_length_s=(4, 2))

            transcription = output["text"].lower()
            timestamps = [[chunk["text"].lower(), chunk["timestamp"][0].tolist(), chunk["timestamp"][1].tolist()]
                          for chunk in output['chunks']]
            total_inferences_since_reboot += 1

            print("\n\ntotal_inferences_since_reboot: ",
                  total_inferences_since_reboot, "\n\n")
            return (transcription, transcription, timestamps)
        except Exception as e:
            raise RuntimeError("Error Running inference with local model", e)


async def cut_timestamps_to_video(video_in, transcription, text_in, timestamps):
    """
    Given original video input, text transcript + timestamps,
    and edit ext cuts video segments into a single video
    """
    global total_cuts_since_reboot

    video_path = Path(video_in)
    video_file_name = video_path.stem
    if (video_in == None or text_in == None or transcription == None):
        raise ValueError("Inputs undefined")

    d = Differ()
    # compare original transcription with edit text
    diff_chars = d.compare(transcription, text_in)
    # remove all text aditions from diff
    filtered = list(filter(lambda x: x[0] != '+', diff_chars))

    # filter timestamps to be removed
    # timestamps_to_cut = [b for (a,b) in zip(filtered, timestamps_var) if a[0]== '-' ]
    # return diff tokes and cutted video!!

    # groupping character timestamps so there are less cuts
    idx = 0
    grouped = {}
    for (a, b) in zip(filtered, timestamps):
        if a[0] != '-':
            if idx in grouped:
                grouped[idx].append(b)
            else:
                grouped[idx] = []
                grouped[idx].append(b)
        else:
            idx += 1

    # after grouping, gets the lower and upter start and time for each group
    timestamps_to_cut = [[v[0][1], v[-1][2]] for v in grouped.values()]

    between_str = '+'.join(
        map(lambda t: f'between(t,{t[0]},{t[1]})', timestamps_to_cut))

    if timestamps_to_cut:
        video_file = ffmpeg.input(video_in)
        video = video_file.video.filter(
            "select", f'({between_str})').filter("setpts", "N/FRAME_RATE/TB")
        audio = video_file.audio.filter(
            "aselect", f'({between_str})').filter("asetpts", "N/SR/TB")

        output_video = f'./videos_out/{video_file_name}.mp4'
        ffmpeg.concat(video, audio, v=1, a=1).output(
            output_video).overwrite_output().global_args('-loglevel', 'quiet').run()
    else:
        output_video = video_in

    tokens = [(token[2:], token[0] if token[0] != " " else None)
              for token in filtered]

    total_cuts_since_reboot += 1
    ping("video_cuts")
    print("\n\ntotal_cuts_since_reboot: ", total_cuts_since_reboot, "\n\n")
    return (tokens, output_video)


async def query_api(audio_bytes: bytes):
    """
    Query for Huggingface Inference API for Automatic Speech Recognition task
    """
    payload = json.dumps({
        "inputs": base64.b64encode(audio_bytes).decode("utf-8"),
        "parameters": {
            "return_timestamps": "char",
            "chunk_length_s": 10,
            "stride_length_s": [4, 2]
        },
        "options": {"use_gpu": False}
    }).encode("utf-8")
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, data=payload) as response:
            print("API Response: ", response.status)
            if response.headers['Content-Type'] == 'application/json':
                return await response.json()
            elif response.headers['Content-Type'] == 'application/octet-stream':
                return await response.read()
            elif response.headers['Content-Type'] == 'text/plain':
                return await response.text()
            else:
                raise RuntimeError("Error Fetching API")


def ping(name):
    url = f'https://huggingface.co/api/telemetry/spaces/radames/edit-video-by-editing-text/{name}'
    print("ping: ", url)

    async def req():
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                print("pong: ", response.status)
    asyncio.create_task(req())


# ---- Gradio Layout -----
video_in = gr.Video(label="Video file", elem_id="video-container")
text_in = gr.Textbox(label="Transcription", lines=10, interactive=True)
video_out = gr.Video(label="Video Out")
diff_out = gr.HighlightedText(label="Cuts Diffs", combine_adjacent=True)
examples = gr.Dataset(components=[video_in], samples=VIDEOS, type="index")

css = """
#cut_btn, #reset_btn { align-self:stretch; }
#\\31 3 { max-width: 540px; }
.output-markdown {max-width: 65ch !important;}
#video-container{
    max-width: 40rem;
}
"""
with gr.Blocks(css=css) as demo:
        transcription_var = gr.State()
    timestamps_var = gr.State()
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            # Edit Video By Editing Text
            This project is a quick proof of concept of a simple video editor where the edits
            are made by editing the audio transcription.
            Using the [Huggingface Automatic Speech Recognition Pipeline](https://huggingface.co/tasks/automatic-speech-recognition)
            with a fine tuned [Wav2Vec2 model using Connectionist Temporal Classification (CTC)](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)
            you can predict not only the text transcription but also the [character or word base timestamps](https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline.__call__.return_timestamps)
            """)

    with gr.Row():

        examples.render()

        def load_example(id):
            video = SAMPLES[id]['video']
            transcription = SAMPLES[id]['transcription'].lower()
            timestamps = SAMPLES[id]['timestamps']

            return (video, transcription, transcription, timestamps)

        examples.click(
            load_example,
            inputs=[examples],
            outputs=[video_in, text_in, transcription_var, timestamps_var],
            queue=False)
    with gr.Row():
        with gr.Column():
            video_in.render()
            transcribe_btn = gr.Button("Transcribe Audio")
            transcribe_btn.click(speech_to_text, [video_in], [
                text_in, transcription_var, timestamps_var])

    with gr.Row():
        gr.Markdown("""
        ### Now edit as text
        After running the video transcription, you can make cuts to the text below (only cuts, not additions!)""")

    with gr.Row():
        with gr.Column():
            text_in.render()
            with gr.Row():
                cut_btn = gr.Button("Cut to video", elem_id="cut_btn")
                # send audio path and hidden variables
                cut_btn.click(cut_timestamps_to_video, [
                    video_in, transcription_var, text_in, timestamps_var], [diff_out, video_out])

                reset_transcription = gr.Button(
                    "Reset to last trascription", elem_id="reset_btn")
                reset_transcription.click(
                    lambda x: x, transcription_var, text_in)
        with gr.Column():
            video_out.render()
            diff_out.render()
    with gr.Row():
        gr.Markdown("""
        #### Video Credits

        1. [Cooking](https://vimeo.com/573792389)
        1. [Shia LaBeouf "Just Do It"](https://www.youtube.com/watch?v=n2lTxIk_Dr0)
        1. [Mark Zuckerberg & Yuval Noah Harari in Conversation](https://www.youtube.com/watch?v=Boj9eD0Wug8)
        """)
demo.queue()
if __name__ == "__main__":
    demo.launch(debug=True)
