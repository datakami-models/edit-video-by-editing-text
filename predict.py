# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from difflib import Differ
import ffmpeg
import os
from typing import Optional
import torch
from transformers import pipeline

from cog import BasePredictor, BaseModel, Input, Path

# MODEL = 'facebook/wav2vec2-large-960h-lv60-self'
# MODEL  = "facebook/wav2vec2-large-960h"
MODEL = "facebook/wav2vec2-base-960h"
# MODEL = "patrickvonplaten/wav2vec2-large-960h-lv60-self-4-gram"

os.environ['HUGGINGFACE_HUB_CACHE'] = 'models/'

class Output(BaseModel):
    video: Optional[Path]
    transcription: Optional[str]


class Predictor(BasePredictor):
    def speech_to_text(self, video_file_path):
        """
        Takes a video path to convert to audio, transcribe audio channel to text and char timestamps

        Using https://huggingface.co/tasks/automatic-speech-recognition pipeline
        """
        video_path = Path(video_file_path)
        print("Converting video to audio...")
        try:
            # convert video to audio 16k using PIPE to audio_memory
            audio_memory, _ = ffmpeg.input(video_path).output(
                '-', format="wav", ac=1, ar='16k').overwrite_output().global_args('-loglevel', 'quiet').run(capture_stdout=True)
        except Exception as e:
            raise RuntimeError("Error converting video to audio")

        print("Converting audio to transcript...")
        try:
            print(f'Transcribing via local model')
            output = self.speech_recognizer(
                audio_memory, return_timestamps="char",  chunk_length_s=10, stride_length_s=(4, 2))

            transcription = output["text"].lower()
            timestamps = [[chunk["text"].lower(), chunk["timestamp"][0].tolist(), chunk["timestamp"][1].tolist()]
                          for chunk in output['chunks']]

            return (transcription, timestamps)
        except Exception as e:
            raise RuntimeError("Error running inference with local model", e)


    def cut_timestamps_to_video(self, video_in, transcription, text_in, timestamps, split_at_word_level=True):
        """
        Given original video input, text transcript + timestamps,
        and edit ext cuts video segments into a single video
        """
        video_path = Path(video_in)
        video_file_name = video_path.stem
        if (video_in == None or text_in == None or transcription == None):
            raise ValueError("Inputs undefined")
        
        transcription_processed = transcription
        text_in_processed = text_in
        timestamps_processed = timestamps
        
        print("Comparing transcripts...")
        # we split the list of character-level timestamps on whitespace, i.e. into words
        if split_at_word_level:
            transcription_processed = transcription.split(" ")
            text_in_processed = text_in.split(" ")
            
            idx = 0
            words = {}
            for character, start, end in timestamps:
                if character != ' ':
                    if idx not in words.keys():
                        words[idx] = {}
                        words[idx]['word'] = ''
                        words[idx]['start'] = start
                    words[idx]['word'] += character
                    words[idx]['end'] = end
                else:
                    idx += 1
            timestamps_processed = [tuple(w.values()) for w in words.values()]
        
        # compare original transcription with edit text
        d = Differ()
        diff = d.compare(transcription_processed, text_in_processed)
        
        # remove all text aditions from diff
        filtered_diff = list(filter(lambda x: x[0] != '+', diff))

        print("Calculating video cutting points...")
        # grouping character timestamps so there are less cuts
        idx = 0
        video_chunks = {}
        for (a, b) in zip(filtered_diff, timestamps_processed):
            if a[0] != '-':
                if idx not in video_chunks:
                    video_chunks[idx] = []
                video_chunks[idx].append(b)
            else:
                idx += 1

        # after grouping, gets the lower and upper start and time for each group
        timestamps_to_cut = [[v[0][1], v[-1][2]] for v in video_chunks.values()]

        between_str = '+'.join(
            map(lambda t: f'between(t,{t[0]},{t[1]})', timestamps_to_cut))

        print("Creating new video...")
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
                  for token in filtered_diff]

        #return (tokens, output_video)
        return {'transcription_processed' : transcription_processed,
                'text_in_processed': text_in_processed,
                'timestamps_processed': timestamps_processed,
                'diff': diff,
                'filtered_diff': filtered_diff,
                'video_chunks': video_chunks,
                'timestamps_to_cut': timestamps_to_cut,
                'between_str': between_str,
                'tokens': tokens,
                'output_video': output_video
        }


    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

        # is cuda available?
        cuda = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        device = 0 if torch.cuda.is_available() else -1
        self.speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model=f'{MODEL}',
            tokenizer=f'{MODEL}',
            framework="pt",
            device=device,
        )

        videos_out_path = Path("./videos_out")
        videos_out_path.mkdir(parents=True, exist_ok=True)

    def predict(
        self,
        video_in: Path = Input(description="Video file to transcribe or edit"),
        mode: str = Input(description="Mode: either transcribe or edit", choices=['edit','transcribe'], default='transcribe'),    
        transcription: str = Input(description="When using mode 'edit', this should be the transcription of the desired output video. Use mode 'transcribe' to create a starting point.", default=None),
        split_at: str = Input(description="When using mode 'edit', split transcription at the word level or character level. Default: word level. Character level is more precise but can lead to matching errors.", choices=['word','character'], default='word')
    ) -> Output:

        if mode == 'edit' and transcription == None:
          raise ValueError("If you choose mode 'edit', you must provide parameter `transcription`.")

        transcription_video_in, timestamps = self.speech_to_text(video_in)
        
        if mode == 'transcribe':
          return Output(video=None, transcription=transcription_video_in)

        elif mode == 'edit':
          split_at_word_level = (split_at == 'word')
          cutting_result = self.cut_timestamps_to_video(video_in, transcription_video_in, transcription, timestamps, split_at_word_level)
          
          return Output(video=Path(cutting_result.get('output_video')), transcription=None)

        else:
          raise ValueError(f"Unknown mode {mode}")

