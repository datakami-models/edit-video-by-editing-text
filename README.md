# Edit video by editing text
A pipeline for fast video editing. The pipeline can be used in two modes: transcribe and edit.

- Demo by [@radames](https://github.com/radames) on [HuggingFace spaces](https://huggingface.co/spaces/radames/edit-video-by-editing-text)
- Demo by [@jd7h](https://github.com/jd7h) on [Replicate](https://replicate.com/jd7h/edit-video-by-editing-text/) with improved transcription matching.

## Modes

- transcribe: make a transcription of the video. Use this to create the input for 'edit' mode.
- edit: provide the desired transcription of the video as input. The pipeline will cut out the parts of the video that are missing in the transcription.
