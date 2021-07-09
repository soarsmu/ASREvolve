from crossasr.asr import ASR
import utils 

class Wav2Vec2(ASR):
    def __init__(self, name="wav2vec2"):
        ASR.__init__(self, name=name)
            
    def recognizeAudio(self, audio_fpath: str) -> str:
        transcription = utils.wav2vec2RecognizeAudio(audio_fpath)
        return transcription
