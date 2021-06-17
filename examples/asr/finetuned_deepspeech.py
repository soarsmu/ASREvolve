from crossasr.asr import ASR
import utils


class FinetunedDeepSpeech(ASR):
    def __init__(self, name="finetuned_deepspeech"):
        ASR.__init__(self, name=name)

    def recognizeAudio(self, audio_fpath: str) -> str:
        transcription = utils.finetunedDeepspeechRecognizeAudio(audio_fpath)
        return transcription
