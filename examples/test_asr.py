import warnings

from numpy.core.numeric import cross
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os, sys
from crossasr import CrossASR
import json
import utils


from asr.wav2vec2 import Wav2Vec2


if __name__ == "__main__":

    config = utils.readJson(sys.argv[1]) # read json configuration file

    tts = utils.getTTS(config["tts"])
    asr_list = config["asrs"]
    asrs = utils.getASRS(asr_list)
    estimator = utils.getEstimator(config["estimator"]) if config["estimator"] else None

    crossasr = CrossASR(tts=tts, asrs=asrs, estimator=estimator, **utils.parseConfig(config))

    corpus_fpath = os.path.join(config["output_dir"], config["corpus_fpath"])
    texts = utils.readCorpus(corpus_fpath=corpus_fpath)
    crossasr.setCorpus(texts=texts)
    crossasr.runAllIterations()
    # crossasr.runOneIteration()
    # crossasr.printStatistic()

    # crossasr.removeASR("deepspeech")
    # new_asr = Wav2Vec2()
    # crossasr.addASR(new_asr)
    # # TODO:
    # # retrain DeepSpeech, exchange deepspeech

    # Retrain Deepspeech
    # prepare folder containing audio files and transcriptions
    # prepare train and test csv
    # update train csv for each iteration

    # crossasr.runOneIteration()
    crossasr.printStatistic()


    
