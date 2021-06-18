import warnings

from numpy.core.numeric import cross
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os, sys, subprocess
from crossasr import CrossASR
import json
import utils


from asr.wav2vec2 import Wav2Vec2
from asr.finetuned_deepspeech import FinetunedDeepSpeech


def trainDeepSpeech(mode="first"):
    cmd = f"docker exec -it gpu0-deepspeech sh -c 'cd DeepSpeech ; bash fine_tune/{mode}.sh' "

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    output = out.decode("utf-8").split("\n")
    for o in output:
        print(o)
        

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
    
    crossasr.runOneIteration()
    crossasr.gatherValidTestCases()
    valid_data = crossasr.getValidData()
    train = valid_data.copy() ## we use all data for finetuning
    test = valid_data.sample(frac=0.50)
    # train = valid_data.drop(test.index)
    fine_tune_data_dir = os.path.join(config["output_dir"], "fine_tune_data")
    if not os.path.exists(fine_tune_data_dir) : os.makedirs(fine_tune_data_dir)
    test_path = os.path.join(fine_tune_data_dir, "test.csv")
    train_path = os.path.join(fine_tune_data_dir, "train.csv")
    test.to_csv(test_path, index=False)
    train.to_csv(train_path, index=False)
    trainDeepSpeech("first")

    crossasr.removeASR("deepspeech")
    crossasr.addASR(FinetunedDeepSpeech())
    crossasr.setTargetASR("finetuned_deepspeech")
    crossasr.deleteASRTranscriptions("finetuned_deepspeech")
    
    for i in range(1, config["num_iteration"]) :
        crossasr.runOneIteration()
        crossasr.gatherValidTestCases()
        valid_data = crossasr.getValidData()
        train = valid_data.drop(test.index)
        test.to_csv(test_path, index=False)
        train.to_csv(train_path, index=False)
        trainDeepSpeech("subsequent")

    crossasr.saveStatistic()


    

    
