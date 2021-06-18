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


def trainDeepSpeech(f, mode="first"):
    cmd = f"docker exec -it gpu0-deepspeech sh -c 'cd DeepSpeech ; bash fine_tune/{mode}.sh' "

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    output = out.decode("utf-8").split("\n")
    for o in output:
        print(o)
        f.write(o + "\n")


if __name__ == "__main__":

    

    config = utils.readJson(sys.argv[1]) # read json configuration file

    tts = utils.getTTS(config["tts"])
    asr_list = config["asrs"]
    asrs = utils.getASRS(asr_list)
    estimator = utils.getEstimator(config["estimator"]) if config["estimator"] else None

    crossasr = CrossASR(tts=tts, asrs=asrs, estimator=estimator, **utils.parseConfig(config))
    log_fpath = crossasr.outputfile_failed_test_case
    log_fpath = log_fpath.replace(".json", ".txt")
    print(f"Log is saved at {log_fpath}")
    f = open(log_fpath, "w+")

    corpus_fpath = os.path.join(config["output_dir"], config["corpus_fpath"])
    texts = utils.readCorpus(corpus_fpath=corpus_fpath)
    crossasr.setCorpus(texts=texts)
    
    # crossasr.runAllIterations()
    # crossasr.printStatistic()
    
    # crossasr.runOneIteration()
    # crossasr.printStatistic()
    # crossasr.removeASR("deepspeech")
    # new_asr = Wav2Vec2()
    # crossasr.addASR(new_asr)
    # crossasr.runOneIteration()
    # crossasr.printStatistic()
    
    # TODO:
    # prepare folder containing audio files and transcriptions
    # mount the folder in the source code for retraining Deepspeech
    #   (loop for each iteration)
    #   run crossasr for 1 iteration
    #   save audio files and transcription in the mounted folder
    #   prepare train and test csv, where the test test is fix for each iteration
    #   update train csv for each iteration
    #   train deepspeech
    #   exchange deepspeech with the fine-tuned model
    #   save the WER from fine-tuned model
    #   (until a number of iterations is already performed)

    crossasr.runOneIteration()
    crossasr.gatherValidTestCases()
    valid_data = crossasr.getValidData()
    test = valid_data.sample(frac=0.50)
    train = valid_data.drop(test.index)
    fine_tune_data_dir = os.path.join(config["output_dir"], "fine_tune_data")
    if not os.path.exists(fine_tune_data_dir) : os.makedirs(fine_tune_data_dir)
    test_path = os.path.join(fine_tune_data_dir, "test.csv")
    train_path = os.path.join(fine_tune_data_dir, "train.csv")
    test.to_csv(test_path, index=False)
    train.to_csv(train_path, index=False)
    trainDeepSpeech(f, "first")

    crossasr.removeASR("deepspeech")
    crossasr.addASR(FinetunedDeepSpeech())
    crossasr.setTargetASR("finetuned_deepspeech")
    crossasr.deleteASRTranscriptions("finetuned_deepspeech")
    
    for i in range(1, config["num_iteration"]) :
        # print("XXXXX")
        # print("XXXXX")
        # print("XXXXX")
        # print(f"ITERATION: {i}")
        f.write("XXX\n")
        f.write("XXX\n")
        f.write("XXX\n")
        f.write(f"ITERATION: {i}\n")
        crossasr.runOneIteration()
        crossasr.gatherValidTestCases()
        valid_data = crossasr.getValidData()
        train = valid_data.drop(test.index)
        test.to_csv(test_path, index=False)
        train.to_csv(train_path, index=False)
        trainDeepSpeech(f, "subsequent")

    crossasr.saveStatistic()

    f.close()
    # print(valid_data)


    

    
