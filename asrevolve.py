import warnings

from numpy.core.numeric import cross
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os, sys, subprocess
from crossasr import CrossASR
import json
import utils

from asr.finetuned_deepspeech import FinetunedDeepSpeech


def trainDeepSpeech(f, mode="first"):
    cmd = f"docker exec -it gpu0-deepspeech sh -c 'cd DeepSpeech ; bash fine_tune/{mode}.sh' "

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    output = out.decode("utf-8").split("\n")
    for o in output:
        print(o)
        f.write(o + "\n")

def substract(df1, df2):
    test_set = set(df2["wav_filename"])
    df = df1[df1["wav_filename"].apply(lambda x: x not in test_set)]
    return df


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

    fine_tune_data_dir = os.path.join(config["output_dir"], "fine_tune_data")
    if not os.path.exists(fine_tune_data_dir):
        os.makedirs(fine_tune_data_dir)
    test_path = os.path.join(fine_tune_data_dir, "test.csv")
    train_path = os.path.join(fine_tune_data_dir, "train.csv")
    
    crossasr.runOneIteration()
    crossasr.gatherFailedTestCases()
    test = crossasr.getFailedData()
    test.to_csv(test_path, index=False)

    crossasr.runOneIteration()
    crossasr.gatherValidTestCases()
    valid_data = crossasr.getValidData()
    train = substract(valid_data, test)
    train.to_csv(train_path, index=False)

    # valid_data.to_csv(os.path.join(fine_tune_data_dir, "valid.csv"))

    trainDeepSpeech(f, "first")

    crossasr.removeASR("deepspeech")
    crossasr.addASR(FinetunedDeepSpeech())
    crossasr.setTargetASR("finetuned_deepspeech")
    crossasr.deleteASRTranscriptions("finetuned_deepspeech")

    for i in range(2, config["num_iteration"]) :
        print(f"ITERATION: {i}")
        f.write(f"ITERATION: {i}\n")
        crossasr.runOneIteration()
        crossasr.gatherValidTestCases()
        valid_data = crossasr.getValidData()
        train = substract(valid_data, test)
        train.to_csv(train_path, index=False)
        trainDeepSpeech(f, "subsequent")

    f.close()
    crossasr.saveStatistic()


    

    
