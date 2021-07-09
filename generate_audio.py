import os 
import time
import random

from crossasr.constant import DATA_DIR, AUDIO_DIR, EXECUTION_TIME_DIR
from crossasr.utils import save_execution_time, make_dir, read_json, read_corpus

from utils import create_tts_by_name, set_seed

def generate(tts_name: str, corpus_path: str, data_dir: str, execution_time_dir:str):
    tts = create_tts_by_name(tts_name)
    
    audio_dir = os.path.join(data_dir, AUDIO_DIR)
    execution_time_dir = os.path.join(execution_time_dir, AUDIO_DIR, tts_name)
    make_dir(execution_time_dir)

    corpus = read_corpus(corpus_path)

    for i in range(0, 3) :
        c = corpus[i]
        text = c.getText()
        filename = c.getId()
        start = time.time()
        tts.generateAudio(text=text, audio_dir=audio_dir, filename=filename)
        end = time.time()
        execution_time = end - start
        fpath = os.path.join(execution_time_dir, filename + ".txt")
        save_execution_time(fpath=fpath, execution_time=execution_time)
        print(f"Generate {i}")
        i += 1
        if tts_name in ["google"]:
            random_number = float(random.randint(15, 40))/10.
            time.sleep(random_number)

if __name__ == "__main__" :
    json_config_path = "config.json"
    config = read_json(json_config_path)

    set_seed(config["seed"])

    corpus_path = os.path.join(config["output_dir"], config["corpus_fpath"])
    output_dir = config["output_dir"]
    data_dir = os.path.join(output_dir, DATA_DIR)
    execution_time_dir = os.path.join(output_dir, EXECUTION_TIME_DIR)

    tts_name = config["tts"]
    generate(tts_name, corpus_path, data_dir, execution_time_dir)

