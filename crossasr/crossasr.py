import os, time, random
import numpy as np
import pandas as pd
import json

import crossasr.constant
from crossasr.constant import INDETERMINABLE_TEST_CASE, SUCCESSFUL_TEST_CASE, FAILED_TEST_CASE
from crossasr.constant import DATA_DIR, EXECUTION_TIME_DIR, CASE_DIR
from crossasr.constant import AUDIO_DIR, TRANSCRIPTION_DIR

from crossasr.utils import preprocess_text
from crossasr.utils import make_dir, read_json, save_execution_time, get_execution_time
from crossasr.text import Text

from crossasr.tts import TTS
from crossasr.asr import ASR

from jiwer import wer


class CrossASR:
    def __init__(self, tts: TTS, asrs: [ASR], output_dir: "", target_asr=None, recompute=False, num_iteration=5, time_budget=3600, max_num_retry=0, text_batch_size=None, seed=None, estimator=None):
        self.tts = tts
        self.asrs = asrs
        self.target_asr = target_asr
        
        self.output_dir = output_dir
        
        self.audio_dir = os.path.join(output_dir, DATA_DIR, AUDIO_DIR) 
        self.transcription_dir = os.path.join(output_dir, DATA_DIR, TRANSCRIPTION_DIR)
        self.init_directory()
        
        ## TODO: make init directory for execution time and case
        self.execution_time_dir = os.path.join(output_dir, EXECUTION_TIME_DIR)
        self.case_dir = os.path.join(output_dir, CASE_DIR)
        self.recompute = recompute
        self.num_iteration = num_iteration
        self.time_budget = time_budget
        self.max_num_retry = max_num_retry
        self.text_batch_size = text_batch_size
        self.estimator = estimator
        self.outputfile_failed_test_case = self.get_outputfile_for_failed_test_case()
        self.valid_data = None

        if seed :
            crossasr.utils.set_seed(seed)

        ## TODO: convert print into global logging
        
    def init_directory(self) :
        # init directory for save the audio
        make_dir(os.path.join(self.audio_dir, self.tts.getName()))

        # init directory for save the transcription
        for asr in self.asrs :
            make_dir(os.path.join(self.transcription_dir, self.tts.getName(), asr.getName()))

    def get_outputfile_for_failed_test_case(self) :
        asrs_dir = "_".join([asr.getName() for asr in self.asrs])
        result_dir = os.path.join(self.output_dir, 
                                "result", 
                                self.tts.getName(), 
                                asrs_dir, 
                                f"num_iteration_{self.num_iteration}", 
                                f"text_batch_size_{self.text_batch_size if self.text_batch_size else 'global' }")
        make_dir(result_dir)
        experiment_name = f"with-estimator-{self.estimator.getName().replace('/','-')}" if self.estimator else "without-estimator"
        return os.path.join(result_dir, experiment_name + ".json")


    def getTTS(self) :
        return self.tts

    def setTTS(self, tts: TTS) :
        self.tts = tts

    def getASRS(self) :
        return self.asrs
    
    def addASR(self, asr: ASR) :
        for curr_asr in self.asrs :
            if asr.getName() == curr_asr.getName() :
                # asr is already on the list of asrs
                return
        self.asrs.append(asr)
    
    def removeASR(self, asr_name: str):
        for i, asr in enumerate(self.asrs) :
            if asr_name == asr.getName() :
                break
        del self.asrs[i]

    def getOutputDir(self):
        return self.audio_dir

    def setOutputDir(self, output_dir: str) :
        self.output_dir = output_dir

        self.audio_dir = os.path.join(output_dir, DATA_DIR, AUDIO_DIR)
        self.transcription_dir = os.path.join(output_dir, DATA_DIR, TRANSCRIPTION_DIR)
        self.execution_time_dir = os.path.join(output_dir, EXECUTION_TIME_DIR)
        self.case_dir = os.path.join(output_dir, CASE_DIR)

    def caseDeterminer(self, text:str, transcriptions: str): 
        # word error rate
        wers = {}

        is_determinable = False

        for k, transcription in transcriptions.items():
            word_error_rate = wer(text, transcription)
            wers[k] = word_error_rate
            if word_error_rate == 0:
                is_determinable = True

        case = {}
        if is_determinable:
            for k in transcriptions.keys():
                if wers[k] == 0:
                    case[k] = SUCCESSFUL_TEST_CASE
                else:
                    case[k] = FAILED_TEST_CASE
        else:
            for k in transcriptions.keys():
                case[k] = INDETERMINABLE_TEST_CASE

        return case

    def saveCase(self, case_dir: str, tts_name: str, asr_name: str, filename:str, case:str) :
        case_dir = os.path.join(case_dir, tts_name, asr_name)
        make_dir(case_dir)
        fpath = os.path.join(case_dir, filename + ".txt")
        file = open(fpath, "w+")
        file.write(case)
        file.close()

    def getCase(self, case_dir: str, tts_name: str, asr_name: str, filename: str):
        case_dir = os.path.join(case_dir, tts_name, asr_name)
        fpath = os.path.join(case_dir, filename + ".txt")
        file = open(fpath, "r")
        case = int(file.readlines()[0][0])
        file.close()

        return case


    def printResult(self, text:str, filename:str) :
        
        print()
        print(f"TTS: {self.tts.getName()}")
        print(f"ASRs: {[asr.getName() for asr in self.asrs]}")
        print()
        print(f"Input text: {text}")
        print()
        print(f"Transcription:")
        for asr in self.asrs :
            transcription_dir = os.path.join(
                self.transcription_dir, self.getTTS().getName())
            transcription = asr.loadTranscription(
                transcription_dir=transcription_dir, filename=filename)
            print(f"\t {asr.getName()}: {preprocess_text(transcription)}")
        print()
        print(f"Cases: ")
        for asr in self.asrs:
            case = self.getCase(self.case_dir, self.tts.getName(), asr.getName(), filename)
            if case == FAILED_TEST_CASE :
                print(f"\t {asr.getName()}: failed test case")
            elif case == SUCCESSFUL_TEST_CASE :
                print(f"\t {asr.getName()}: successful test case")
            else :
                print(f"\t {asr.getName()}: indeterminable test case")
        print()

    def printStatistic(self) :
        print()
        print("Number of Failed Test Case Found")
        for i, res in enumerate(self.result["number_of_failed_test_cases_per_asr"]) :
            print("Iteration: ", i)
            for k, v in res.items():
                print(f"\t{k}: {v}")

        # for k, v in self.result["number_of_failed_test_cases_per_asr"].items() :
        #     print(f"\t{k}: {v[-1]}")
        # print(f"\tTotal: {self.result['number_of_failed_test_cases_all'][-1]}")
        print()

    def processText(self, text: str, filename: str) :
        """
        Run CrossASR on a single text
        Description: Given a sentence as input, the program will generate a test case. The program needs some parameters, i.e. a TTS and ASRs used
        :params text:
        :params filename:
        :returns case:
        :returns execution time:
        """
        execution_time = 0.

        directory = os.path.join(self.execution_time_dir, AUDIO_DIR, self.getTTS().getName())
        make_dir(directory)
        time_for_generating_audio_fpath = os.path.join(directory, filename + ".txt")
        
        audio_fpath = self.getTTS().getAudioPath(
            text=text, audio_dir=self.audio_dir, filename=filename)
        
        if self.recompute or not os.path.exists(audio_fpath):
            # print(audio_fpath)
            start_time = time.time()
            self.getTTS().generateAudio(text=text, audio_fpath=audio_fpath)
            save_execution_time(fpath=time_for_generating_audio_fpath, execution_time=time.time() - start_time)
        
        ## add execution time for generating audio
        execution_time += get_execution_time(
            fpath=time_for_generating_audio_fpath)    
        
        transcription_dir = os.path.join(self.transcription_dir, self.getTTS().getName())
        
        transcriptions = {}
        for asr in self.asrs :
            directory = os.path.join(
                self.execution_time_dir, TRANSCRIPTION_DIR, self.getTTS().getName(), asr.getName())
            make_dir(directory)
            time_for_recognizing_audio_fpath = os.path.join(
                directory, filename + ".txt")

            if self.recompute or not os.path.exists(time_for_recognizing_audio_fpath):
                start_time = time.time()
                # TODO:  
                # change recognize audio -> input audio instead of fpath
                # audio = asr.loadAudio(audio_fpath=audio_fpath)
                # transcription = asr.recognizeAudio(audio=audio)
                # asr.saveTranscription(transcription_fpath, transcription)
                transcription = asr.recognizeAudio(audio_fpath=audio_fpath)
                asr.setTranscription(transcription)
                asr.saveTranscription(transcription_dir=transcription_dir, filename=filename)
                save_execution_time(fpath=time_for_recognizing_audio_fpath, execution_time=time.time() - start_time)
            
            transcription = asr.loadTranscription(
                transcription_dir=transcription_dir, filename=filename)
            num_retry = 0
            while transcription == "" and num_retry < self.max_num_retry :
                start_time = time.time()
                asr.recognizeAudio(audio_fpath=audio_fpath)
                asr.saveTranscription(
                    transcription_dir=transcription_dir, filename=filename)
                save_execution_time(
                    fpath=time_for_recognizing_audio_fpath, execution_time=time.time() - start_time)
                transcription = asr.loadTranscription(
                    transcription_dir=transcription_dir, filename=filename)

                if asr.getName() == "wit" :
                    random_number = float(random.randint(9, 47))/10.
                    time.sleep(random_number)

                num_retry += 1

            transcriptions[asr.getName()] = preprocess_text(transcription)

            ## add execution time for generating audio
            execution_time += get_execution_time(
                fpath=time_for_recognizing_audio_fpath)    
            

        cases = self.caseDeterminer(text, transcriptions)
        # if sum(cases.values()) == 0 :
        #     print(text)
        #     print(transcriptions["wav2vec2"])
        #     print(cases)
        #     print()
        
        for asr_name, case in cases.items() :
            self.saveCase(self.case_dir, self.getTTS().getName(), asr_name, filename, str(case))

        # print(f"Execution time: {execution_time}")
        return cases, execution_time


    def processTextBatch(self, curr_texts: [Text], processed_texts: [Text], cases):
        start_time = time.time()
        curr_cases = []
        
        if self.estimator and len(processed_texts) > 0:
            labels = get_labels_from_cases(cases)
            self.trainEstimator(processed_texts, labels)
            # print(f"Length texts: {len(curr_texts)}")
            # start_time_classifier = time.time()
            curr_texts = self.rank(curr_texts)
            # end_time_classifier = time.time()
            # print({f"Time for prediciton: {end_time_classifier-start_time_classifier}s"})

        execution_time = 0.

        i = 0
        for text in curr_texts:
            # print("================")
            # print(f"{text.getId()}")
            case, exec_time = self.processText(
                text=text.getText(), filename=f"{text.getId()}")
            curr_cases.append(case)
            execution_time += exec_time
            i += 1
            if execution_time + time.time() - start_time > self.time_budget:
                # print(f"Number of processed texts {i}")
                break
        
        curr_processed_texts = curr_texts[:i]
        unprocessed_texts = curr_texts[i:]

        return curr_cases, curr_processed_texts, unprocessed_texts

    def setCorpus(self, texts: [Text]) :
        self.corpus = texts
        self.initiateVariables()

    
    def initiateVariables(self) :
        self.remaining_texts = self.corpus
        self.processed_texts = []
        self.cases = []
        # self.num_failed_test_cases = []
        # self.num_failed_test_cases_per_asr = {}
        # for asr in self.asrs:
        #     self.num_failed_test_cases_per_asr[asr.getName()] = []
        
        self.num_failed_test_cases_per_asr = []
        
        self.result = {}


    def runOneIteration(self) :
        if self.text_batch_size:
            curr_texts = self.remaining_texts[:self.text_batch_size]
            self.remaining_texts = self.remaining_texts[self.text_batch_size:]
        else:  # use global visibility
            curr_texts = self.remaining_texts

        if len(curr_texts) > 0:

            curr_cases, curr_processsed_texts, unprocessed_texts = self.processTextBatch(
                curr_texts, self.processed_texts, self.cases)
            self.cases.extend(curr_cases)
            self.processed_texts.extend(curr_processsed_texts)
            if self.text_batch_size:
                self.remaining_texts.extend(unprocessed_texts)
            else:
                self.remaining_texts = unprocessed_texts

            # self.num_failed_test_cases.append(
            #     calculate_cases(self.cases, mode=FAILED_TEST_CASE))
            self.num_failed_test_cases_per_asr.append({})
            for asr in self.asrs:
                self.num_failed_test_cases_per_asr[-1][asr.getName()] = calculate_cases_per_asr(
                    self.cases, mode=FAILED_TEST_CASE, asr_name=asr.getName())
            
            # shuffle the remaining texts
            np.random.shuffle(self.remaining_texts)

            # self.result["number_of_failed_test_cases_all"] = self.num_failed_test_cases
            self.result["number_of_failed_test_cases_per_asr"] = self.num_failed_test_cases_per_asr
        else:
            print("Texts are not enough!")

    def gatherValidTestCases(self) :
        wav_filenames = []
        transcripts = []
        input_texts = self.get_text_only(self.processed_texts)
        ids = self.get_id_only(self.processed_texts)
        source_audio_dir = os.path.join(self.audio_dir, self.tts.getName())
        for input_text, filename, case in zip(input_texts, ids, self.cases):
            if case[self.target_asr] != INDETERMINABLE_TEST_CASE: # either failed test cases or succesfull test cases
                src_audio_fpath = "/" + source_audio_dir + f"/{filename}.wav"
                wav_filenames.append(src_audio_fpath)
                transcripts.append(input_text)

        self.valid_data = pd.DataFrame(data={"wav_filename": wav_filenames, "transcript": transcripts})
    
    def getValidData(self):
        return self.valid_data

        
    def runAllIterations(self) :

        for i in range(self.num_iteration):
            # print(f"Iteration: {i+1}")
            self.runOneIteration()

        self.saveStatistic()

        if self.target_asr:
            self.saveFailedTestCases(self.processed_texts, self.cases)


    def saveStatistic(self) :
        with open(self.outputfile_failed_test_case, 'w') as outfile:
            json.dump(self.result, outfile, indent=2, sort_keys=True)


    def processCorpus(self, texts: [Text]):
        """
        Run CrossASR on a corpus
        given a corpus, which is a list of sentences, the CrossASR generates test cases.
        :param texts: a list of tuple(sentence, id)
        """
        
        remaining_texts = texts
        curr_texts = []
        processed_texts = []
        cases = []
        num_failed_test_cases = []
        num_failed_test_cases_per_asr = {}
        num_processed_texts = []
        for asr in self.asrs:
            num_failed_test_cases_per_asr[asr.getName()] = []
        
        for i in range(self.num_iteration):
            # print(f"Iteration: {i+1}")
            
            if self.text_batch_size :
                curr_texts = remaining_texts[:self.text_batch_size]
                remaining_texts = remaining_texts[self.text_batch_size:]
            else : # use global visibility
                curr_texts = remaining_texts

            if len(curr_texts) > 0 :
                
                curr_cases, curr_processsed_texts, unprocessed_texts = self.processTextBatch(curr_texts, processed_texts, cases)
                cases.extend(curr_cases)
                processed_texts.extend(curr_processsed_texts)
                if self.text_batch_size :
                    remaining_texts.extend(unprocessed_texts)
                else :
                    remaining_texts = unprocessed_texts

                num_failed_test_cases.append(calculate_cases(cases, mode=FAILED_TEST_CASE))
                for asr in self.asrs:
                    num_failed_test_cases_per_asr[asr.getName()].append(calculate_cases_per_asr(
                        cases, mode=FAILED_TEST_CASE, asr_name=asr.getName()))
                num_processed_texts.append(len(processed_texts))
            else :
                print("Texts are not enough!")
            
            # shuffle the remaining texts
            np.random.shuffle(remaining_texts)
        
        data = {}
        data["number_of_failed_test_cases_all"] = num_failed_test_cases
        data["number_of_failed_test_cases_per_asr"] = num_failed_test_cases_per_asr
        data["number_of_processed_texts"] = num_processed_texts
        with open(self.outputfile_failed_test_case, 'w') as outfile:
            json.dump(data, outfile, indent=2, sort_keys=True)

        if self.target_asr :
            self.saveFailedTestCases(processed_texts, cases)
    
    def saveFailedTestCases(self, processed_texts, cases) :
        failed_test_case_dir = os.path.join(self.output_dir, "failed_test_cases", self.tts.getName(), self.target_asr)
        make_dir(failed_test_case_dir)
        ids = self.get_id_only(processed_texts)
        input_texts = self.get_text_only(processed_texts)
        source_audio_dir = os.path.join(self.audio_dir, self.tts.getName())
        for input_text, filename, case in zip(input_texts, ids, cases)  :
            if case[self.target_asr] == FAILED_TEST_CASE :
                src_audio_fpath = source_audio_dir + f"/{filename}.wav"
                trgt_audio_fpath = failed_test_case_dir + f"/{filename}.wav"
                os.system(f"cp {src_audio_fpath} {trgt_audio_fpath}")
                ground_truth_file = failed_test_case_dir + f"/{filename}.txt"
                f = open(ground_truth_file, 'w+')
                f.write(input_text)
                f.close()


    def get_text_only(self, texts:[Text]) -> [str]:
        res = []
        for t in texts :
            res.append(t.getText()) 
        return res    


    def get_id_only(self, texts: [Text]) -> [str]:
        res = []
        for t in texts:
            res.append(t.getId())
        return res

    
    def trainEstimator(self, processed_texts, labels):
        train_texts = self.get_text_only(processed_texts)
        self.estimator.fit(train_texts, labels)

    def rank(self, texts:[Text]): 
        
        ranking = self.estimator.predict(self.get_text_only(texts))
        
        ## https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        texts = [x for _, x in reversed(sorted(zip(ranking, texts)))]
        
        return texts
        
def calculate_cases(cases, mode:str):
    count = 0
    for c in cases :
        for _, v in c.items() :
            if v == mode :
                count += 1
    return count


def calculate_cases_per_asr(cases, mode:str, asr_name:str):
    count = 0
    for c in cases:
        for k, v in c.items():
            if k == asr_name and v == mode:
                count += 1
    return count


def get_labels_from_cases(cases) :
    def determine_label(case) :
        if INDETERMINABLE_TEST_CASE in case.values() :
            return INDETERMINABLE_TEST_CASE
        if FAILED_TEST_CASE :
            return FAILED_TEST_CASE
        return SUCCESSFUL_TEST_CASE

    labels = []
    for case in cases :
        label = determine_label(case)
        labels.append(label)
    
    return labels
