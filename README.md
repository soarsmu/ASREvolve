# Can Differential Testing Improve Automatic Speech Recognition Systems?

Accepted at IEEE ICSME 2021 NIER Track | [Paper PDF](https://mhilmiasyrofi.github.io/papers/ASREvolve.pdf) 
 
Due to the widespread adoption of Automatic Speech Recognition (ASR) systems in many critical domains, ensuring the quality of recognized transcriptions is of great importance. A recent work, CrossASR++, can automatically uncover many failures in ASR systems by taking advantage of the differential testing technique. It employs a Text-To-Speech (TTS) system to synthesize audios from texts and then reveals failed test cases by feeding them to multiple ASR systems for cross-referencing. However, no prior work tries to utilize the generated test cases to enhance the quality of ASR systems. In this paper, we explore the subsequent improvements brought by leveraging these test cases from two aspects, which we collectively refer to as a novel idea, evolutionary differential testing. On the one hand, we fine-tune a target ASR system on the corresponding test cases generated for it. On the other hand, we fine-tune a crossreferenced ASR system inside CrossASR++, with the hope to boost CrossASR++’s performance in uncovering more failed test cases. Our experiment results empirically show that the above methods to leverage the test cases can substantially improve both the target ASR system and CrossASR++ itself. After fine-tuning, the number of failed test cases uncovered decreases by 25.81% and the word error rate of the improved target ASR system drops by 45.81%. Moreover, by evolving just one cross-referenced ASR system, CrossASR++ can find 5.70%, 7.25%, 3.93%, and 1.52% more failed test cases for 4 target ASR systems, respectively.


## Prepare Environment

### Prepare Environment for CrossASR++

Please follow [the tutorial of original CrossASR++ on how they prepare the environment](https://github.com/soarsmu/CrossASRplus/tree/main/examples)

### Prepare Environment for Fine-tuning DeepSpeech

Please follow this [tutorial on how to prepare environment for fine-tuning Deepspeech](https://github.com/mhilmiasyrofi/FineTuneDeepSpeech)


## RQ1. Can test cases generated through differential testing be leveraged to improve an ASR system under test?

**Run Original CrossASR**

```
bash run-rq1-crossasr.sh
```


**Run ASREvolve**

```
bash run-rq1-asrevolve.sh
```

**Result** 

![RQ1. Improving System Under Test](https://github.com/soarsmu/ASREvolve/blob/master/images/rq1.png?raw=true)



## RQ2. Can the generated test cases be leveraged to improve the performance of CrossASR++?

**Run Original CrossASR**

```
bash run-rq2-crossasr.sh
```


**Run ASREvolve**

```
bash run-rq2-asrevolve.sh
```

**Result** 

![RQ2. Improving CrossASR++](https://github.com/soarsmu/ASREvolve/blob/master/images/rq2-summary.png?raw=true)


## Visualisation

File `experiment.ipynb` provide the way to make some visualizations of the experiment results.
