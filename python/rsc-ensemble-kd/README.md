# Improving Respiratory Sound Classification with Architecture-Agnostic Knowledge Distillation from Ensembles
Code for INTERSPEECH 2025 paper titled "Improving Respiratory Sound Classification with Architecture-Agnostic Knowledge Distillation from Ensembles"

The code is based on BTS: Bridging Text and Sound Modalities for Metadata-Aided Respiratory Sound Classification. You can also refer to its code [here](https://github.com/RSC-Toolkit/BTS).

## TODO

- [ ] Add teacher logit generation code and example. (Currently we already provide the teacher logits that can be used for replicating our experiments)

## Prerequisites
Please check environments and requirements before you start. If required, we recommend you to either upgrade versions or install them for smooth running.

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

### Environments
`Ubuntu xx.xx`  
`Python 3.8.xx`

## Environmental set-up

Install the necessary packages with:

run `requirements.txt`

```
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

For the reproducibility, we used torch=2.0.1+cu117 and torchaudio=2.0.1+cu117, so we highly recommend install as follow:

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

## Datasets
Download the ICBHI files and unzip it.
All details is described in the [paper w/ code](https://paperswithcode.com/dataset/icbhi-respiratory-sound-database)

```
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
or 
wget --no-check-certificate https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
```

All `*.wav` and `*.txt` should be saved in data/icbhi_dataset/audio_test_data. (i.e., mkdir `audio_test_data` into `data/icbhi_dataset/` and move `*.wav` and `*.txt` into `data/icbhi_dataset/audio_test_data/`)

Note that ICBHI dataset consists of a total of 6,898 respiratory cycles, of which 1,864 contain crackles, 886 contain wheezes, and 506 contain both crackles and wheezes, in 920 annotated audio samples from 126 subjects.

## Run

### Distillation

You can train BTS baselines, which are used for making a teacher model using the following command. This example script trains 5 models with seed=1,2,3,4,5. However teacher logits from 30 models are already provided in `/teacher_logits/` directory.

```
$ ./scripts/train_BTS.sh 
```

Distill BTS-d student from 5 teachers, using the provided teacher logits.
```
$ ./scripts/train_BTS-d[5].sh
```

You can modify the `soft_label_mode` to test different configurations using the following rules:
- [`none`/`random`/`mean`] to select hard label, random teacher or mean teacher, respectively
- `_k` to select the number of teachers
- `_cherry` to use the cherrypicked/curated teachers

So for example:
- `--soft_label_mode random_15`     -> best random teacher (k=15)
- `--soft_label_mode mean_5`        -> best mean teacher (k=5)
- `--soft_label_mode mean_cherry`   -> curated teacher ensemble from selected top 5 teachers

#### Custom Teachers

If you want to use your own teachers. Create a .pt files with (n_samples, n_teachers, n_classes) shape and pass their paths to `--train_teacher_logits` and `--eval_teacher_logits` arguments in training script. Test logits are only used for checking loss. So for example, create `train.pt` and `test.pt` in root and append to the training command:
`--train_teacher_logits train.pt --eval_teacher_logits test.pt`

### Ensembles

We use notebooks to evaluate the ensembled logits. See `notebooks/check_ensemble_score.ipynb` for an example how to check the ensemble score. You must first run the training code at least once to preprocess the dataset.

#### Custom Ensemble

To test your own trained BTS-d++ model results, create a .pt files with (n_samples, n_teachers, n_classes) and update the paths in the notebook.

## ICBHI Data

The database consists of a total of 5.5 hours of recordings containing 6898 respiratory cycles, of which 1864 contain crackles, 886 contain wheezes, and 506 contain both crackles and wheezes, in 920 annotated audio samples from 126 subjects.

The downloaded data looks like [[kaggle](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database), [paper w/ code](https://paperswithcode.com/dataset/icbhi-respiratory-sound-database)]:

<pre>
data/icbhi_dataset
├── metadata.txt
│    ├── Patient number
│    ├── Age
│    ├── Sex
│    ├── Adult BMI (kg/m2)
│    ├── Adult Weight (kg)
│    └── Child Height (cm)
│
├── official_split.txt
│    ├── Patient number_Recording index_Chest location_Acqiosotopm mode_Recording equipment
│    |    ├── Chest location
│    |    |    ├── Trachea (Tc),Anterior left (Al),Anterior right (Ar),Posterior left (Pl)
│    |    |    └── Posterior right (Pr),Lateral left (Ll),Lateral right (Lr)
│    |    |
│    |    ├── Acquisition mode
│    |    |    └── sequential/single channel (sc), simultaneous/multichannel (mc)
│    |    |
│    |    └── Recording equipment 
│    |         ├── AKG C417L Microphone (AKGC417L), 
│    |         ├── 3M Littmann Classic II SE Stethoscope (LittC2SE), 
│    |         ├── 3M Litmmann 3200 Electronic Stethoscope (Litt3200), 
│    |         └── WelchAllyn Meditron Master Elite Electronic Stethoscope (Meditron)
│    |    
│    └── Train/Test   
│
├── patient_diagnosis.txt
│    ├── Patient number
│    └── Diagnosis
│         ├── COPD: Chronic Obstructive Pulmonary Disease
│         ├── LRTI: Lower Respiratory Tract Infection
│         └── URTI: Upper Respiratory Tract Infection
│
└── patient_list_foldwise.txt
</pre>
