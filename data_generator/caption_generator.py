from cProfile import label
import json
import numpy as np
import os
import h5py
from tqdm import tqdm
from data_generator.data_to_attribute import get_attribute_data
import pandas as pd
import random


def statistics_phrase(stats):
    mean = float(stats.get("mean", np.nan))
    std  = float(stats.get("std",  np.nan))
    vmin = float(stats.get("min",  np.nan))
    vmax = float(stats.get("max",  np.nan))
    pos_min = int(stats.get("min_pos", 0))
    pos_max = int(stats.get("max_pos", 0))

    caption = "The statistics summary of the time series: mean is {:.2f}, std is {:.2f}, min is {:.2f} at position {}, max is {:.2f} at position {}.".format(
            mean, std, vmin, pos_min, vmax, pos_max
        )

    return caption

def season_phrase(seasonal):
    stype = str(seasonal.get("type", "")).lower()
    period = seasonal.get("period", None)
    if "no" in stype or period is None:
        return random.choice(["There is no clear repeating cycle.", "No strong repeating cycle is visible."])
    p = int(period)
    if p <= 6:      bucket = "a short cycle"
    elif p <= 24:   bucket = "a daily scale cycle"
    elif p <= 60:   bucket = "a medium cycle"
    else:           bucket = "a long cycle"
    return f"The series shows {bucket} with a period of about {p} steps."

def noise_phrase(noise):
    if noise:
        return f"the readings has {noise['type']}."

    return ""

def trend_phrase(trend_list):
    # trend is a list.
    if len(trend_list) == 0:
        answer = "the readings has no trend, "
    else:
        chosen_trend_indices = np.random.choice(trend_list, size=min(2, len(trend_list)), replace=False)
        answer = ""
        for trend_info in chosen_trend_indices:
            answer += f"the readings also has a trend of {trend_info['type']} from {trend_info['start_point']} to {trend_info['end_point']}, "

    return answer

META_PATH = 'data/UTSD-full-npy/meta.csv'
def process_UTSD(input_dir, output_dir, seed: int = 123):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "utsd_captions.jsonl")
    random.seed(seed)
    np.random.seed(seed)
    
    # Start generating captions
    meta = pd.read_csv(META_PATH)
    meta_dict = {(row["Domain"], row["Dataset"]): row["Description"] for _, row in meta.iterrows()}
    
    with open(jsonl_path, "w") as f:
        for category in tqdm(os.listdir(input_dir)):
            print('Processing category:', category)
            category_path = os.path.join(input_dir, category)
            if not os.path.isdir(category_path):
                print(f"Skip non-directory: {category_path}")
                continue

            for dataset_name in os.listdir(category_path):
                for fn in os.listdir(os.path.join(category_path, dataset_name)):
                    if not fn.endswith(".npy"):
                        continue

                    description = meta_dict.get((category, dataset_name)) # dataset-level description
                    ds_path = os.path.join(category_path, dataset_name, fn)
                    data = np.load(ds_path).T  # expected shape (BS, L)
                    data = np.nan_to_num(data, nan=0.0)
    
                    BS, L = data.shape
                
                    for i in range(BS):
                        x = data[i]
                        attribute_pool = get_attribute_data(x) # dict

                        stats = attribute_pool.get("statistics", {})
                        stat_clause = statistics_phrase(stats)

                        seasonal = attribute_pool.get("seasonal", {})
                        season_clause = season_phrase(seasonal)

                        noise = attribute_pool.get("noise", {})
                        noise_clause = noise_phrase(noise)

                        trend_list = attribute_pool.get("trend", [])
                        trend_clause = trend_phrase(trend_list)

                        clauses = [stat_clause, season_clause, trend_clause, noise_clause,]
                        random.shuffle(clauses)
                        take = random.choice([2, 3, 4])
                        try:
                            caption = description + " ".join(clauses[:take])
                        except:
                            caption = "cannot describe"
                            print(fn)

                        rec = {
                            "source_path": ds_path,
                            "index": i,
                            "category": category,
                            "dataset": dataset_name,
                            "caption": caption
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print('Done!')




def get_time_of_day(dt):
    # Convert to Python datetime
    dt = np.datetime64(dt).astype('datetime64[m]').astype(object)
    hour = dt.hour

    if 0 <= hour < 4:
        return "midnight"
    elif 4 <= hour < 8:
        return "night"
    elif 8 <= hour < 12:
        return "morning"
    elif 12 <= hour < 14:
        return "noon"
    elif 14 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 22:
        return "evening"
    else:
        return "night"



def build_variable_paragraphs(all_clauses, choice_idx=None):
    paragraphs = []

    # keep resampling until overall >= 2 clauses
    total_clauses = 0
    while total_clauses < 2:
        paragraphs.clear()
        total_clauses = 0

        for var_name, clauses in all_clauses.items():
            if not clauses:
                continue

            if choice_idx is not None:
                n = random.choice(choice_idx)  # randomly decide how many to include
                n = min(n, len(clauses))       # cap by available clauses
                selected_clauses = random.sample(clauses, n)
            else:
                selected_clauses = clauses

            # clean and normalize
            selected_clauses = [
                c.strip().capitalize().rstrip('.') for c in selected_clauses if c
            ]

            total_clauses += len(selected_clauses)

            if selected_clauses:
                joined = ". ".join(selected_clauses) + "."
                var_sentence = f"For {var_name.replace('_', ' ')}, {joined}"
                paragraphs.append(var_sentence)

    return " ".join(paragraphs)



def process_capture(input_dir, output_dir, seed: int = 123):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "capture.jsonl")
    random.seed(seed)
    np.random.seed(seed)    
    
    with open(jsonl_path, "w") as f:
        description = "60-second tri-axial accelerometer recordings sampled at 30 Hz from wrist-worn sensors, providing quantitative representations of human motion and activity intensity."

        data_path = os.path.join(input_dir, "X.npy")
        data = np.load(data_path, mmap_mode='r') # BS, L, nvar
        data = np.nan_to_num(data, nan=0.0)
        time = np.load(os.path.join(input_dir, "time.npy"), mmap_mode='r')  # BS,
        label = np.load(os.path.join(input_dir, "Y.npy"), mmap_mode='r')  # BS,

        BS, L, nvar = data.shape

        for i in range(BS):
            mts = data[i].T # shape (nvar, L)
            variable_name = ["X-axis acceleration", "Y-axis acceleration", "Z-axis acceleration"]

            activity = str(label[i]) # make sure is string
            time_of_day = get_time_of_day(time[i])
            action_clause = f"Combined with above information, the subject is {activity} during {time_of_day}. "

            all_clauses = {}
            for var_idx in range(nvar):
                x = mts[var_idx]
                var_name = variable_name[var_idx]

                attribute_pool = get_attribute_data(x) # dict

                stats = attribute_pool.get("statistics", {})
                stat_clause = statistics_phrase(stats)

                seasonal = attribute_pool.get("seasonal", {})
                season_clause = season_phrase(seasonal)

                noise = attribute_pool.get("noise", {})
                noise_clause = noise_phrase(noise)

                trend_list = attribute_pool.get("trend", [])
                trend_clause = trend_phrase(trend_list)

                all_clauses[var_name] = [stat_clause, season_clause, trend_clause, noise_clause,]

            # aggregate a big pool from all variables  
            paragraphs = build_variable_paragraphs(all_clauses, choice_idx=[0,1,2])
            caption = description + " " + paragraphs + " " + action_clause
            
            rec = {
                "source_path": data_path,
                "index": i,
                "category": "IoT",
                "dataset": "Capture24",
                "caption": caption
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print('Done!')

def process_auditory(input_dir, output_dir, seed: int = 123):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "auditory.jsonl")
    random.seed(seed)
    np.random.seed(seed)    
    
    with open(jsonl_path, "w") as f:
        description = "Six-second three-channel EEG recordings sampled at 65 Hz, capturing neural signal fluctuations for modeling temporal patterns and cognitive states."

        data_path = os.path.join(input_dir, "X.npy")
        data = np.load(data_path) # BS, nvar, L
        data = np.nan_to_num(data, nan=0.0)
        label = np.load(os.path.join(input_dir, "activity.npy")) # BS,
        BS, nvar, L = data.shape

        for i in range(BS):
            mts = data[i] # shape (nvar, L)
            variable_name = ["Left Hemisphere Electroencephalography", "Parietal Region Electroencephalography", "Occipital Electroencephalography"]

            activity = str(label[i]) # make sure is string
            action_clause = f"Combined with above information, the subject is {activity}."

            all_clauses = {}
            for var_idx in range(nvar):
                x = mts[var_idx]
                var_name = variable_name[var_idx]

                attribute_pool = get_attribute_data(x) # dict

                stats = attribute_pool.get("statistics", {})
                stat_clause = statistics_phrase(stats)

                seasonal = attribute_pool.get("seasonal", {})
                season_clause = season_phrase(seasonal)

                noise = attribute_pool.get("noise", {})
                noise_clause = noise_phrase(noise)

                trend_list = attribute_pool.get("trend", [])
                trend_clause = trend_phrase(trend_list)

                all_clauses[var_name] = [stat_clause, season_clause, trend_clause, noise_clause,]

            # aggregate a big pool from all variables  
            paragraphs = build_variable_paragraphs(all_clauses, choice_idx=[0,1,2])
            caption = description + " " + paragraphs + " " + action_clause
            
            rec = {
                "source_path": data_path,
                "index": i,
                "category": "IoT",
                "dataset": "auditory",
                "caption": caption
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print('Done!')


def process_maus(input_dir, output_dir, seed: int = 123):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "maus.jsonl")
    random.seed(seed)
    np.random.seed(seed)    
    
    def describe_state(nasa_tlx, psqi):
        """
        Return one descriptive sentence about cognitive workload and sleep quality.
        """

        # Discretize NASA-TLX (mental workload)
        if nasa_tlx <= 33:
            wl = "low mental workload"
        elif nasa_tlx <= 66:
            wl = "moderate mental workload"
        else:
            wl = "high mental workload"

        # Discretize PSQI (sleep quality)
        if psqi <= 5:
            sleep = "good sleep quality"
        elif psqi <= 10:
            sleep = "moderate sleep quality"
        else:
            sleep = "poor sleep quality"

        # Compose one sentence
        return f"From above information, one can see the subject exhibits {wl} and {sleep}, indicating {('a well-rested' if psqi <= 5 else 'a fatigued')} condition during task performance."

    with open(jsonl_path, "w") as f:
        description = "Six-second multimodal physiological recordings at 65 Hz, including Photoplethysmography, Electrocardiogram, and Galvanic Skin Response, collected during N-back tasks to capture mental workload variations and model short-term cognitive dynamics."

        data_path = os.path.join(input_dir, "X.npy")
        data = np.load(data_path) # BS, nvar, L
        data = np.nan_to_num(data, nan=0.0)
        nasa_tlx = np.load('data/normwear_data/maus/NASA_TLX.npy') # BS,
        psqi = np.load('data/normwear_data/maus/PSQI.npy') # BS, 

        # discretize PSQI and NASA_TLX into categories with scientfic basis

        BS, nvar, L = data.shape
        for i in range(BS):
            mts = data[i] # shape (nvar, L)
            variable_name = ["Photoplethysmography", "Electrocardiogram", "Galvanic Skin Response"]

            action_clause = describe_state(nasa_tlx[i], psqi[i])

            all_clauses = {}
            for var_idx in range(nvar):
                x = mts[var_idx]
                var_name = variable_name[var_idx]

                attribute_pool = get_attribute_data(x) # dict

                stats = attribute_pool.get("statistics", {})
                stat_clause = statistics_phrase(stats)

                seasonal = attribute_pool.get("seasonal", {})
                season_clause = season_phrase(seasonal)

                noise = attribute_pool.get("noise", {})
                noise_clause = noise_phrase(noise)

                trend_list = attribute_pool.get("trend", [])
                trend_clause = trend_phrase(trend_list)

                all_clauses[var_name] = [stat_clause, season_clause, trend_clause, noise_clause,]

            # aggregate a big pool from all variables  
            paragraphs = build_variable_paragraphs(all_clauses, choice_idx=[0,1,2])
            caption = description + " " + paragraphs + " " + action_clause
            
            rec = {
                "source_path": data_path,
                "index": i,
                "category": "IoT",
                "dataset": "maus",
                "caption": caption
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print('Done!')
        


def process_mendeley(input_dir, output_dir, seed: int = 123):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "mendeley.jsonl")
    random.seed(seed)
    np.random.seed(seed)    
    

    def describe_emotion(emotion, valence, arousal):
        """
        Discretize valence/arousal into high/low based on scientific thresholds, and describe the resulting emotion.
        """

        # According to Russell's circumplex model of affect:
        # Valence threshold around 0 (neutral), Arousal threshold around 0.5 (moderate activation)
        valence_cat = "high" if valence >= 0 else "low"
        arousal_cat = "high" if arousal >= 0.5 else "low"

        descriptions = f"The subject exhibits {valence_cat}-valence and {arousal_cat}-arousal, "
        descriptions += f"indicating an emotional state consistent with {emotion}."

        return descriptions


    with open(jsonl_path, "w") as f:
        description = "Six-second multimodal physiological recordings sampled at 65 Hz, including Electrocardiogram and Galvanic Skin Response, collected while participants viewed emotional stimuli to capture physiological variations associated with affective states and support modeling of emotion and activity dynamics."

        data_path = os.path.join(input_dir, "X.npy")
        data = np.load(data_path) # BS, nvar, L
        data = np.nan_to_num(data, nan=0.0)
        emotion = np.load(os.path.join(input_dir, "Emotion.npy")) # BS, string
        valence = np.load(os.path.join(input_dir, "Valence.npy")) # BS,
        arousal = np.load(os.path.join(input_dir, "Arousal.npy")) # BS,

        # help me form a paragraph, discretize valence and arousal to high/low using scientific thresholds, then say it leads to emotion X.

        BS, nvar, L = data.shape

        for i in range(BS):
            mts = data[i] # shape (nvar, L)
            variable_name = ["Galvanic Skin Response","Electrocardiogram"]

            action_clause = describe_emotion(emotion[i], valence[i], arousal[i])

            all_clauses = {}
            for var_idx in range(nvar):
                x = mts[var_idx]
                var_name = variable_name[var_idx]

                attribute_pool = get_attribute_data(x) # dict

                stats = attribute_pool.get("statistics", {})
                stat_clause = statistics_phrase(stats)

                seasonal = attribute_pool.get("seasonal", {})
                season_clause = season_phrase(seasonal)

                noise = attribute_pool.get("noise", {})
                noise_clause = noise_phrase(noise)

                trend_list = attribute_pool.get("trend", [])
                trend_clause = trend_phrase(trend_list)

                all_clauses[var_name] = [stat_clause, season_clause, trend_clause, noise_clause,]

            # aggregate a big pool from all variables  
            paragraphs = build_variable_paragraphs(all_clauses, choice_idx=[0,1,2])
            caption = description + " " + paragraphs + " " + action_clause
            
            rec = {
                "source_path": data_path,
                "index": i,
                "category": "IoT",
                "dataset": "mendeley",
                "caption": caption
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print('Done!')


def process_phyatt(input_dir, output_dir, seed: int = 123):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "Phyatt.jsonl")
    random.seed(seed)
    np.random.seed(seed)    
    
    with open(jsonl_path, "w") as f:
        description = "Six-second multimodal recordings at 65 Hz, including four-position EEG, Galvanic Skin Response, and Photoplethysmography, collected during natural speech listening to capture auditory attention fluctuations and model cognitive-attention dynamics."

        data_path = os.path.join(input_dir, "X.npy")
        data = np.load(data_path, mmap_mode='r') # BS, nvar, L
        data = np.nan_to_num(data, nan=0.0)
        label = np.load(os.path.join(input_dir, "activity.npy"), mmap_mode='r')  # BS,

        BS, nvar, L = data.shape

        for i in range(BS):
            mts = data[i] # shape (nvar, L)
            variable_name = ["Electroencephalography (Frontal)",
                             "Electroencephalography (Left)",
                             "Electroencephalography (Occipital)",
                             "Electroencephalography (Right)",
                             "Galvanic Skin Response",
                             "Photoplethysmography"]

            activity = str(label[i]) # make sure is string
            action_clause = f"Combined with above information, the subject is {activity}. "

            all_clauses = {}
            for var_idx in range(nvar):
                x = mts[var_idx]
                var_name = variable_name[var_idx]

                attribute_pool = get_attribute_data(x) # dict

                stats = attribute_pool.get("statistics", {})
                stat_clause = statistics_phrase(stats)

                seasonal = attribute_pool.get("seasonal", {})
                season_clause = season_phrase(seasonal)

                noise = attribute_pool.get("noise", {})
                noise_clause = noise_phrase(noise)

                trend_list = attribute_pool.get("trend", [])
                trend_clause = trend_phrase(trend_list)

                all_clauses[var_name] = [stat_clause, season_clause, trend_clause, noise_clause,]

            # aggregate a big pool from all variables  
            paragraphs = build_variable_paragraphs(all_clauses, choice_idx=[0,1,2])
            caption = description + " " + paragraphs + " " + action_clause
            
            rec = {
                "source_path": data_path,
                "index": i,
                "category": "IoT",
                "dataset": "Phyatt",
                "caption": caption
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print('Done!')


def process_dalia(input_dir, output_dir, seed: int = 123):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "dalia.jsonl")
    random.seed(seed)
    np.random.seed(seed)    
    
    with open(jsonl_path, "w") as f:
        description = "Six-second multimodal recordings at 65 Hz from wrist-worn sensors, including tri-axial accelerometer, Photoplethysmography, Galvanic Skin Response, and Electrocardiogram, collected during daily activities to model human activity patterns."

        data_path = os.path.join(input_dir, "X.npy")
        data = np.load(data_path, mmap_mode='r') # BS, nvar, L
        data = np.nan_to_num(data, nan=0.0)
        label = np.load(os.path.join(input_dir, "activity.npy"), mmap_mode='r')  # BS,

        BS, nvar, L = data.shape

        for i in range(BS):
            mts = data[i] # shape (nvar, L)
            variable_name = ["Accelerometer X-axis",
                             "Accelerometer Y-axis",
                             "Accelerometer Z-axis",
                             "Photoplethysmography",
                             "Galvanic Skin Response",
                             "Electrocardiogram"]

            activity = str(label[i]) # make sure is string
            action_clause = f"Combined with above information, the subject is {activity}. "

            all_clauses = {}
            for var_idx in range(nvar):
                x = mts[var_idx]
                var_name = variable_name[var_idx]

                attribute_pool = get_attribute_data(x) # dict

                stats = attribute_pool.get("statistics", {})
                stat_clause = statistics_phrase(stats)

                seasonal = attribute_pool.get("seasonal", {})
                season_clause = season_phrase(seasonal)

                noise = attribute_pool.get("noise", {})
                noise_clause = noise_phrase(noise)

                trend_list = attribute_pool.get("trend", [])
                trend_clause = trend_phrase(trend_list)

                all_clauses[var_name] = [stat_clause, season_clause, trend_clause, noise_clause,]

            # aggregate a big pool from all variables  
            paragraphs = build_variable_paragraphs(all_clauses, choice_idx=[0,1,2])
            caption = description + " " + paragraphs + " " + action_clause
            
            rec = {
                "source_path": data_path,
                "index": i,
                "category": "IoT",
                "dataset": "dalia",
                "caption": caption
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print('Done!')


def process_ppg_ecg(input_dir, output_dir, seed: int = 123):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "ppg_ecg.jsonl")
    random.seed(seed)
    np.random.seed(seed)    

    # need a function to discretize blood pressure levels
    def discretize_bp(systolic, diastolic):
        """
        Discretize blood pressure levels into categories based on systolic and diastolic values.
        Categories:
        - Normal
        - Elevated
        - Hypertension Stage 1
        - Hypertension Stage 2
        - Hypertensive Crisis
        """
        if systolic < 120 and diastolic < 80:
            return "normal blood pressure"
        elif 120 <= systolic < 130 and diastolic < 80:
            return "elevated blood pressure"
        elif (130 <= systolic < 140) or (80 <= diastolic < 90):
            return "hypertension stage 1"
        elif (140 <= systolic < 180) or (90 <= diastolic < 120):
            return "hypertension stage 2"
        else:
            return "hypertensive crisis"
    
    with open(jsonl_path, "w") as f:
        description = "Six-second cardiovascular recordings at 65 Hz, including Photoplethysmography and Electrocardiogram, collected under rest and exercise to support cuff-less blood pressure estimation and arterial dynamics modeling."

        data_path = os.path.join(input_dir, "X.npy")
        data = np.load(data_path, mmap_mode='r') # BS, nvar, L
        data = np.nan_to_num(data, nan=0.0)
        max_bp = np.load(os.path.join(input_dir, "max_bp.npy"), mmap_mode='r')  # BS,
        min_bp = np.load(os.path.join(input_dir, "min_bp.npy"), mmap_mode='r')  # BS,
        
        BS, nvar, L = data.shape

        for i in range(BS):
            action_clause = f"The subject has {discretize_bp(max_bp[i], min_bp[i])}."
            mts = data[i] # shape (nvar, L)
            variable_name = ["Photoplethysmography",
                             "Electrocardiogram"]

            all_clauses = {}
            for var_idx in range(nvar):
                x = mts[var_idx]
                var_name = variable_name[var_idx]

                attribute_pool = get_attribute_data(x) # dict

                stats = attribute_pool.get("statistics", {})
                stat_clause = statistics_phrase(stats)

                seasonal = attribute_pool.get("seasonal", {})
                season_clause = season_phrase(seasonal)

                noise = attribute_pool.get("noise", {})
                noise_clause = noise_phrase(noise)

                trend_list = attribute_pool.get("trend", [])
                trend_clause = trend_phrase(trend_list)

                all_clauses[var_name] = [stat_clause, season_clause, trend_clause, noise_clause,]

            # aggregate a big pool from all variables  
            paragraphs = build_variable_paragraphs(all_clauses, choice_idx=[0,1,2])
            caption = description + " " + paragraphs + " " + action_clause
            
            rec = {
                "source_path": data_path,
                "index": i,
                "category": "IoT",
                "dataset": "ppg_ecg",
                "caption": caption
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print('Done!')


import concurrent.futures

def run_all():
    tasks = [
        ('data/UTSD-full-npy', process_UTSD),
        ('data/normwear_data/auditory', process_auditory),
        ('data/normwear_data/dalia', process_dalia),
        ('data/normwear_data/maus', process_maus),
        ('data/normwear_data/mendeley', process_mendeley),
        ('data/normwear_data/Phyatt', process_phyatt),
        ('data/normwear_data/ppg_ecg', process_ppg_ecg),
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for input_dir, func in tasks:
            output_dir = input_dir  # same as input
            futures.append(executor.submit(func, input_dir, output_dir))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"⚠️ Error in a task: {e}")

if __name__ == "__main__":
    run_all()
    

# SINGLE GPU RUN:
# if __name__ == "__main__":
#     # input_dir = 'data/UTSD-full-npy'
#     # output_dir = 'data/UTSD'

#     # process_UTSD(input_dir, output_dir)

#     input_dir = 'data/normwear_data/auditory'
#     output_dir = 'data/normwear_data/auditory'
#     process_auditory(input_dir, output_dir)

#     input_dir = 'data/normwear_data/dalia'
#     output_dir = 'data/normwear_data/dalia'
#     process_dalia(input_dir, output_dir)

#     input_dir = 'data/normwear_data/maus'
#     output_dir = 'data/normwear_data/maus'
#     process_maus(input_dir, output_dir)

#     input_dir = 'data/normwear_data/mendeley'
#     output_dir = 'data/normwear_data/mendeley'
#     process_mendeley(input_dir, output_dir)

#     input_dir = 'data/normwear_data/Phyatt'
#     output_dir = 'data/normwear_data/Phyatt'
#     process_phyatt(input_dir, output_dir)

#     input_dir = 'data/normwear_data/ppg_ecg'
#     output_dir = 'data/normwear_data/ppg_ecg'
#     process_ppg_ecg(input_dir, output_dir)

#     # python -m generate_UTSD_caption