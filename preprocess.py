import json
import os
import math
import librosa

DATASET_PATH = "GTZAN/genres_original"
JSON_PATH = "data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_MFCC = 13, nfft = 2048, hop_length = 512, num_segments = 5):
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    samples_per_segment = 22050 * 30 # 30 seconds
    num_samples_per_segment = samples_per_segment // num_segments
    num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    
    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure that we are not at the root level
        if dirpath is not dataset_path:
            
            # save the semantic label
         
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            
            # process files for a specific genre
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr = 22050)
                
                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    last_sample = start_sample + num_samples_per_segment
                    
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:last_sample], sr = sr, n_mfcc = num_MFCC, n_fft = nfft, hop_length = hop_length)
                    mfcc = mfcc.T
                    
                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))
                        
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent = 4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments = 10)
    print("Data saved to {}".format(JSON_PATH))