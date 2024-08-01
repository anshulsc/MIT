

import os
import pandas as pd
from datetime import datetime
import librosa
from utils.utils import (
    list_subdirectories,
    list_files,
    load_audio,
    preprocess_audio,
    get_formant_dispersal,
    get_wiener_entropy_mean,
    extract_features
)


def process_file(file_info):
    """
    Process a single audio file and extract features.

    Parameters:
        file_info (tuple): A tuple containing file information (id_, day_folder, file, label).

    Returns:
        dict: A dictionary of extracted features and additional metadata.
    """
    id_, day_folder, file, label = file_info
    y, sr = librosa.load(file)
    y_preprocessed = preprocess_audio(y, sr)
    day = datetime.strptime(day_folder, '%B %d').date().replace(year=2000)
    return {
        'CowID': id_,
        'Day': day,
        'Label': label,
        **extract_features(y_preprocessed, sr),
        'Formant Dispersal' : get_formant_dispersal(file),
        'Wiener Entropy Mean': get_wiener_entropy_mean(y_preprocessed),
        'File': file
    }


def extract_features_from_audio_path(base_path):
    """
    Extract features from audio files in the given base path.

    Parameters:
        base_path (str): The base path to the directory containing audio files.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted features.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Directory not found: {base_path}")

    ids = list_subdirectories(base_path)
    all_features = []

    for id_ in ids:
        days_path = os.path.join(base_path, id_)
        calving_date_path = os.path.join(days_path, 'c_date.txt')
        if not os.path.exists(calving_date_path):
            raise FileNotFoundError(f"Calving date file not found: {calving_date_path}")

        with open(calving_date_path, 'r') as f:
            calving_date_str = f.read().strip()
        calving_date = datetime.strptime(calving_date_str, '%B %d').replace(year=2000)

        days = list_subdirectories(days_path)
        for day_folder in days:
            current_date = datetime.strptime(day_folder, '%B %d').replace(year=2000)
            if current_date < calving_date:
                label = 'Pre-calving'
            elif current_date > calving_date:
                label = 'Post-calving'
            else:
                label = 'Calving'

            vocals_path = os.path.join(days_path, day_folder, 'Vocals')
            vocal_files = list_files(vocals_path)

            for vf in vocal_files:
                file_path = os.path.join(vocals_path, vf)
                file_info = (id_, day_folder, file_path, label)
                try:
                    features = process_file(file_info)
                    all_features.append(features)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return pd.DataFrame(all_features)

# Example usage:
if __name__ == "__main__":
    base_path = "/path/to/Cows"  # Replace with the actual path
    output_path = os.path.join(base_path, "audio_features.csv")

    features_df = extract_features_from_audio_path(base_path)
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
