import pandas as pd
import numpy as np
import json

def calculate_subjective_features(gaze_file, yolo_file, output_file):
    # Load Gaze Data
    gaze_df = pd.read_csv(gaze_file, delimiter=';')
    yolo_df = pd.read_csv(yolo_file, delimiter=',')

    # Merge gaze data with YOLO data on frame number
    merged_df = pd.merge(gaze_df, yolo_df, left_on='VideoFrame', right_on='frame', how='left')

    # Filter only relevant object classes (car, person, bicycle) #important?
    merged_df = merged_df[merged_df['class'].isin(['car', 'person', 'bicycle'])]

    # Convert PixelX, PixelY to numeric values
    merged_df['PixelX'] = pd.to_numeric(merged_df['PixelX'], errors='coerce')
    merged_df['PixelY'] = pd.to_numeric(merged_df['PixelY'], errors='coerce')

    # Compute frequency of gaze shifts
    merged_df['prev_x'] = merged_df['PixelX'].shift(1)
    merged_df['prev_y'] = merged_df['PixelY'].shift(1)
    merged_df['gaze_shift'] = np.sqrt((merged_df['PixelX'] - merged_df['prev_x']) ** 2 +
                                      (merged_df['PixelY'] - merged_df['prev_y']) ** 2)
    gaze_shift_frequency = merged_df['gaze_shift'].count()

    # Compute average gaze shift distance
    avg_gaze_shift_distance = merged_df['gaze_shift'].mean()

    # Find the most viewed object class
    most_viewed_class = merged_df['class'].mode()[0] if not merged_df['class'].isna().all() else 'None'

    # Calculate of the frequency of views of different classes
    class_frequency = merged_df['class'].value_counts().to_dict()

    # Most frequent class
    most_frequent_class = merged_df['class'].mode()[0] if not merged_df['class'].isna().all() else 'None'
    most_frequent_class_count = merged_df['class'].value_counts().max() if not merged_df['class'].isna().all() else 0

    # Calculate viewing duration per class
    merged_df['same_class'] = merged_df['class'] == merged_df['class'].shift(1)
    merged_df['view_duration'] = merged_df['same_class'].cumsum()  # Laufende Summe für gleiche Klasse

    # Compute total viewing time for each class
    class_durations = merged_df.groupby('class')['view_duration'].max()

    # Determine the class with the longest viewing duration
    longest_viewed_class = class_durations.idxmax() if not class_durations.empty else 'None'

    # Save results
    results = {
        'gaze_shift_frequency': gaze_shift_frequency,
        'avg_gaze_shift_distance': avg_gaze_shift_distance,
        'most_viewed_class': most_viewed_class,
        'most_frequent_class': most_frequent_class,
        'longest_viewed_class': longest_viewed_class,
        'most_frequent_class_count': most_frequent_class_count
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(output_file, index=False)
    class_frequency_df = pd.DataFrame(list(class_frequency.items()), columns=['class', 'frequency'])
    class_frequency_df.to_csv('/Users/ahmadmohamad/Desktop/hci/class_frequency_output.csv', index=False)

 #   return results

#test_run
calculate_subjective_features('/Users/ahmadmohamad/Desktop/hci/crit_car_01_gaze_tracking_Varjo.csv', '/Users/ahmadmohamad/Desktop/hci/crit_car_01_Object_detection_YOLO.csv', '/Users/ahmadmohamad/Desktop/hci/output_subjective_features.csv')
