import pandas as pd
val_df=pd.read_csv("/home/student/m/mnsiah/modality_fusion/preprocessing_pipeline/new_val_1.csv")
print(len(val_df))
# filter and remove undownloade audio
import os
#function to remove audio that could not be downloaded
def remove_audio(sample_df,datatype=None):
  # Get list of successfully downloaded audio files
  if datatype == 'test':
    downloaded_files = os.listdir("/share/users/student/m/mnsiah/processed_files/audio_test")
    successfully_downloaded_vids = [f.split('.')[0] for f in downloaded_files if f.endswith('.wav')]

    filtered_test_df = sample_df[sample_df['video_id'].isin(successfully_downloaded_vids)].reset_index(drop=True)
    print(f"Original sample_df length: {len(sample_df)}")
    print(f"Filtered sample_df length: {len(filtered_test_df)}")

    return filtered_test_df
  elif datatype == 'train':
    downloaded_files = os.listdir("/share/users/student/m/mnsiah/processed_files/audio_train")
    successfully_downloaded_vids = [f.split('.')[0] for f in downloaded_files if f.endswith('.wav')]
    filtered_train_df = sample_df[sample_df['video_id'].isin(successfully_downloaded_vids)].reset_index(drop=True)
    print(f"Original sample_df length: {len(sample_df)}")
    print(f"Filtered sample_df length: {len(filtered_train_df)}")
    return filtered_train_df
  else:
    downloaded_files = os.listdir("/share/users/student/m/mnsiah/processed_files/audio_val")

    successfully_downloaded_vids = [f.split('.')[0] for f in downloaded_files if f.endswith('.wav')]

    filtered_val_df = sample_df[sample_df['video_id'].isin(successfully_downloaded_vids)].reset_index(drop=True)
  # Extract video IDs from filenames (assuming format is video_id.wav)
    print(f"Original sample_df length: {len(sample_df)}")
    print(f"Filtered sample_df length: {len(filtered_val_df)}")
    return filtered_val_df



#example usage to remmove undownloaded audio from df
final_df=remove_audio(val_df,'val')
#save
final_df.to_csv('final_val.csv',index=False)
print(final_df['target_label'].value_counts())
