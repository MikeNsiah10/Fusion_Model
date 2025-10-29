import yt_dlp
import os


def download_audio(sample_df,datatype=None):
    # Get unique video IDs
    video_ids = sample_df['video_id'].unique()

    # Create a folder to hold downloaded audio
    if datatype=='test':
        audio_dir = "/share/users/student/m/mnsiah/processed_files/audio_test"
        os.makedirs(audio_dir, exist_ok=True)
        current_audio_dir = audio_dir
    elif datatype == 'train':
        audio_dir = "/share/users/student/m/mnsiah/processed_files/audio_train"
        os.makedirs(audio_dir, exist_ok=True)
        current_audio_dir = audio_dir
    else: # Assuming sample_df is sample_val_df
        audio_dir_val = "/share/users/student/m/mnsiah/processed_files/audio_val"
        os.makedirs(audio_dir_val, exist_ok=True)
        current_audio_dir = audio_dir_val

    # List to store video IDs that failed to download
    failed_downloads = []

    # Download audio for each video using yt-dlp
    for vid in video_ids:
        video_url = f'https://www.youtube.com/watch?v={vid}'
        output_template = os.path.join(current_audio_dir, '%(id)s.%(ext)s')

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': output_template,
            'quiet': True, # Suppress console output for cleaner loop
            'nocheckcertificate': True, # Add this option to skip certificate checks
            'no_warnings': True, # Add this option to suppress warnings
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            print(f"✅ Downloaded audio for video ID: {vid}")
        except Exception as e:
            print(f"❌ Error downloading audio for video ID {vid}: {e}")
            failed_downloads.append(vid)

    # This print statement is now inside the function and outside the loop
    print("\n✅ Attempted to download audio for all unique video IDs.")


#Example usage 
import pandas as pd
train_df=pd.read_csv("/home/student/m/mnsiah/modality_fusion/preprocessing_pipeline/new_val.csv")


#download_audio(train_df,'train')
download_audio(train_df,'val')