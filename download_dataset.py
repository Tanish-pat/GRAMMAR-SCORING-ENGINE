import os
import subprocess
import requests

FULL_LINK = "https://drive.google.com/drive/folders/1nLuqJjhzmznew5WRYAjVO5GG8UhU4OM2?usp=sharing"
FOLDER_ID = "1nLuqJjhzmznew5WRYAjVO5GG8UhU4OM2"
DESTINATION = "dataset"

def is_drive_folder_accessible(link):
    print("🔎 Checking if Google Drive link is reachable and public...")
    try:
        response = requests.get(link, timeout=10)
        if response.status_code == 200:
            if "To access this item, you must be signed in" in response.text:
                print("❌ The folder exists but is not publicly accessible. Set sharing to 'Anyone with the link – Viewer'.")
                return False
            if "Google Drive – Page Not Found" in response.text:
                print("❌ The folder does not exist or the link is broken.")
                return False
            print("✅ Google Drive folder is reachable and appears to be publicly accessible.")
            return True
        else:
            print(f"❌ Received unexpected HTTP status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error reaching Google Drive: {e}")
        return False

def download_from_drive(folder_id, destination):
    if os.path.exists(destination):
        print(f"✅ Dataset folder already exists at '{destination}'. Skipping download.")
        return

    print(f"📥 Downloading dataset to '{destination}'...")
    try:
        subprocess.run([
            "gdown",
            "--folder",
            f"https://drive.google.com/drive/folders/{folder_id}",
            "-O", destination
        ], check=True)
        print("✔️ Download complete.")
    except subprocess.CalledProcessError:
        print("❌ Failed to download dataset. Ensure gdown is installed and the link is public (Viewer access).")

if __name__ == "__main__":
    if is_drive_folder_accessible(FULL_LINK):
        download_from_drive(FOLDER_ID, DESTINATION)








# import os
# import subprocess

# FULL_LINK = "https://drive.google.com/drive/folders/1nLuqJjhzmznew5WRYAjVO5GG8UhU4OM2?usp=sharing"
# FOLDER_ID = "1nLuqJjhzmznew5WRYAjVO5GG8UhU4OM2"
# DESTINATION = "dataset"

# def download_from_drive(folder_id, destination):
#     if os.path.exists(destination):
#         print(f"✅ Dataset folder already exists at '{destination}'. Skipping download.")
#         return

#     print(f"📥 Downloading dataset to '{destination}'...")
#     try:
#         subprocess.run([
#             "gdown",
#             "--folder",
#             f"https://drive.google.com/drive/folders/{folder_id}",
#             "-O", destination
#         ], check=True)
#         print("✔️ Download complete.")
#     except subprocess.CalledProcessError:
#         print("❌ Failed to download dataset. Ensure gdown is installed and the folder is set to public (Viewer access).")

# if __name__ == "__main__":
#     download_from_drive(FOLDER_ID, DESTINATION)
