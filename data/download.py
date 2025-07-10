import os
import sys
from googleapiclient.discovery import build
from google.oauth2 import service_account
import io
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

FOLDER_ID = '1GOX2sLHVNgaFZ8QtBuLUYioz33kzbjl2'

def authenticate_gdrive_service_account():
    """Authenticate using service account and return Google Drive service object."""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            'service-account-key.json', scopes=SCOPES)
        print("Service account authentication successful!")
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        print(f"Service account authentication failed: {e}")
        print("Make sure 'service-account-key.json' exists and is valid.")
        sys.exit(1)

def get_files_in_folder(service, folder_id):
    """Get all files in the specified folder."""
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            pageSize=1000,
            fields="nextPageToken, files(id, name)"
        ).execute()
        
        items = results.get('files', [])
        return items
    except Exception as e:
        print(f"Error fetching files: {e}")
        print("Make sure the service account has access to the folder.")
        print("The folder owner needs to share it with the service account email.")
        return []

def download_file(service, file_id, file_name, download_path):
    """Download a file from Google Drive."""
    try:
        request = service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download progress for {file_name}: {int(status.progress() * 100)}%")
        
        file_path = os.path.join(download_path, file_name)
        with open(file_path, 'wb') as f:
            f.write(file_io.getvalue())
        
        print(f"✓ Downloaded: {file_name}")
        return True
    
    except Exception as e:
        print(f"✗ Error downloading {file_name}: {str(e)}")
        return False

def download_cfb_files(start_index, end_index, download_path="./source_info"):
    """
    Download data_x_{i} and data_z_{i} files for indices from start_index to end_index.
    
    Args:
        start_index (int): Starting index (inclusive)
        end_index (int): Ending index (inclusive)
        download_path (str): Local path to save downloaded files
    """
    
    os.makedirs(download_path, exist_ok=True)
    
    print("Authenticating with Google Drive using service account...")
    service = authenticate_gdrive_service_account()
    
    print("Fetching file list from Google Drive folder...")
    all_files = get_files_in_folder(service, FOLDER_ID)
    
    if not all_files:
        print("No files found in the folder.")
        return
    
    file_mapping = {file['name']: file['id'] for file in all_files}
    
    print(f"Found {len(all_files)} files in the folder.")
    print(f"Downloading data_x and data_z files for indices {start_index} to {end_index}...")
    
    success_count = 0
    total_files = 0
    
    for i in range(start_index, end_index + 1):
        data_x_filename = f"pm_{i}.npy"
        data_z_filename = f"pr_{i}.npy"
        
        if data_x_filename in file_mapping:
            total_files += 1
            if download_file(service, file_mapping[data_x_filename], data_x_filename, download_path):
                success_count += 1
        else:
            print(f"⚠ File not found: {data_x_filename}")
        
        if data_z_filename in file_mapping:
            total_files += 1
            if download_file(service, file_mapping[data_z_filename], data_z_filename, download_path):
                success_count += 1
        else:
            print(f"⚠ File not found: {data_z_filename}")
    
    print(f"\nDownload completed!")
    print(f"Successfully downloaded: {success_count}/{total_files} files")
    print(f"Files saved to: {os.path.abspath(download_path)}")

def main():
    """Main function with command line interface."""
    if len(sys.argv) != 3:
        print("Usage: python script.py <start_index> <end_index>")
        print("Example: python script.py 1 10")
        sys.exit(1)
    
    try:
        start_index = int(sys.argv[1])
        end_index = int(sys.argv[2])
        
        if start_index < 1 or end_index < 1 or start_index > end_index:
            print("Error: Invalid index range. Both indices must be >= 1 and start_index <= end_index")
            sys.exit(1)
        
        if end_index > 108:
            print("Warning: end_index > 108. Based on your description, files might not exist beyond index 108.")
        
        download_cfb_files(start_index, end_index)
        
    except ValueError:
        print("Error: Please provide valid integer indices.")
        sys.exit(1)

if __name__ == "__main__":
    main()