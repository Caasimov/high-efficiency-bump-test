import os

def get_data(url: str, output: str, type: str) -> None:
    """ Download data from Google Drive
    """
    # File download
    gdown.download(url, output, quiet=False)
    
    # Unzip the data
    with zipfile.ZipFile(output, 'r') as zip_ref:
        file_names = zip_ref.namelist()
        with tqdm(total=len(file_names), desc=f"Extracting {type.upper()} files", unit="file") as pbar:
            for file in file_names:
                zip_ref.extract(file, f"data/raw/{type}")
                pbar.update()
    print(f"{type.upper()} data downloaded and extracted to {output}.")
        
def build_directories() -> None:
    """ Build directories for data storage
    """
    folders_created = 0
    if not os.path.exists("data"):
        os.mkdir("data")
        folders_created += 1
    if not os.path.exists("data/raw"):
        os.mkdir("data/raw")
        folders_created += 1
    if not os.path.exists("data/raw/hdf5"):
        os.mkdir("data/raw/hdf5")
        folders_created += 1
    if not os.path.exists("data/raw/json"):
        os.mkdir("data/raw/json")
        folders_created += 1
    if not os.path.exists("data/processed"):
        os.mkdir("data/processed")
        folders_created += 1
    if not os.path.exists("plots"):
        os.mkdir("plots")
        folders_created += 1
    if not os.path.exists("plots/IO"):
        os.mkdir("plots/IO")
        folders_created += 1
    if not os.path.exists("plots/signals"):
        os.mkdir("plots/signals")
        folders_created += 1
    
    print(f"{folders_created} directories created.")

if __name__ == "__main__":
    # HDF5 data
    url_hdf5 = "https://drive.google.com/drive/folders/1h7XJ1OzCvXu0b7K2q5hQPN3q4LqJXRqz"
    output_hdf5 = "data/raw/data_hdf5.zip"
    
    # JSON data
    url_json = "https://drive.google.com/drive/folders/1h0x7Jn7W6TJJCDGG7Fvy_36UdWWdjAbW"
    output_json = "data/raw/data_json.zip"
    
    # Get necessary packages
    os.system("pip install -r requirements.txt")
    
    import gdown
    import zipfile
    from tqdm import tqdm
    
    # Build required directories
    build_directories()
    
    # Download data from Google Drive
    if len(os.listdir("data/raw/hdf5")) == 0:
        get_data(url_hdf5, output_hdf5, 'hdf5')
    else:
        print("HDF5 data already downloaded.")
        
    if len(os.listdir("data/raw/json")) == 0:
        get_data(url_json, output_json, 'json')
    else:
        print("JSON data already downloaded.")
    
