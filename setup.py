import os

def get_data(url: str, output: str, type: str=None) -> None:
    """ Download data from Google Drive
    
    Parameters
    __________
    url: str
        URL to download data from
    output: str
        Output file path
    type: Optional[str]
        Type of data to download
    
    Returns
    __________
    None
    """
    
    import gdown
    import zipfile
    from tqdm import tqdm
    
    if type is not None and not isinstance(type, str):
        raise ValueError("`type` must be a string")

    # File download
    try:
        gdown.download(url, output, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"Failed to download file: {e}")
        return

    # Unzip the data
    try:
        with zipfile.ZipFile(output, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            extraction_path = output.split(f'/{type}.zip')[0]
            os.makedirs(extraction_path, exist_ok=True)
            with tqdm(total=len(file_names), desc=f"Extracting {type.upper() if type else 'files'}", unit="file") as pbar:
                for file in file_names:
                    if "__MACOSX" not in file and not file.endswith("Thumbs.db") and not file.startswith('.'):
                        zip_ref.extract(file, extraction_path)
                        pbar.update()
    except zipfile.BadZipFile:
        print(f"Invalid zip file: {output}")
        return
    except Exception as e:
        print(f"Failed to extract files: {e}")
        return

    # Remove temp zip file
    try:
        os.remove(output)
    except Exception as e:
        print(f"Failed to remove temp zip file: {e}")
        return

    print(f"{type.upper() if type else 'Files'} downloaded and extracted to {extraction_path}.")
        
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
    if not os.path.exists("plots/deBode"):
        os.mkdir("plots/deBode")
        folders_created += 1
    if not os.path.exists("plots/spectra"):
        os.mkdir("plots/spectra")
        folders_created += 1
    
    print(f"\n{folders_created} directories created.\n")

if __name__ == "__main__":
    # HDF5 data
    url_hdf5 = "https://drive.google.com/uc?export=download&id=1OpUBaBFCn3mmfZF4MF4V9KhdqmCFHf64"
    output_hdf5 = "data/raw/hdf5.zip"
    
    # JSON data
    url_json = "https://drive.google.com/uc?export=download&id=16JsuEAAvFSSeL6r64ZpznrSsF82iM3No"
    output_json = "data/raw/json.zip"
    
    # Get necessary packages
    try:
        os.system("pip install -r requirements.txt")
    except Exception as e:
        print(f"Failed to install required packages: {e}")

    # Build required directories
    build_directories()
    
    # Download data from Google Drive
    if len(os.listdir("data/raw/hdf5")) == 0:
        get_data(url_hdf5, output_hdf5, 'hdf5')
    else:
        print("HDF5 data already downloaded or 'hdf5' folder occupied.")
        
    if len(os.listdir("data/raw/json")) == 0:
        get_data(url_json, output_json, 'json')
    else:
        print("JSON data already downloaded or 'json' folder occupied.")
    
