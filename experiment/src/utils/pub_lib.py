import os
import orjson
from pathlib import Path

def get_pub_filepath(base_dir: str, pmid: str) -> str:
    """
    Generate a file path for the publication based on its PMID.

    Args:
        base_dir (str): The base directory where publications are stored.
        pmid (str): The PubMed ID of the publication.

    Returns:
        str: The file path for the publication.
    """
    sub_dir1 = pmid[:2]
    sub_dir2 = pmid[2:4]
    return os.path.join(base_dir, sub_dir1, sub_dir2, f"{pmid}.json")

def save_pub(pub_data: dict, base_dir: str, overwrite: bool = False) -> None:
    """
    Save a publication JSON to the correct subdirectory.

    Args:
        pub_data (dict): The publication data to save.
        base_dir (str): The base directory where publications are stored.
        overwrite (bool): If True, overwrite the file if it exists. Defaults to False.

    Raises:
        ValueError: If PMID is missing in the publication data.
        FileExistsError: If the file already exists and overwrite is False.
    """
    pmid = pub_data.get("PMID")
    if not pmid:
        raise ValueError("PMID is missing in the publication data.")

    filepath = get_pub_filepath(base_dir, pmid)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if not overwrite and os.path.exists(filepath):
        raise FileExistsError(f"File {filepath} already exists.")

    with open(filepath, "wb") as f:
        f.write(orjson.dumps(pub_data))

def read_pub(base_dir: str, pmid: str) -> dict:
    """
    Read a publication JSON from the correct subdirectory.

    Args:
        base_dir (str): The base directory where publications are stored.
        pmid (str): The PubMed ID of the publication.

    Returns:
        dict: The publication data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = get_pub_filepath(base_dir, pmid)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")

    with open(filepath, "rb") as f:
        return orjson.loads(f.read())

def validate_pub(pub_data: dict) -> bool:
    """
    Validate the structure of the publication JSON data.

    Args:
        pub_data (dict): The publication data to validate.

    Returns:
        bool: True if the structure is valid, False otherwise.
    """
    required_fields = {"PMID", "MEDLINE", "PMCID", "PMC"}
    return required_fields.issubset(pub_data.keys())

def pub_exists(base_dir: str, pmid: str) -> bool:
    """
    Check if a publication file already exists for the given PMID.

    Args:
        base_dir (str): The base directory where publications are stored.
        pmid (str): The PubMed ID of the publication.

    Returns:
        bool: True if the publication file exists, False otherwise.
    """
    filepath = get_pub_filepath(base_dir, pmid)
    return os.path.exists(filepath)