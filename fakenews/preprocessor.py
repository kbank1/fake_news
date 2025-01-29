import pandas as pd

def remove_duplicates_errors(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by:
    - removing duplicates within fake-category (keep 1)
    - removing duplicates across fake-categories (delete both)
    - deleting texts that are shorter than their title (error messages, headers etc.)
    """
    # Remove duplicates within fake-category
    data = data.drop_duplicates(subset=("text", "fake"), keep='first', ignore_index=True)

    # Remove duplicates across fake-category
    data = data.drop_duplicates(subset=("text"), keep=False, ignore_index=True)

    # Delete false texts
    data["text_len"] = data['text'].str.len()
    data["title_len"] = data['title'].str.len()
    data = data[data["text_len"] >= data["title_len"]]

    data = data.drop(columns=["text_len", "title_len"])

    return data
