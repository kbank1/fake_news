import pandas as pd

from fakenews.preprocessor import remove_duplicates_errors

def load_data() -> pd.DataFrame:
    """
    Load local JSON files from PolitiFact and GossipCop.
    Add target column to all tables and concatenate them.
    Remove duplicates and errors.
    Balance true and false occurrences.
    Return final dataframe.
    """

    politifact_hf = pd.read_json('raw_data/politifact_hf.json', orient='index')
    politifact_hr = pd.read_json('raw_data/politifact_hr.json', orient='index')
    gossipcop_hf = pd.read_json('raw_data/gossipcop_hf.json', orient='index')
    gossipcop_hr = pd.read_json('raw_data/gossipcop_hr.json', orient='index')

    politifact_hf[['fake']] = 1
    politifact_hr[['fake']] = 0
    gossipcop_hf[['fake']] = 1
    gossipcop_hr[['fake']] = 0

    files = [politifact_hf, politifact_hr, gossipcop_hf, gossipcop_hr]
    data = pd.concat(files, ignore_index=True)

    data = remove_duplicates_errors(data)

    true = data[data['fake'] == 0].sample(n=3500)
    false = data[data['fake'] == 1].sample(n=3500)

    files = [true, false]
    data = pd.concat(files, ignore_index=True)

    return data
