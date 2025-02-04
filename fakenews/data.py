import pandas as pd

from fakenews.preprocessor import remove_duplicates_errors

def load_data() -> pd.DataFrame:

    def load_data_fakenewsnet() -> pd.DataFrame:
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

        data = remove_duplicates_errors(data, title=True)

        true = data[data['fake'] == 0].sample(n=3500)
        false = data[data['fake'] == 1].sample(n=3500)

        files = [true, false]
        fakenewsnet = pd.concat(files, ignore_index=True)

        return fakenewsnet


    def load_data_guardian_corpus() -> pd.DataFrame:
        """
        Load local csv files from Fakenewscorpus and the Guardian.
        Both were sampled from larger datasets (5000 each).
        Add target column to all tables and concatenate them.
        Remove duplicates and errors.
        Sample 7000 observations to match Fakenewsnet Data size.
        Return final dataframe.
        """

        corpus_fake = pd.read_csv('raw_data/corpus_fake.csv')
        guardian_new = pd.read_csv('raw_data/guardian_new.csv')

        corpus_fake["fake"]=1
        corpus_fake["text"]=corpus_fake["content"]
        guardian_new["fake"]=0
        guardian_new["text"]=guardian_new["Content"]

        t1="skip past newsletter promotionSign up to The BreakdownFree weekly newsletterThe latest rugby union news and analysis, plus all the week's action reviewed"
        t2="Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy. We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply"
        t3="skip past newsletter promotionSign"
        t4=".after newsletter promotion"

        tlist=[t1, t2, t3, t4]

        def clean_guardian(text):
            for t in tlist:
                text = text.replace(t, '')
            return text

        guardian_new["text"]=guardian_new.text.apply(clean_guardian)

        files = [corpus_fake, guardian_new]
        data = pd.concat(files, ignore_index=True)

        data = remove_duplicates_errors(data, title=False)

        true = data[data['fake'] == 0].sample(n=3500)
        false = data[data['fake'] == 1].sample(n=3500)

        files = [true, false]
        fakenewscorpus = pd.concat(files, ignore_index=True)

        return fakenewscorpus

    fakenewsnet=load_data_fakenewsnet()
    fakenewscorpus=load_data_guardian_corpus()

    files = [fakenewsnet, fakenewscorpus]
    data = pd.concat(files, ignore_index=True)
    print("data successfully loaded")
    print (data.shape)

    return data
