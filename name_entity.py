import spacy
import pandas as pd

raw_text="""The Board of Control for Cricket in India (BCCI) is the governing body for cricket in India and is under the jurisdiction of Ministry of Youth Affairs and Sports, Government of India.[2] The board was formed in December 1928 as a society, registered under the Tamil Nadu Societies Registration Act. It is a consortium of state cricket associations and the state associations select their representatives who in turn elect the BCCI Chief. Its headquarters are in Wankhede Stadium, Mumbai. Grant Govan was its first president and Anthony De Mello its first secretary. """
raw_text2 = "Far too much waffling!"
#Loading the name entity recognition in spacy
model = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

#read csv for reviews file
reviews = pd.read_csv('reviews.csv')


for title in reviews['title']:
    print("\nTITLE: ", title)
    #fit the model on the reviews
    md_title= model(title)
    for w in md_title.ents:
        print(w.text,w.label_)

    print('\n-----------------------Next Title------------------------------------')
        



#spacy.displacy.render(text, style="ent",jupyter=True)
