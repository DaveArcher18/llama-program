import numpy as np
import pandas as pd
import random
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


# backend model for zero shot object categorizer
classifier_zero_shot = pipeline("zero-shot-classification")

def zero_shot_object_categorizer(text, classifier = classifier_zero_shot):
    '''
    This function takes a piece of text and a zero shot text classifier. It returns 
    a string including the given text and the label assigned to it by the classifier, chosen from 
    the list labels.
    
    Arguments:
    text                    The text that will be categorized 
    classifier              A zero shot text classifier pipeline from Hugging Face
    
    Requirements:
    A hugging face zero shot classifier pipeline. eg:
    classifier_zero_shot = pipeline("zero-shot-classification")
    '''
    labels = ['animal', 'food', 'fruit', 'car', 'boat', 'airplane', 'appliance', 'electronic', 'accessory', 'furniture', 'kitchen', 'cutlery', 'crockery', 'person', 'fish', 'instrument', 'tool', 'sports equipment', 'vehicle', 'holy place', 'power tool']
    out = classifier(text, labels)
    category = out['labels'][np.argmax(out['scores'])]
    return f'this is a picture of {text}, a type of {category}'

# backend for masked language modelling based solutions 
unmasker = pipeline('fill-mask', model = 'roberta-large')

def MLM_object_categorizer(text, unmasker = unmasker):
    """This function uses masked language modelling in order to categorize the given text. The assigned
    category is produced by filling a mask over the category (thus leveraging the original model's training
    data).
    
    Arguments: 
    text                    The text that will be categorized 
    unmasker                A text unmasker pipeline from Hugging Face

    Requirements:
    A hugging face text unmasker pipeline. eg:
    unmasker = pipeline('fill-mask', model = 'roberta-large')
    """
    
    categories = [i['token_str'].lstrip() for i in unmasker(f'{text} is a type of <mask>')]
    if text in categories:
        categories.remove(text)
    return f'this is a picture of {text}, a type of {categories[0]}'
 
object_categorization_data = pd.read_csv('object_categorizer_dataset.csv')    

def random_sample_n_rows(data, n):
    """This funciton randomly samples and concatenates n rows from data. It takes the 
    object and the category and makes a sentance, its then concatenates the n sentances into one string.
    
    Arguments:
    data                A pandas dataframe
    n                   The number of rows
    
    Requirements:
    The object_categorization_data that is contained in this folder as:
    'object_categorizer_dataset.csv'
    """
    indexes = random.sample(range(0, len(data)), n)
    rows = []
    for i in indexes:
       row = data.iloc[i]
       rows.append(f'{row[0]} is a type of {row[1]}')
    return ', '.join(rows)    

def random_sampling_MLM_object_categorizer(text, n = 8, unmasker = unmasker, data = object_categorization_data):
    """This function uses masked language modelling in order to categorize the given text. It is also prompted by 
    n rows sampled from the object_categorizer_data using random_sample_n_rows. Sampling from the dataset is meant
    to improve the assigned label. The assigned category is produced by filling a mask over the category (thus 
    leveraging the original model's training data).
    
    Arguments: 
    text                    The text that will be categorized 
    unmasker                A text unmasker pipeline from Hugging Face
    n                       Number of rows sampled
    data                    A pandas dataframe

    Requirements:
    A hugging face text unmasker pipeline. eg:
    unmasker = pipeline('fill-mask', model = 'roberta-large')
    
    The object_categorization_data that is contained in this folder as:
    'object_categorizer_dataset.csv'

    The random_sample_n_rows function contained in this script

    """
    
    categories = [i['token_str'].lstrip() for i in unmasker(random_sample_n_rows(data, n) + f' {text} is a type of <mask>')]
    if text in categories:
        categories.remove(text)
    return f'this is a picture of {text}, a type of {categories[0]}'

# backend for SBERT object categorizer and SBERT sampling 
SBERT = SentenceTransformer('paraphrase-mpnet-base-v2')

def make_data_embeddings(df, SBERT = SBERT):
    """This function takes a dataframe and an instance of SBERT, makes sentances out of the 
    dataframe's contents and then returns a torch tensor containing the SBERT embeddings for 
    all of the sentances.
    
    Arguments:
    df              A pandas dataframe
    SBERT           An instance of the transformer SBERT

    Requirements:
    An instance of SBERT eg:
    SBERT = SentenceTransformer('paraphrase-mpnet-base-v2')
    """


    sentences = []

    for i in range(len(df)):
        row = df.iloc[i]
        sentences.append(f"{row[0]} is a type of {row[1]}")
    
    embeddings = torch.from_numpy(SBERT.encode(sentences))
    return embeddings

#stores the embeddings
data_embeddings = make_data_embeddings(object_categorization_data)


def find_nearest_row(text, embeddings, SBERT = SBERT):
    """This function takes a piece of text, a tensor of text embeddings and an instance of SBERT. It computes the SBERT embeddings
    of the text and returns the index of the embedding that is nearest to the embedded text using cosine similarity
    
    Arguments:
    text                    The text that will be compared to the embeddings
    embeddings              SBERT embeddings of the corpus that the text will be compared to
    SBERT                   An instance of SBERT

    Requirements:
    A torch tensor containing SBERT embeddings of the object_categorization_data eg:
    data_embeddings = make_data_embeddings(object_categorization_data)
    
    An instance of SBERT eg:
    SBERT = SentenceTransformer('paraphrase-mpnet-base-v2')

    """
    embedded_text = torch.from_numpy(SBERT.encode(f"{text} is a type of"))
    distances = torch.zeros(len(embeddings))
    
    for i in range(len(embeddings)):
        distances[i] = torch.dot(embeddings[i], embedded_text)/(torch.norm(embeddings[i])*torch.norm(embedded_text))
    
    return torch.argmax(distances).item()
                        
    
# states the category is that of the row that is nearest to the given text
def SBERT_object_categorizer(text, embeddings = data_embeddings, df = object_categorization_data):
    """This function uses SBERT similarity to assign a categroy to text from the dataframe df. It finds the row most similar to
    text using the embeddings and find_nearest_row and then assigns the category from the nearest row in the dataframe
    
    Arguments:
    text                    The text that will be categorized
    embeddings              The SBERT embeddings of the rows of df
    df                      The object_categorization_dataset

    Requirements:
    The object_categorization_data that is contained in this folder as:
    'object_categorizer_dataset.csv'
    
    A torch tensor containing SBERT embeddings of the object_categorization_data eg:
    data_embeddings = make_data_embeddings(object_categorization_data)
    
    An instance of SBERT eg:
    SBERT = SentenceTransformer('paraphrase-mpnet-base-v2')
    """
    
    index = find_nearest_row(text, embeddings)
    category = df.iloc[index][1]
    
    return  f'this is a picture of {text}, a type of {category}'


def SBERT_similarity_sampler(text, n = 8, embeddings = data_embeddings, df = object_categorization_data):
    """This functions returns the text of the n most similar rows of df to the given text 
    (in accoradance with SBERT representations and euclidean distance).
    
    Arguments:
    text                    The text for which similar rows will be found
    n                       The number of rows that will be found
    embeddings              The SBERT embeddings of the rows of df
    df                      A pandas dataframe

    Requirements:
    The object_categorization_data that is contained in this folder as:
    'object_categorizer_dataset.csv'
    
    A torch tensor containing SBERT embeddings of the object_categorization_data eg:
    data_embeddings = make_data_embeddings(object_categorization_data)
    
    An instance of SBERT eg:
    SBERT = SentenceTransformer('paraphrase-mpnet-base-v2')
    """
    embedded_text = torch.from_numpy(SBERT.encode(f"{text} is a type of"))
    distances = torch.zeros(len(embeddings))
    
    for i in range(len(embeddings)):
        distances[i] = torch.cdist(embeddings[i].unsqueeze(0), embedded_text.unsqueeze(0))
        
    n_best_indexes = torch.topk(distances, n, largest = False)[1].numpy().tolist()    
    
    rows = []
    for i in n_best_indexes:
        row = df.iloc[i]
        rows.append(f'{row[0]} is a type of {row[1]}')
    
    return ', '.join(rows)

# uses MLM to determine category but also uses SBERT to find prompts similar to given text
def SBERT_MLM_object_categorizer(text, unmasker = unmasker, embeddings = data_embeddings, df = object_categorization_data):
    """This function uses masked language modelling in order to categorize the given text. It is prompted using the 8 rows 
    from object_categorization_data that are most similar to text. The assigned category is produced by filling a mask over 
    the category (thus leveraging the original model's training data).
    
    Arguments: 
    text                    The text that will be categorized 
    unmasker                A text unmasker pipeline from Hugging Face
    embeddings              The SBERT embeddings of the rows of df
    df                      A pandas dataframe

    Requirements:
    A hugging face text unmasker pipeline. eg:
    unmasker = pipeline('fill-mask', model = 'roberta-large')

    Requirements:
    The object_categorization_data that is contained in this folder as:
    'object_categorizer_dataset.csv'
    
    A torch tensor containing SBERT embeddings of the object_categorization_data eg:
    data_embeddings = make_data_embeddings(object_categorization_data)
    
    An instance of SBERT eg:
    SBERT = SentenceTransformer('paraphrase-mpnet-base-v2')
    """
    categories = [i['token_str'].lstrip() for i in unmasker(SBERT_similarity_sampler(text) + f' {text} is a type of <mask>')]
    if text in categories:
        categories.remove(text)
    return f'this is a picture of {text}, a type of {categories[0]}'
