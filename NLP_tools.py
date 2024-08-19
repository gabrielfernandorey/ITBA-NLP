import re, os
import unicodedata
from functools import wraps
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
from opensearch_data_model import TopicKeyword
from opensearch_io import get_entities_news, get_title_news
from collections import defaultdict 
from collections import Counter

#-------------------
class Cleaning_text:
    '''
    Limpiar elementos no deseados del texto 
    '''

    def __init__(self):
        # Definir los caracteres Unicode no deseados
        self.unicode_pattern    = ['\u200e', '\u200f', '\u202a', '\u202b', '\u202c', '\u202d', '\u202e', '\u202f']
        self.urls_pattern       = re.compile(r'http\S+')
        self.simbols_chars      = r"""#&’'"`´“”″()[]*+,-.;:/<=>¿?!¡@\^_{|}~©√≠"""                 # Lista de símbolos a eliminar
        self.simbols_pattern    = re.compile(f"[{re.escape(self.simbols_chars)}]")    
        self.escape_pattern     = ['\n', '\t', '\r']
        
    def _clean_decorator(clean_func):
        @wraps(clean_func)
        def wrapper(self, input_data):
            def clean_string(text):
                return clean_func(self, text)

            if isinstance(input_data, str):
                return clean_string(input_data)
            elif isinstance(input_data, list):
                return [clean_string(item) for item in input_data]
            else:
                raise TypeError("El argumento debe ser una cadena o una lista de cadenas.")
        return wrapper

    @_clean_decorator
    def unicode(self, text):
        for pattern in self.unicode_pattern:
            text = text.replace(pattern, ' ')
        return text

    @_clean_decorator
    def urls(self, text):
        return self.urls_pattern.sub(' ', text)
    
    @_clean_decorator
    def simbols(self, text):
        return self.simbols_pattern.sub(' ', text)

    @_clean_decorator
    def accents_emojis(self, text):
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
    @_clean_decorator
    def escape_sequence(self, text):
        for pattern in self.escape_pattern:
            text = text.replace(pattern, ' ').strip()
        return text
    
    @_clean_decorator
    def str_lower(self, text):
        return text.lower()
    
#---------------------------------------------------------------------------------------------------------
def clean_all(entities, accents=True, lower=True) -> list:
    """
    Función que toma una lista de entidades, realiza una operación de limpieza 
    y devuelve una lista de entidades limpias.
    """
    cleaner = Cleaning_text()

    entities_clean = []
    for ent in entities:
        clean_txt = cleaner.unicode(ent)
        clean_txt = cleaner.urls(clean_txt)
        clean_txt = cleaner.simbols(clean_txt)
        
        if accents:
            clean_txt = cleaner.accents_emojis(clean_txt)

        clean_txt = cleaner.escape_sequence(clean_txt)

        if lower:
            clean_txt = cleaner.str_lower(clean_txt)
        
        entities_clean.append(" ".join(clean_txt.split()))
            
    return entities_clean


#-----------------------------------------------------------------------
def top_keywords(topic_id: int, topic_model: object, PATH: str ) -> dict:
    """
    Funcion que devuelve un diccionario de tuplas con el nombre del keyword y su peso,
    filtrado por un umbral de corte (media)
    """
    try:
        # Stopwords
        SPANISH_STOPWORDS = list(pd.read_csv(PATH+'spanish_stop_words.csv' )['stopwords'].values)
        SPANISH_STOPWORDS_SPECIAL = list(pd.read_csv(PATH+'spanish_stop_words_spec.csv' )['stopwords'].values)
        
        keywords = topic_model.topic_representations_[topic_id]
        topic_keywords = [TopicKeyword(name=keyword, score=score) for keyword, score in keywords if keyword != '']
        
        freq_k = []
        for name_score in topic_keywords:
            freq_k.append(name_score['score'])
        umbral_k = np.array(freq_k).mean()

        topics_keywords_top = {}
        for name_score in topic_keywords:
            if name_score['score'] >= umbral_k and name_score['name'] not in SPANISH_STOPWORDS+SPANISH_STOPWORDS_SPECIAL:
                topics_keywords_top[name_score['name']] = name_score['score']
        
        return topics_keywords_top
    except:
        return {}
       
#-------------------------------------------------------------------------
def top_entities(topic_id: int, topic_model: object, data_news: object) -> dict:
    """
    Las entidades mas representativas del topico se extraen de las entidades mas similares al topico
    """
    try:

        docs_per_topics = [i for i, x in enumerate(topic_model.topics_) if x == topic_id]
        score_docs = topic_model.probabilities_[docs_per_topics]

        doc_probs_x_topic = []
        for i, doc in enumerate(docs_per_topics):
            doc_probs_x_topic.append([data_news.iloc[doc].asset_id, round(score_docs[i],4)])

        df_query = pd.DataFrame(doc_probs_x_topic, columns=['ID','score']).sort_values('score', ascending=False, ignore_index=True)

        # Obtener un umbral de corte para los documentos del topico y filtrar
        ## umbral
        threshold = df_query.score.mean()

        ## Nuevo df filtrado por el corte y ordenado por mayor similitud
        df_filtered = df_query[df_query["score"] > threshold]

        # Entidades de documentos ordenados para el topico elegido (cantidad por documento=n_entities)
        entities_topic = []
        for doc_ID in list(df_filtered["ID"]):
            entities_topic.append(get_entities_news(doc_ID))

        # Crear un diccionario para contar en cuántos documentos aparece cada palabra
        document_frequencies = defaultdict(int)

        # Crear un conjunto para cada documento y contar las palabras únicas
        for lista in entities_topic:
            unique_words = set(lista)
            for palabra in unique_words:
                document_frequencies[palabra] += 1

        # Ordenar las palabras por la frecuencia de documentos de mayor a menor
        sorted_frequencies = sorted(document_frequencies.items(), key=lambda item: item[1], reverse=True)

        # Calcular el umbral
        freq_e = [item[1] for item in sorted_frequencies]
        umbral_e = np.mean(freq_e)

        # Obtener el resultado ordenado de las primeras 10 entidades segun criterio de corte
        topic_entities_top = {}
        c=0
        for idx in range(len(sorted_frequencies)):
            if sorted_frequencies[idx][1] >= umbral_e:
                if c != 10:
                    topic_entities_top[sorted_frequencies[idx][0]] = sorted_frequencies[idx][1]
                else:
                    break
                c += 1 

            elif len(topic_entities_top) <= 3:
                topic_entities_top[sorted_frequencies[idx][0]] = sorted_frequencies[idx][1]
        
        return topic_entities_top
    except:
        return {}
    
#-------------------------------------------------------------------------
def topic_documents(topic_id: int, topic_model: object, data_news: object) -> list:
    """
    función que devuelve los ids de los documentos top del tópico, 
    los titulos de los documentos top del tópico y
    el umbral de corte.
    """
    try:
        # Cantidad de documentos por topico
        docs_per_topics = [i for i, x in enumerate(topic_model.topics_) if x == topic_id]
        score_docs = topic_model.probabilities_[docs_per_topics]

        # Obtener los IDs de los documentos y sus probabilidades 
        doc_probs_x_topic = []
        for i, doc in enumerate(docs_per_topics):
            doc_probs_x_topic.append([data_news.iloc[doc].asset_id, round(score_docs[i],4)])

        df_query = pd.DataFrame(doc_probs_x_topic, columns=['ID', 'score']).sort_values('score', ascending=False, ignore_index=True)
        
        # Calcular la media
        threshold = df_query.score.mean()

        ## Nuevo df filtrado por el corte y ordenado por mayor similitud
        df_filtered = df_query[df_query["score"] > threshold]

        docs_title_topic = []
        for doc_ID in list(df_filtered["ID"]):
            docs_title_topic.append(get_title_news(doc_ID))

        return docs_title_topic, threshold
    except:
        return [], 0.0
    
#---------------------------------------------------------------------------
def best_document(topic, topic_model, docs_embedding, id_data, title_data, data) -> str:
    """
    Función que devuelve el texto del documento mas cercano al topico elegido
    """
    try:       
        # Obtenemos la matriz de similitud coseno entre topicos y documentos
        sim_matrix = cosine_similarity(topic_model.topic_embeddings_, docs_embedding)

        best_doc_index = sim_matrix[topic + 1].argmax()      

        return id_data[best_doc_index], title_data[best_doc_index], data[best_doc_index]

    except:
        return None, "None", "None"
    
#---------------------------------------------------------------------------------------------------------
def get_topic_name(texto: str, topic_id: int, topic_model: object, client, model="gpt-3.5-turbo", solo_titulos=False):
    """
    Funcion que devuelve el nombre de un topico generado por LLM si encuentra modelo, sino labels del topico.
    """
    if client:
        load_dotenv()
        model=os.environ.get('MODEL', model)
        messages = [{"role": "system","content":
                            """ Debes responder generando un topico del texto ingresado de tal forma que sea genérico y representativo,
                                el topico debe estar rigurosamente expresado como maximo en 5 palabras, 
                                el formato de salida debe ser la descipcion del topico."""},
                    {"role": "user", "content": texto}] 
        
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            frequency_penalty=0,
            max_tokens=150,
            n=1,
            presence_penalty=0.6,
            temperature=0.3,
            top_p=1.0,
            stop=None
        )
        message = response.choices[0].message.content
    else:
        topic_labels = topic_model.generate_topic_labels()
        message = topic_labels[topic_id]

    return message

#-----------------------------------------------------------------------------------------------------------
def get_bigrams(word_list, number_consecutive_words=2) -> list:
    """
    Make a function to get all two-word combinations
    """
    
    ngrams = []
    adj_length_of_word_list = len(word_list) - (number_consecutive_words - 1)
    
    #Loop through numbers from 0 to the (slightly adjusted) length of your word list
    for word_index in range(adj_length_of_word_list):
        
        #Index the list at each number, grabbing the word at that number index as well as N number of words after it
        ngram = word_list[word_index : word_index + number_consecutive_words]
        
        #Append this word combo to the master list "ngrams"
        ngrams.append(ngram)
        
    return ngrams

#-----------------------------------------------------------------------------------------------------------
def get_neighbor_words(keyword, bigrams, pos_label = None):
    """
    return the most frequent words that appear next to a particular keyword
    """
    
    neighbor_words = []
    keyword = keyword.lower()
    
    for bigram in bigrams:
        
        #Extract just the lowercased words (not the labels) for each bigram
        words = [word.lower() for word, label in bigram]        
        
        #Check to see if keyword is in the bigram
        if keyword in words:
            idx = words.index(keyword)
            for word, label in bigram:
                
                #Now focus on the neighbor word, not the keyword
                if word.lower() != keyword:
                    #If the neighbor word matches the right pos_label, append it to the master list
                    if label == pos_label or pos_label == None:
                        if idx == 0:
                            neighbor_words.append(" ".join([keyword, word.lower()]))
                        else:
                            neighbor_words.append(" ".join([word.lower(), keyword]))
                    
    return Counter(neighbor_words).most_common()

#-----------------------------------------------------------------------------------------------------------
def keywords_with_neighboards(keywords_spa, POS_1='NOUN', POS_2='ADJ'):
    """
    Funcion que devuelve dos listas:
    - lista de keywords with neighboards (segun argumentos POS_1 y POS_2)
    - lista de keywords mas frecuentes (segun argumentos POS_1 y POS_2)
    """

    doc_kwn = []
    commons = []
    for keywords in keywords_spa:
    
        # Obtenemos las keywords del tipo (Universal Dependences) mas frecuentes de cada doc
        words = []
        for k_spa in keywords:
            if k_spa[1] == POS_1:
                words.append(k_spa[0])

        cont_words = Counter(words)

        common = cont_words.most_common()
        commons.append( [com for com in common if com[1] > 1] )

        # Calcular un umbral de corte (en repeticiones) para los keywords obtenidos
            ## suma de todos los valores
        valores = [valor for _, valor in common]

            ## Calcular los pesos como proporcionales a los valores mismos
        pesos = np.array(valores) / np.sum(valores)

            ## Calcular el umbral ponderado, valor 2 o superior ( debe repetirse la keyword al menos una vez )
        threshold = max(2, round(np.sum(np.array(valores) * pesos),4))

        
        # Obtenemos los bigramas del doc
        tokens_and_labels = [(token[0], token[1]) for token in keywords if token[0].isalpha()]

        bigrams = get_bigrams(tokens_and_labels)

        keywords_neighbor = []
        for item_common in common:
            if item_common[1] >= threshold or len(keywords_neighbor) < 6: # corte por umbral o menor a 6
                item_common[0]
                
                kwn = get_neighbor_words(item_common[0], bigrams, pos_label=POS_2)
                if kwn != []:
                    keywords_neighbor.append( kwn )

        sorted_keywords_neighbor = sorted([item for sublist in keywords_neighbor for item in sublist ], key=lambda x: x[1], reverse=True)
        
        doc_kwn.append(sorted_keywords_neighbor)

    return doc_kwn, commons





        








