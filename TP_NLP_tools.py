from opensearch_data_model import Topic, News, os_client, TOPIC_INDEX_NAME
from opensearchpy import helpers
from opensearchpy import Float, OpenSearch, Field, Integer, Document, Keyword, Text, Boolean, DenseVector, Nested, Date, Object, connections, InnerDoc, helpers


import re, os
import unicodedata
from functools import wraps
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
from opensearch_data_model import TopicKeyword
from collections import defaultdict 
from collections import Counter

#-----------------------------------------------------------------------------

OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', "localhost")
auth = ('admin', 'PassWord#1234!')
port = 9200
os_client = connections.create_connection(
    hosts = [{'host': OPENSEARCH_HOST, 'port': port}],
    http_auth = auth,
    http_compress = True, # enables gzip compression for request bodies
    use_ssl = True,
    verify_certs = False,
    alias='default'
    # ssl_assert_hostname = False,
    # ssl_show_warn = False
)

# Index Topic
TOPIC_DIMENSIONS = 384
TOPIC_INDEX_NAME = 'topic'
TOPIC_INDEX_PARAMS = {
    'number_of_shards': 1,      # 1 fragmento (no hay replicas)
    'knn': True                 # aplicamos knn para que la base utilice como algoritmo de clusterizacion
}

knn_params = {                      # parámetros de knn
    "name": "hnsw",
    "space_type": "cosinesimil",    # usamos similitud coseno
    "engine": "nmslib"              # tipo de algortimo que va a utilizar
}
#-------------------------------------------------------------------------------------------------
class TopicKeyword(InnerDoc):
    name = Keyword()
    score = Float()

class TopicEntities(InnerDoc):
    name = Keyword()
    score = Float()

class DocsProbs(InnerDoc):
    doc_ID = Keyword()
    score = Float()

class KNNVector(Field):
    name = "knn_vector"
    def __init__(self, dimension, method, **kwargs):
        super(KNNVector, self).__init__(dimension=dimension, method=method, **kwargs)

class SimilarTopics(Document):
    topic_id = Keyword()
    similar_to = Keyword()
    similarity = Float()
    common_keywwords = Keyword()
    keywords_not_in_similar = Keyword()
    keywords_not_in_topic = Keyword()

class Topic(Document):
    index = Integer()                                   # nro topico
    name = Text()                                       # nombre del topico
    vector = KNNVector(TOPIC_DIMENSIONS, knn_params)    # vector
    similarity_threshold = Float()                      # umbral
    created_at = Date()                                 # fecha de creacion
    to_date = Date()                                    # fecha de entrenamiento
    from_date = Date()                                  # fecha de entrenamiento
    keywords = Object(TopicKeyword)                     # keywords del topico (representacion) 
    entities = Object(TopicEntities)                    # entities del topico
    id_best_doc = Integer()                             # id del mejor documento
    title_best_doc = Text()                             # titulo del mejor documento
    best_doc = Text()                                   # texto del mejor documento
    
    class Index:
        name = TOPIC_INDEX_NAME
        if not os_client.indices.exists(index=TOPIC_INDEX_NAME):
            settings = {
                'index': TOPIC_INDEX_PARAMS
            }

    def save(self, ** kwargs):
        self.meta.id = f'{self.index}' + '_' + self.name.replace(' ', '_')
        return super(Topic, self).save(** kwargs)

#----------------------------------------------------------------------------------
def init_opensearch():
    """
    Inicializa los indices de la base Opensearch si no fueron creados
    """
    # código de inicialización
    if not os_client.indices.exists(index=TOPIC_INDEX_NAME):
        Topic.init()
        print("Indice Topic creado")
    else:
        print("El índice Topic ya existe. Saltando inicialización de base de datos.")
        
    return 


#--------------------------------------------------------------------------------------
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
def top_keywords(topic_id: int, topic_model: object) -> dict:
    """
    Funcion que devuelve un diccionario de tuplas con el nombre del keyword y su peso,
    filtrado por un umbral de corte (media)
    """
    try:
        # Stopwords
        SPANISH_STOPWORDS = list(pd.read_csv('spanish_stop_words.csv' )['stopwords'].values)
        SPANISH_STOPWORDS_SPECIAL = list(pd.read_csv('spanish_stop_words_spec.csv' )['stopwords'].values)
        
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
def top_entities(topic_id: int, topic_model: object, data_news: object, n_entities=10):
    """
    Las entidades mas representativas del topico se extraen de las entidades de las noticias mas similares al topico
    filtradas por el umbral del tópico
    topic_id        : id del topico
    topic_model     : modelo de topicos
    data_news       : dataframe de documentos seleccionados para el proceso
    n_entities      : cant. de entidades extraidas por cada documento del topico
    """

    try:
        # Obtener el numero ID y la pos_relativa para cada documento del topico seleccionado,  ordenados de mayor a menor por su probabilidad
        docs_per_topics = [i for i, x in enumerate(topic_model.topics_) if x == topic_id]
        probs = topic_model.probabilities_[docs_per_topics]

        doc_probs_x_topic = []
        for i, doc in enumerate(docs_per_topics):
            doc_probs_x_topic.append([data_news.index[doc], doc, data_news.iloc[doc].title, round(probs[i],4)])
            
        df_query = pd.DataFrame(doc_probs_x_topic, columns=['ID','pos_rel','titulo','score']).sort_values('score', ascending=False, ignore_index=True)

        # Obtener un umbral de corte para los documentos del topico y filtrar
        ## umbral
        threshold = df_query['score'].mean()

        ## Nuevo df filtrado por el corte y ordenado por mayor similitud
        df_filtered = df_query[df_query["score"] > threshold]
        
        # Entidades de documentos ordenados para el topico elelgido (cantidad por documento=n_entities)
        entities_topic = []
        for doc in list(df_filtered["ID"]):
            entities_topic.append(data_news.loc[doc]["entities"][:n_entities])

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
    
    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        return False   
    
#-------------------------------------------------------------------------
def topic_threshold(topic_id, topic_model, probs):
    """
    función que devuelve los ids de los documentos top del tópico, 
    los titulos de los documentos top del tópico y
    el umbral de corte.
    """
    try:
        docs_per_topics = [i for i, x in enumerate(topic_model.topics_) if x == topic_id]

        return np.array([ probs[doc_idx] for doc_idx in docs_per_topics ]).mean()
    
    except:
        return 0
    
#------------------------------------------------------
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

#----------------------------------------------------------------------------------------------------
