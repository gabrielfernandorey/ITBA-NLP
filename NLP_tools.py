import re
import unicodedata
from functools import wraps
import numpy as np
from itertools import islice

from sklearn.metrics.pairwise import cosine_similarity
from opensearch_data_model import TopicKeyword
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


#------------------------------------
def top_keywords(topic, topic_model):
    try:
        keywords = topic_model.topic_representations_[topic]
        topic_keywords = [TopicKeyword(name=keyword, score=score) for keyword, score in keywords if keyword != '']
        
        freq_k = []
        for name_score in topic_keywords:
            freq_k.append(name_score['score'])
        umbral_k = np.array(freq_k).mean()

        topics_keywords_top = {}
        for name_score in topic_keywords:
            if name_score['score'] >= umbral_k:
                topics_keywords_top[name_score['name']] = name_score['score']
        
        return topics_keywords_top
    except:
        return {}
       
#-------------------------------------------------------------------------
def top_entities(topic, topic_model, docs_embedding, data, entities_clean, n_entities=5):

    try:
        # Cantidad de documentos por topico
        T = topic_model.get_document_info(data)
        docs_per_topics = T.groupby(["Topic"]).apply(lambda x: x.index).to_dict()

        # Similitud coseno entre el topico y los documentos del topico
        s_coseno = []
        for i in docs_per_topics[topic]:
            s_coseno.append(cosine_similarity([topic_model.topic_embeddings_[topic + 1]], [docs_embedding[i]])[0][0])

        # Indices
        idx_coseno_sort = np.argsort(s_coseno)[::-1]

        # Ordenado por mayor similitud
        docs_per_topics_unsort = docs_per_topics[topic]

        # Entidades de documentos ordenados por similitud del topico elelgido
        entities_topic = []
        for doc in docs_per_topics_unsort[idx_coseno_sort]:
            entities_topic.append(entities_clean[doc][:n_entities])

        # Crear un diccionario para contar en cuántos documentos aparece cada palabra
        document_frequencies = defaultdict(int)

        # Crear un conjunto para cada documento y contar las palabras únicas
        for lista in entities_topic:
            unique_words = set(lista)
            for palabra in unique_words:
                document_frequencies[palabra] += 1

        # Ordenar las palabras por la frecuencia de documentos de mayor a menor
        sorted_frequencies = sorted(document_frequencies.items(), key=lambda item: item[1], reverse=True)

        freq_e = []
        for item in sorted_frequencies:
            freq_e.append(item[1])
        umbral_e = np.array(freq_e).mean()

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
        
        return topic_entities_top
    except:
        return {}
    
#-------------------------------------------------------------------------
def topic_documents(topic, topic_model, probs, df_news, data):
    """
    función que devuelve los ids de los documentos top del tópico, 
    los titulos de los documentos top del tópico y
    el umbral de corte.
    """
    try:
        # Cantidad de documentos por topico
        docs_per_topics = [i for i, x in enumerate(topic_model.topics_) if x == topic]

        # Obtener los IDs de los documentos y sus probabilidades 
        docs_IDs = {}
        doc_probs_x_topic = []
        for doc_idx in docs_per_topics:
            
            docs_IDs[df_news.indice[doc_idx]] = probs[doc_idx]
            doc_probs_x_topic.append(probs[doc_idx])
        
        # Calcular la media, el desvío estándar
        threshold = np.mean(doc_probs_x_topic)

        # Filtra los docs que superan o igualan al valor del umbral calculado
        filter = {}
        for k,v in docs_IDs.items():
            if v >= threshold:
                filter[k] = v
        
        # Ordeno de mayor a menor
        ids_filter_sort = dict(sorted(filter.items(), key=lambda item: item[1], reverse=True))

        title_filter_sort = [ df_news.loc[df_news['indice'] == idx].values[0][1] for idx in ids_filter_sort.keys() ]

        return ids_filter_sort, title_filter_sort, threshold
    except:
        return {}, {}, 0.0
    
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
    
#---------------------------------------------------------------------------------------------------------
def get_topic_name(texto, client, model="gpt-3.5-turbo", solo_titulos=False):
    """
    Funcion que devuelve el nombre de un topico generado por LLM
    """
    messages = [{"role": "system", "content": "Debes responder el topico del texto ingresado por el usuario," \
                "el topico debe estar expresado en lo posible como maximo en 5 palabras," \
                "el formato de salida debe ser la descipcion del topico."},
                {"role": "user", "content": texto}]
    
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        frequency_penalty=0,
        max_tokens=150,
        n=1,
        presence_penalty=0.6,
        temperature=0.5,
        top_p=1.0,
        stop=None
    )
    message = response.choices[0].message.content

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





        








