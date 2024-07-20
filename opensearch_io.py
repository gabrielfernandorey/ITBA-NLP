from dateutil.parser import parse
from datetime import datetime
import json

import streamlit as st

from opensearch_data_model import Topic, News, os_client, TOPIC_INDEX_NAME, NEWS_INDEX_NAME
from opensearchpy import helpers


#----------------------------------------------------------------------------------
def init_opensearch():
    """
    Inicializa los indices de la base Opensearch si no fueron creados
    """
    if 'show_message' not in st.session_state:
        st.session_state.show_message = True
        # código de inicialización
        if not os_client.indices.exists(index=TOPIC_INDEX_NAME):
            Topic.init()
            #index_name = 'topic'
            #body = {"settings": {"index.mapping.total_fields.limit": 2000 }}
            #os_client.indices.put_settings(index=index_name, body=body)
            print("Indice Topic creado")
        else:
            print("El índice Topic ya existe. Saltando inicialización de base de datos.")
        
        if not os_client.indices.exists(index=NEWS_INDEX_NAME):
            News.init()
            print("Indice News creado")
        else:
            print("El índice News ya existe. Saltando inicialización de base de datos.")
        
    return 
#----------------------------------------------------------------------------------
def save_news(data, df, entities, keywords) -> str:
    """
    Almacenar las noticias en el indice de la base 
    """
    index_name = 'news'
    bulk_data = []

    for idx, text_news in enumerate(data):
        doc = {
            'index': {
                '_index': index_name,
                '_id': int(df.index[idx])
            }
        }
        reg = {
            'title': str(df.iloc[idx].in__title),
            'news' : str(text_news), 
            'author': str(df.iloc[idx]['Author Name']),
            'vector': None,
            'keywords' : keywords[idx],
            'entities' : entities[idx],
            'created_at': parse(str(df.iloc[idx]['start_time_local'])).isoformat(),
            'process': False
        }
        bulk_data.append(json.dumps(doc))
        bulk_data.append(json.dumps(reg))

    # Convertir la lista en un solo string separado por saltos de línea
    bulk_request_body = '\n'.join(bulk_data) + '\n'

    # Enviar la solicitud bulk
    response = os_client.bulk(body=bulk_request_body)

    return response
#----------------------------------------------------------------------------------
def update_news(documents_ids: list, docs_embedding) -> bool:
    """
    Marcar en True los registros de noticias procesadas y guardar el embedding correspondiente.
    """

    if len(documents_ids) != len(docs_embedding):
        raise ValueError("El número de IDs de documentos y embeddings no coinciden.")
        return False
    
    index_name = 'news'

    # Construir el cuerpo de la solicitud para el API _bulk
    acciones = []
    for doc_id, embedding in zip(documents_ids, docs_embedding):
        update_body = { 
                        "doc": {
                            "vector": embedding,  
                            "process": True,
                        }
        }
        accion = {
            "_op_type": "update",
            "_index": index_name,
            "_id": doc_id,
            "doc": update_body["doc"]
        }
        acciones.append(accion)

    # Realizar la actualización por lotes
    helpers.bulk(os_client, acciones)

    return True
#----------------------------------------------------------------------------------
def get_news(date: str = None) -> list:
    """
    Obtener las noticias de la base que tengan el campo 'process' en False y, opcionalmente, filtradas por fecha.
    Parámetros:
    - date: Fecha en formato 'día-mes-año'
    """
    index_name = 'news'
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match_all": {}}
                ],
                "filter": [
                    {"term": {"process": False}}
                ]
            }
        }
    }

    # Agregar filtro de fecha si se proporciona
    if date:
        try:
            year, month, day  = map(int, date.split('-'))
            date_filter = {
                "range": {
                    "created_at": {
                        "gte": f"{year}-{month:02d}-{day:02d}T00:00:00",
                        "lte": f"{year}-{month:02d}-{day:02d}T23:59:59"
                    }
                }
            }
            query["query"]["bool"]["filter"].append(date_filter)
        except ValueError:
            print(day, month, year)
            raise ValueError("La fecha debe estar en el formato 'día-mes-año' ")


    # Inicializar la búsqueda con scroll
    scroll = '2m'  # Mantener el contexto de scroll por 2 minutos
    response = os_client.search(index=index_name, body=query, scroll=scroll, size=1000) # mantener en 1000 para baja latencia

    # Obtener el ID de scroll
    scroll_id = response['_scroll_id']
    documents = [hit['_source'] for hit in response['hits']['hits']]
    doc_ID = [hit['_id'] for hit in response['hits']['hits']]

    # Seguir buscando hasta que no queden más resultados
    while True:
        response = os_client.scroll(scroll_id=scroll_id, scroll=scroll)
        if len(response['hits']['hits']) == 0:
            break
        scroll_id = response['_scroll_id']
        documents.extend([hit['_source'] for hit in response['hits']['hits']])
        doc_ID.extend([hit['_id'] for hit in response['hits']['hits']])

    db_news = []
    for idx in range(len(documents)):
    
        _id   =  doc_ID[idx]
        title =  documents[idx]['title']
        news  =  documents[idx]['news']
        
        try:
            keywords =  documents[idx]['keywords']
        except:
            keywords = ['']
        try:
            entities =  documents[idx]['entities']
        except:
            entities = ['']
        
        created_at = documents[idx]['created_at']
                   
        db_news.append([_id, title, news, keywords, entities, created_at])

    return db_news
#----------------------------------------------------------------------------------
def get_topics_opensearch(date_filter=None) -> dict:
    """
    Devuelve 100 registros de topicos del indice topic
    """
    index_name='topic'
    if date_filter:
        # Crear el rango de fecha para filtrar el mismo día completo
        date_filter_gte = f"{date_filter}T00:00:00"
        date_filter_lte = f"{date_filter}T23:59:59"

        query = {
            "size": 100, 
            "query": {
                "bool": {
                    "must": [
                        {"range": {"to_date": {"gte": date_filter_gte, "lte": date_filter_lte}}}
                    ]
                }
            }
        }
    else:
        query = {
            "query": {
                "match_all": {}
            }
        }
    
    response = os_client.search(index=index_name, body=query)
    topics = [hit['_source'] for hit in response['hits']['hits']]
    return topics
#----------------------------------------------------------------------------------
