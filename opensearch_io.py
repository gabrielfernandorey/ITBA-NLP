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
                '_id': int(df.iloc[idx]['asset_id']),
            }
        }
        reg = {
            'pos_id': -1,
            'asset_id': int(df.iloc[idx]['asset_id']),
            'title': str(df.iloc[idx]['title']),
            'news' : str(text_news), 
            'author': str(df.iloc[idx]['media']),
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
def update_news(documents_ids: list, pos_id: list, docs_embedding, topics: list, probs: list) -> bool:
    """
    Guardar el embedding, topico y probabilidad correspondiente en indice news
    """

    if len(documents_ids) != len(docs_embedding):
        raise ValueError("El número de IDs de documentos y embeddings no coinciden.")
        return False
    
    index_name = 'news'

    # Construir el cuerpo de la solicitud para el API _bulk
    acciones = []
    for doc_id, pid, embedding, topic, prob in zip(documents_ids, pos_id, docs_embedding, topics, probs):
    
        update_body = { 
                        "doc": {
                            "pos_id": pid,
                            "vector": embedding,
                            "topic": topic,   
                            "prob": prob,
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

    try:
        # Realizar la actualización por lotes
        helpers.bulk(os_client, acciones)
    
    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        st.write("Error en el proceso.")

    return True
#----------------------------------------------------------------------------------
def get_news(date: str = None, process: bool = False) -> list:
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
                    {"term": {"process": process}}
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
def get_entities_news(doc_ID: str) -> list :
    """
    Funcion que devuelve una lista de entidades
    de una noticia - (indice news de opensearch)
    """
    try:
        index_name = 'news'

        query = {
            "query": {
                "bool": {
                    "must": [
                        {   "term": {
                                "_id": doc_ID
                            }
                        }
                    ]
                }
            }
        }

        response = os_client.search(index=index_name, body=query)
        entities_doc = [ hit['_source']['entities'] for hit in response['hits']['hits']] 

        return entities_doc[0]
    except:
        return []

#----------------------------------------------------------------------------------
def get_title_news(doc_ID: str) -> list :
    """
    Funcion que devuelve el titulo 
    de una noticia - (indice news de opensearch)
    """
    try:
        index_name = 'news'

        query = {
            "query": {
                "bool": {
                    "must": [
                        {   "term": {
                                "_id": doc_ID
                            }
                        }
                    ]
                }
            }
        }

        response = os_client.search(index=index_name, body=query)
        entities_doc = [ hit['_source']['title'] for hit in response['hits']['hits']] 

        return entities_doc[0]
    except:
        return []
#----------------------------------------------------------------------------------
def get_topics_opensearch(date_filter=None) -> dict:
    """
    Devuelve 1000 registros de topicos del indice topic 
    """
    index_name='topic'
    try:
        if date_filter:
            # Crear el rango de fecha para filtrar el mismo día completo
            date_filter_gte = f"{date_filter}T00:00:00"
            date_filter_lte = f"{date_filter}T23:59:59"

            query = {
                "size": 1000, 
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
                "size": 1000,
                "query": {
                    "match_all": {}
                }
            }
        
        response = os_client.search(index=index_name, body=query)
        topics = [hit['_source'] for hit in response['hits']['hits']]
        return topics
    
    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        return None
#----------------------------------------------------------------------------------
def select_data_from_news(topic: int) -> list:
    
    query = {
        "size": 1000,
        "query": {
            "bool": {
                "must": [
                    {   "term": {
                            "topic": topic
                        }
                    }
                ]
            }
        }
    }
                    
    response = os_client.search(index='news', body=query, scroll='2m')

    # Obtener el scroll ID
    scroll_id = response['_scroll_id']
    total_hits = response['hits']['total']['value']
    all_hits = response['hits']['hits']

    while len(response['hits']['hits']) > 0:
        response = os_client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = response['_scroll_id']
        all_hits.extend(response['hits']['hits'])

    ID    = [hit['_id'] for hit in all_hits]
    title = [hit['_source']['title'] for hit in all_hits]
    news  = [hit['_source']['news'] for hit in all_hits]
    probs = [hit['_source']['prob'] for hit in all_hits]

    return ID, title, news, probs

#-----------------------------------------------------------------------------------
def delete_index_opensearch(index_name: str) -> bool:

    try:
        # Consulta para eliminar todos los documentos
        delete_query = {
                        "query": {
                        "match_all": {}
                        }
        }

        # Ejecutar la operación de borrado por consulta
        response = os_client.delete_by_query(index=index_name, body=delete_query)

        return True

    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        return False

#----------------------------------------------------------------------------------
def get_topics_date() -> list:

    try:
        index_name='topic'
        query = {   "size": 1000,
                    "query": {
                        "match_all": {}
                    }
        }

        response = os_client.search(index=index_name, body=query)
        from_date  = [ hit['_source']['from_date'] for hit in response['hits']['hits']] 
        to_date    = [ hit['_source']['to_date'] for hit in response['hits']['hits']] 
        return from_date, to_date
    
    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        return [],[]

    