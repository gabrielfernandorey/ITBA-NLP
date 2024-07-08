import json, os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import unicodedata

from datetime import datetime
from dateutil.parser import parse
#----------------------------------------------------------------------------------
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode
#----------------------------------------------------------------------------------
from opensearch_data_model import Topic, TopicKeyword, News, os_client, TOPIC_INDEX_NAME, NEWS_INDEX_NAME
#----------------------------------------------------------------------------------
from NLP_tools import Cleaning_text, top_keywords, top_entities, topic_documents, get_topic_name, best_document, clean_all, keywords_with_neighboards
#----------------------------------------------------------------------------------
import spacy
#----------------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
#----------------------------------------------------------------------------------
from openai import OpenAI


#----------------------------------------------------------------------------------
def init_opensearch():
    
    if 'show_message' not in st.session_state:
        st.session_state.show_message = True
        # código de inicialización
        if not os_client.indices.exists(index=TOPIC_INDEX_NAME):
            Topic.init()
            print("Índice Topic creado")
        else:
            print("El índice Topic ya existe. Saltando inicialización de base de datos.")
        
        if not os_client.indices.exists(index=NEWS_INDEX_NAME):
            News.init()
            print("Índice News creado")
        else:
            print("El índice News ya existe. Saltando inicialización de base de datos.")
        
    return 

#----------------------------------------------------------------------------------
def data_ingestion():     
    
    st.title("Ingesta de noticias")
    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')

    # Carga de archivo -------------------------------------------------
    uploaded_file = st.file_uploader(
                        "Upload a news file Parquet",
                        type=["parquet"],
                        help="You must download the news file",
    )

    if not uploaded_file:

        # Crear un cuadro de entrada de texto
        input_text = st.text_area("Or enter only one news text", height=200)

        if st.button("Process News Text"):
            
            process_text(PATH, input_text)

        else:

            st.stop()
    else:
        uploaded_file.read()
        st.success(f"File {uploaded_file.name} uploaded successfully")
    
        # Leer el archivo Parquet
        df = pd.read_parquet(uploaded_file)
    
        # Mostrar el DataFrame
        st.write("Vista previa del archivo Parquet:")
        st.write(df)
        
        # Mostrar total de registro
        st.write("Registros: ", len(df))

        # Boton para procesar el archivo
        if st.button("Process file"):

            process_file(df, PATH)
            
    return
        
#----------------------------------------------------------------------------------
def process_file(df, PATH):

    data = list(df['in__text'])

    # Cargar el modelo de spaCy para español
    spa = spacy.load("es_core_news_lg")

    # Crear un espacio para la barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Detectar entidades y keywords para todos los documentos usando spaCy
    entities_spa = []
    keywords_spa = []

    st.write("Procesando entidades y keywords")
    
    
    for i, doc in enumerate(data):
        procesado = spa(doc)
        entities_spa.append([(ent.text, ent.label_) for ent in procesado.ents])
        keywords_spa.append([(ext.text, ext.pos_) for ext in procesado])    

        # Actualizar la barra de progreso
        progress = (i + 1) / len(data)
        progress_bar.progress(progress)
        status_text.text(f'Procesando documento {i+1} de {len(data)}')
    

    # Stopwords
    SPANISH_STOPWORDS = list(pd.read_csv(PATH+'spanish_stop_words.csv' )['stopwords'].values)
    SPANISH_STOPWORDS_SPECIAL = list(pd.read_csv(PATH+'spanish_stop_words_spec.csv' )['stopwords'].values)

 
    # Procesamiento de entidades encontradas
    entities = []
    original_entities = []
    word_count = {}
    for item in entities_spa:
        for ent in item:
            if ent[1] == 'PER' or ent[1] == 'ORG' or ent[1] == 'LOC':
                words = str(ent[0]).lower()
                words = clean_all([words], accents=False)[0]
                words = " ".join(words.split())
                if len(words) > 2 and len(words.split()) <= 2:   # valida palabra mas de una letra & maximo 2 palabras por token
                    add = True
                    for token in words.split():
                        if token.isalpha():
                            if token in SPANISH_STOPWORDS or token in SPANISH_STOPWORDS_SPECIAL:
                                add = False
                        elif token.isnumeric():
                            if len(token) > 5:
                                add = False
                        else:
                            if token in SPANISH_STOPWORDS_SPECIAL:
                                add = False

                        if add:
                            if words not in word_count:
                                word_count[words] = {'count': 0, 'original': ent[0]}
                            else:
                                word_count[words]['count'] += 1   

        # Ordenar el diccionario por el valor del conteo en orden descendente
        sorted_word_count = dict(sorted(word_count.items(), key=lambda item: item[1]['count'], reverse=True))    
        
        word_count = {}

        pre_original_entities = [value['original'] for _, value in sorted_word_count.items()]
                             
        # Crear la lista de entidades procesadas por noticia (entrenamiento)
        pre_entities = [key for key, _ in sorted_word_count.items()] # if _['count'] > 1]

        # Obtener las últimas palabras de las entidades con más de una palabra
        ultimas_palabras = [ent.split()[-1] for ent in pre_entities if len(ent.split()) > 1]

        # Filtrar si las últimas palabras coinciden con alguna unica palabra
        filtro_ulp = [ent for ent in pre_entities if not (len(ent.split()) == 1 and ent in ultimas_palabras)]

        # Obtener las palabras únicas
        unicas_palabras = [ ent for ent in filtro_ulp if len(ent.split()) == 1]

        # Filtrar si las palabras únicas coinciden con las una entidad con más de una palabra
        filtro_unp = [ ent for ent in filtro_ulp if not ent in unicas_palabras ]

        umbral=10
        # entidades procesadas
        entities.append( filtro_unp[:umbral] )
        original_entities.append([pre for pre in pre_original_entities if pre.lower() in filtro_unp[:umbral]])

    
    # Procesamiento de keywords encontradas como 'MISC'
    keywords = []
    original_keywords = []
    word_count = {}
    for i, item in enumerate(keywords_spa):
        for ent in item:
            if ent[1] == 'MISC':
                words = str(ent[0]).lower()
                words = clean_all([words], accents=False)[0]
                if len(entities[i]) < 5: # Si se encontraron menos de 5 entidades, obtenemos keywords.
                    if len(words) > 2 and len(words.split()) < 2:
                        add = True
                        for token in words.split():
                            if token.isalpha():
                                if token in SPANISH_STOPWORDS or token in SPANISH_STOPWORDS_SPECIAL:
                                    add = False
                            elif token.isnumeric():
                                if len(token) > 5:
                                    add = False
                            else:
                                if token in SPANISH_STOPWORDS_SPECIAL:
                                    add = False

                            if add:
                                if words not in word_count:
                                    word_count[words] = {'count': 0, 'original': ent[0]}
                                else:
                                    word_count[words]['count'] += 1   

        # Ordenar el diccionario por el valor del conteo en orden descendente      
        sorted_word_count = dict(sorted(word_count.items(), key=lambda item: item[1]['count'], reverse=True))  
        
        word_count = {}

        # Crear la lista de entidades procesadas por noticia (para guardar en DB)
        original_keywords.append([value['original'] for _, value in sorted_word_count.items()]) # and value['count'] > 1] )
                                    
        # Crear la lista de entidades procesadas por noticia (entrenamiento)
        pre_keywords = [key for key, _ in sorted_word_count.items()]   # and _['count'] > 1]
                            
        # entidades para entrenar (seleccionamos hasta 5 primeras)
        keywords.append( pre_keywords[:5] )


    k_w_n, common = keywords_with_neighboards(keywords_spa)    
    
    # filtramos que al menos se repitan una vez
    filtered_k_w_n = [ [tupla[0] for tupla in sublista if tupla[1] > 1] for sublista in k_w_n ]
    
    # filtramos hasta 5 keywords
    filtered_common = [ [tupla[0] for i, tupla in enumerate(sublista) if i < 6] for sublista in common ]
    
    # Unificar Keywords + Keywords with neighboards
    keywords_plus = [ list(set(keywords[i]+filtered_k_w_n[i]+filtered_common[i])) for i in range(len(keywords)) ] 


    # Grabar noticias en la base
    st.write("Actualizando noticias en OpenSearch...")

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
            'entities' : original_entities[idx],
            'keywords' : keywords_plus[idx],
            'created_at': datetime.now().isoformat(),
            'process': False
        }
        bulk_data.append(json.dumps(doc))
        bulk_data.append(json.dumps(reg))

    # Convertir la lista en un solo string separado por saltos de línea
    bulk_request_body = '\n'.join(bulk_data) + '\n'

    # Enviar la solicitud bulk
    response = os_client.bulk(body=bulk_request_body)

    if response['errors']:
        st.write("Errores encontrados al insertar los documentos")
    else:
        st.write("Actualizado.")

    return

#----------------------------------------------------------------------------------
def process_text(PATH, input_text):

    try:
        topic_model = BERTopic.load(PATH+"modelos/bertopic_model_app")

        new_doc_embedding = topic_model.embedding_model.embed(input_text)

        # Buscamos en la base a que topico pertenece el nuevo documento
        query = {
            "size": 1,
            "query": {
                "knn": {
                    "vector": {
                        "vector": list(new_doc_embedding),
                        "k" : 3
                    }
                }
            }
        }
        response = os_client.search(index='topic', body=query)
    
        st.write(f" Topico: {response['hits']['hits'][0]['_source']['index']} - {response['hits']['hits'][0]['_source']['name']}")

        estim = round(response['hits']['hits'][0]['_source']['similarity_threshold'],4)
        if estim >= 0.8:
            color = "#80EE19"
            font_size=20
            st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)
        elif estim >= 0.5 and estim < 0.8:
            color = "#EEA719"
            font_size=16
            st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)
        else:
            color = "#FF5733"
            font_size=14
            st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)

    except:
    
        st.warning("Error al estimar el Tópico")

    return
#----------------------------------------------------------------------------------
def view_news():

    def style_tags(tags):
        styled_tags = ' | '.join([f' {tag} ' for tag in tags])
        return styled_tags
    
    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')

    index_name = 'news'
    db_news = []
    for doc in News.search().query().scan():
        index       = doc.meta.id
        title       = doc.to_dict()['title']
        author      = doc.to_dict()['author']
        try:
            keywords =  doc.to_dict()['keywords']
        except:
            keywords = [""]
        try:
            entities =  doc.to_dict()['entities']
        except:
            entities = [""]

        created_at  = doc.to_dict()['created_at'] 

        db_news.append([index, title, style_tags(keywords), style_tags(entities), author, created_at])

    # Convertir a DataFrame
    df = pd.DataFrame(db_news, columns=["indice", "titulo", "keywords", "entidades", "autor", "creado" ])
    

    # Mostrar el DataFrame como una grilla en Streamlit
    st.title("Base de Noticias")
    
    # Configurar opciones de la grilla
    gb = GridOptionsBuilder.from_dataframe(df)

    # Definir anchos de columna específicos
    gb.configure_column('indice', width=70) 
    gb.configure_column('titulo', width=400) 
    gb.configure_column('keywords', width=200) 
    gb.configure_column('entidades', width=200) 
    gb.configure_column('autor', width=50)   
    gb.configure_column('creado', width=50) 
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('single')
    grid_options = gb.build()

    # Mostrar la grilla con AgGrid
    response = AgGrid(
        df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=300,
        width=1400,
    )
    
    # Obtener la selección de la grilla
    selected_rows = response['selected_rows'] if response['selected_rows'] is not None else []

    # Mostrar los datos de la fila seleccionada
    if len(selected_rows) > 0:
        idx = selected_rows["indice"].values
        
        index_name = 'news'
        document_id = int(idx)
        # Realizar la búsqueda del documento por _id
        response = os_client.get(index=index_name, id=document_id)

        st.text_area(f":orange[Texto de la noticia seleccionada] | {response['_source']['title']}", response['_source']['news'], height=200)

        st.text_area(":orange[Keywords de la noticia seleccionada]", style_tags(response['_source']['keywords']), height=100)

        st.text_area(":orange[Entidades de la noticia seleccionada]", style_tags(response['_source']['entities']), height=150)

        # Boton para estimar tópico
        if st.button("Estimar Tópico"):

            try:
                topic_model = BERTopic.load(PATH+"modelos/bertopic_model_app")

                new_doc_embedding = topic_model.embedding_model.embed(response['_source']['news'])

                query = {
                    "size": 5,
                    "query": {
                        "knn": {
                            "vector": {
                                "vector": list(new_doc_embedding),
                                "k" : 1000
                            }
                        }
                    }
                }
                response = os_client.search(index='topic', body=query)
    
                st.write(f" Topico: {response['hits']['hits'][0]['_source']['index']} - {response['hits']['hits'][0]['_source']['name']}")

                estim = round(response['hits']['hits'][0]['_source']['similarity_threshold'],4)
                if estim >= 0.8:
                    color = "#80EE19"
                    font_size=20
                    st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)
                elif estim >= 0.5 and estim < 0.8:
                    color = "#EEA719"
                    font_size=16
                    st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)
                else:
                    color = "#FF5733"
                    font_size=14
                    st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)

            except:
                st.warning("Error al estimar el Tópico")
        
    else:
        st.write("No hay filas seleccionadas.")

    return

#----------------------------------------------------------------------------------
def topic_process(openai_api_key):

    # Boton para procesar el archivo
    if st.button("Iniciar Proceso"):

        load_dotenv()
        PATH=os.environ.get('PATH_LOCAL')

        client = OpenAI(api_key= openai_api_key)

        # Configurar la busqueda de todas las noticias no procesadas ( False ) en la base (al menos 10.000)
        index_name = 'news'
        search_query = {
            'query': {
                'match': {
                    'process': False  
                }
            },
            'size': 10000
        }

        # Realizar la búsqueda
        response = os_client.search( body=search_query, index=index_name )

        db_news = []
        for reg in response['hits']['hits']:
            _id =  reg['_id']
            title =  reg['_source']['title']
            news =  reg['_source']['news']
            try:
                keywords =  reg['_source']['keywords'] 
            except:
                keywords = ['']
            try:
                entities =  reg['_source']['entities'] 
            except:
                entities = ['']
            created_at =  reg['_source']['created_at'] 
            
            db_news.append([_id, title, news, keywords, entities, created_at])

        df_news = pd.DataFrame(db_news , columns=["indice", "titulo", "noticia", "keywords", "entidades", "creado"])
        
        id_data    = list(df_news['indice'])
        title_data = list(df_news['titulo'])
        data       = list(df_news['noticia'])
        entities   = list(df_news['entidades'])
        keywords   = list(df_news['keywords'])

        st.write("Datos recuperados de la base...")
    
        # Vocabulario - Unificar Entities + Keywords
        vocab = list(set().union(*entities, *keywords))

        if vocab == []:
            st.write('<span style="color: yellow;">Vocabulario vacio / sin nuevos datos a procesar...</span>', unsafe_allow_html=True)
            return

        st.write("Datos preprocesados...")

        # Modelo
        tfidf_vectorizer = TfidfVectorizer(
            tokenizer=None,
            max_df=0.9,
            min_df=0.1,
            ngram_range=(1, 2),
            vocabulary=vocab,
            # max_features=100_000
        )
        tfidf_vectorizer.fit(data)

        # Step 1 - Extract embeddings
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        # Step 2 - Reduce dimensionality
        umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine', random_state=42)
        # Step 3 - Cluster reduced embeddings
        hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        # Step 4 - Tokenize topics
        vectorizer_model = tfidf_vectorizer
        # Step 5 - Create topic representation
        ctfidf_model = ClassTfidfTransformer()
        # Step 6 - (Optional) Fine-tune topic representations with a `bertopic.representation` model
        representation_model = KeyBERTInspired()

        # All steps together
        topic_model = BERTopic(
            embedding_model=embedding_model,            # Step 1 - Extract embeddings
            umap_model=umap_model,                      # Step 2 - Reduce dimensionality
            hdbscan_model=hdbscan_model,                # Step 3 - Cluster reduced embeddings
            vectorizer_model=vectorizer_model,          # Step 4 - Tokenize topics
            ctfidf_model=ctfidf_model,                  # Step 5 - Extract topic words
            representation_model=representation_model,  # Step 6 - (Optional) Fine-tune topic represenations
            language='spanish',
            # calculate_probabilities=True
        )

        st.write("Entrenando...")

        # Entrenamiento
        topics, probs = topic_model.fit_transform(data)

        topic_model.save(PATH+"modelos/bertopic_model_app")
        np.save(PATH+"modelos/topics_app.npy", topics)
        np.save(PATH+"modelos/probs_app.npy", probs)
        st.write("Modelo guardado.")

        st.write(f"Topicos encontrados: {len(set(topics))-1}")

        st.write("Generando embeddings...")
        # Obtenemos embeddings de todos los documentos
        docs_embedding = topic_model.embedding_model.embed(data)
        np.save(PATH+"modelos/topic_embeddings_app.npy", docs_embedding)
        st.write("Embeddings guardados.")

        st.write("Actualizando tópicos encontrados...")

        # Grabar todos los topicos en la base
        for topic in topic_model.get_topics().keys():
            if topic > -1:

                topic_keywords_top  = top_keywords(topic, topic_model)
                topic_entities_top  = top_entities(topic, topic_model, docs_embedding, data, entities)
                topic_documents_ids, topic_documents_title, threshold  = topic_documents(topic, topic_model, probs, df_news, data)
                id_best_doc, title_best_doc, best_doc = best_document(topic, topic_model, docs_embedding, id_data, title_data, data)

                topic_doc = Topic(
                    index = topic,   
                    name = get_topic_name(' '.join(topic_documents_title), client),
                    vector = list(topic_model.topic_embeddings_[topic + 1 ]), 
                    similarity_threshold = threshold,                      
                    created_at = datetime.now(),
                    to_date = parse('2024-04-02'),
                    from_date = parse('2024-04-01'),         
                    keywords = topic_keywords_top,
                    entities = topic_entities_top,
                    id_best_doc = id_best_doc,
                    title_best_doc = title_best_doc,
                    best_doc = best_doc,
                    docs = topic_documents_ids
                ) 

                topic_doc.save()

        st.write("Actualizando base de noticias...")      

        # Marcar registros de noticias procesados
        index_name = 'news'
        search_query = {
            'query': {
                'match': {
                    'process': False  
                }
            },
            'size': 10000
        }

        # Realizar la búsqueda
        response = os_client.search( body=search_query, index=index_name )

        for i, reg in enumerate(response['hits']['hits']):
            doc_id = reg['_id']
            
            update_body = {
                            "doc": {                              
                                "process": True
                            }
            }

            # Realizar la actualización
            os_client.update(index=index_name, id=doc_id, body=update_body)

        st.write("Proceso finalizado.")

    return


#----------------------------------------------------------------------------------
def view_all_topics():
    
    try:
        load_dotenv()
        PATH=os.environ.get('PATH_LOCAL')
        topic_model = BERTopic.load(PATH+"modelos/bertopic_model_app")
        
        db_topics = []
        data_topics = {}
        for doc in Topic.search().query().scan():
            index = doc.to_dict()['index']
            name = doc.to_dict()['name']
            similarity_threshold = doc.to_dict()['similarity_threshold']
            create_at = doc.to_dict()['created_at']
            from_date = doc.to_dict()['from_date']
            to_date = doc.to_dict()['to_date']
            id_best_doc = doc.to_dict()['id_best_doc']
            title_best_doc = doc.to_dict()['title_best_doc']
            best_doc = doc.to_dict()['best_doc']

            db_topics.append([index, name, round(similarity_threshold, 4), create_at, from_date, to_date, title_best_doc, id_best_doc])
            data_topics[index] = [doc.to_dict()['name'],
                                  doc.to_dict()['title_best_doc'],
                                  doc.to_dict()['best_doc'],
                                  doc.to_dict()['entities'],
                                  doc.to_dict()['keywords'],
                                  doc.to_dict()['docs']
                                  ]

        # Convertir a DataFrame
        df = pd.DataFrame(db_topics, columns=["indice", "nombre", "umbral", "creado", "desde", "hasta", "titulo noticia mas cercana", "id noticia"])
        

        # Mostrar el DataFrame como una grilla en Streamlit
        st.title("Tópicos de noticias en la base")

        # Configurar opciones de la grilla
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_selection('single')
        grid_options = gb.build()

        # Mostrar la grilla con AgGrid
        response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            height=300,
            width=1400,
        )

        # Obtener la selección de la grilla
        selected_rows = response['selected_rows'] if response['selected_rows'] is not None else []

        # Mostrar los datos de la fila seleccionada
        if len(selected_rows) > 0:
        
            idx = selected_rows["indice"].values

            st.subheader(f":orange[tópico seleccionado:] {data_topics[idx[0]][0]}")

            st.text_area(f":orange[Noticia más cercana al tópico] {data_topics[idx[0]][1]}" , data_topics[idx[0]][2], height=200)

            st.text_area(":orange[Entidades del tópico seleccionado]", " | ".join([ent for ent, _ in data_topics[idx[0]][3].items()]), height=50)

            st.text_area(":orange[Palabras clave del tópico seleccionado]", " | ".join([ent for ent, _ in data_topics[idx[0]][4].items()]), height=50)

            st.subheader(f":orange[otras noticias relacionadas al tópico] : {data_topics[idx[0]][0]}")

            # Crear una consulta de múltiples IDs
            others_docs = []          
            index_name = 'news'
            mget_query = {
                "docs": [{"_index": index_name, "_id": doc_id} for doc_id in data_topics[idx[0]][5]]
            }
            
            # Realizar la búsqueda de múltiples IDs
            response = os_client.mget(body=mget_query, index=index_name)
            
            # Procesar la respuesta
            for i, doc in enumerate(response['docs']):
                if doc['found']:
                    id = doc['_id']
                    title = doc['_source']['title']
                    probs = round(data_topics[idx[0]][5][str(id)],4)
                    others_docs.append([id, title, probs])
            
            df_other_docs = pd.DataFrame(others_docs, columns=['indice', 'titulo', 'probs']) 

            # Configurar opciones de la grilla
            gbo = GridOptionsBuilder.from_dataframe(df_other_docs)

            gbo.configure_pagination(paginationAutoPageSize=True)
            gbo.configure_column('indice', width=120) 
            gbo.configure_column('titulo', width=1280)
            gbo.configure_selection('single')
            grid_options_o = gbo.build()

            # Mostrar la grilla con AgGrid
            response_2 = AgGrid(
                df_other_docs,
                gridOptions=grid_options_o,
                enable_enterprise_modules=True,
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                width=1400,
            )
            
            # Obtener la selección de la grilla
            selected_rows_2 = response_2['selected_rows'] if response_2['selected_rows'] is not None else []

            # Mostrar los datos de la fila seleccionada
            if len(selected_rows_2) > 0:

                idx_2 = selected_rows_2["indice"].values

                index_name = 'news'
                search_query = {
                    'query': {
                        'match': {
                            '_id': int(idx_2)
                        }
                    }
                }

                # Realizar la búsqueda
                response = os_client.search(
                                            body=search_query,
                                            index=index_name
                )
                st.text_area(f":orange[Texto de la noticia] | {response['hits']['hits'][0]['_source']['author']}" \
                             f" | {response['hits']['hits'][0]['_source']['created_at'][:10]} ", response['hits']['hits'][0]['_source']['news'], height=1000)

        else:
            st.write("No hay filas seleccionadas.")
        
    except:
        st.write("Error en el proceso.")
    return
        

#----------------------------------------------------------------------------------
def control():

    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')

    index_name = 'topic'

    db_topics = []
    for i, doc in enumerate(Topic.search().query().scan()):
        db_topics.append(doc.to_dict())

    db_view = [ item['name'] for item in db_topics]

    # Crear una caja de selección múltiple
    seleccionados = st.multiselect("Seleccionar dos o más opciones para agrupar:", db_view)

    # Boton para agrupar
    if st.button("Agrupar topicos"):

        # Mostrar las opciones seleccionadas
        if seleccionados:
            st.write("Has seleccionado:", seleccionados)


        else:
            st.write("No has seleccionado ninguna opción.")

    else:

        st.stop()

    return

   