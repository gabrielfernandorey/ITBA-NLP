import os, re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from collections import Counter, defaultdict

from datetime import datetime
from dateutil.parser import parse
#----------------------------------------------------------------------------------
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode
#----------------------------------------------------------------------------------
from opensearch_data_model import Topic, TopicKeyword, News, os_client, TOPIC_INDEX_NAME, NEWS_INDEX_NAME
from opensearch_io import save_news, get_news, update_news

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
def process_text(PATH, input_text):
    """
    Función para estimar el topico de una noticia ingresada manualmente como texto
    """
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

        estim = round(response['hits']['hits'][0]['_score'],4)
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
def data_ingestion():     
    """
    Ingesta de noticias - Proceso por lote o estima un topico de una noticia ingresada manualmente
    """
    st.title("Ingesta de noticias")
    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')

    # Carga de archivo 
    uploaded_file = st.file_uploader(
                        "Cargar un nuevo archivo Parquet",
                        type=["parquet"],
                        help="Debe cargar un archivo de noticias compatible",
    )

    if not uploaded_file:

        # Crear un cuadro de entrada de texto
        input_text = st.text_area("O ingrese una noticia para estimar un tópico", height=200)

        if st.button("Procesar noticia"):
            
            process_text(PATH, input_text) # Estimar topico de noticia ingresada como texto

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
        
        # Mostrar total de registros
        st.write("Registros: ", len(df))

        # Boton para procesar el archivo
        if st.button("Process file"):

            process_file(df, PATH) # Procesar lote de noticias
            
    return
        
#----------------------------------------------------------------------------------
def process_file(df, PATH):
    """
    Proceso de lote de noticias 
    """
        
    data = list(df['in__text'])
    
    # Cargar el modelo de spaCy para español
    spa = spacy.load("es_core_news_lg")

    # Stopwords
    SPANISH_STOPWORDS = list(pd.read_csv(PATH+'spanish_stop_words.csv' )['stopwords'].values)
    SPANISH_STOPWORDS_SPECIAL = list(pd.read_csv(PATH+'spanish_stop_words_spec.csv' )['stopwords'].values)
    
    # Crear un espacio para la barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    st.write("Procesando entidades y keywords...")    

    # Detectar ENTIDADES para todos los documentos usando spaCy
    entities = []
    keywords_spa = []
    for i, data_in in enumerate(data):

        # Actualizar la barra de progreso
        progress = (i + 1) / len(data)
        progress_bar.progress(progress)
        status_text.text(f'Procesando documento {i+1} de {len(data)}')

        # Contabilizar palabras en doc original
        normalized_text = re.sub(r'\W+', ' ', data_in.lower())
        words_txt_without_stopwords = [word for word in normalized_text.split() if word not in SPANISH_STOPWORDS+SPANISH_STOPWORDS_SPECIAL]
        words_txt_counter = Counter(words_txt_without_stopwords)
        words_counter = {elemento: cuenta for elemento, cuenta in sorted(words_txt_counter.items(), key=lambda item:item[1], reverse=True) if cuenta > 1}

        # Extraer entidades del doc segun atributos
        extract = spa(data_in)
        entidades_spacy = [(ent.text, ent.label_) for ent in extract.ents]
        ent_select = [ent for ent in entidades_spacy if ent[1] == 'PER' or ent[1] == 'ORG' or ent[1] == 'LOC' ]

        # Extraer keywords del doc
        keywords_spa.append([(ext.text, ext.pos_) for ext in extract]) 

        # Extraer entidades de "maximo 3 palabras"
        ent_max_3 = [ent[0] for ent in ent_select if len(ent[0].split()) <= 3]
        ent_clean = clean_all(ent_max_3, accents=False)
        ent_unique = list(set([ word for word in ent_clean if word not in SPANISH_STOPWORDS+SPANISH_STOPWORDS_SPECIAL] ))

        ents_proc = {}
        for ent in ent_unique:
            
            # Criterio de selección 
            weight = 0
            for word in ent.split():
                if word in words_counter:
                    weight += 1 /len(ent.split()) * words_counter[word]
            
            ents_proc[ent] = round(weight,4)

        ents_proc_sorted = {k: v for k, v in sorted(ents_proc.items(), key=lambda item: item[1], reverse=True) if v > 0}

        # Crear la lista preliminar de entidades procesadas por noticia 
        pre_entities = [key for key, _ in ents_proc_sorted.items()] 

        # Obtener las últimas palabras de cada entidad que tenga mas de una palabra por entidad
        last_words = list(set([ent.split()[-1] for ent in pre_entities if len(ent.split()) > 1 ]))

        # Eliminar palabra única si la encuentra al final de una compuesta
        pre_entities_without_last_word_equal = []
        for idx, ent in enumerate(pre_entities):
            if not (len(ent.split()) == 1 and ent in last_words):
                pre_entities_without_last_word_equal.append(ent)

        # Obtener las palabras únicas
        unique_words = [ ent.split()[0] for ent in pre_entities_without_last_word_equal if len(ent.split()) > 1 ]

        # Eliminar palabra única si la encuentra al comienzo de una compuesta
        pre_entities_without_first_word_equal = []
        for idx, ent in enumerate(pre_entities_without_last_word_equal):
            if not (len(ent.split()) == 1 and ent in unique_words):
                pre_entities_without_first_word_equal.append(ent)

        # obtener entidades filtradas
        if len(pre_entities_without_first_word_equal) > 10:
            umbral = 10 + (len(pre_entities_without_first_word_equal)-10) // 2
            filter_entities = pre_entities_without_first_word_equal[:umbral] 
        else:
            filter_entities = pre_entities_without_first_word_equal[:10]

        pre_original_entities = []
        # capturar las entidades en formato original
        for ent in filter_entities:
            pre_original_entities.append([elemento for elemento in ent_max_3 if elemento.lower() == ent.lower()])

        sort_original_entities = sorted(pre_original_entities, key=len, reverse=True)
        
        try:
            entities.append( [ent[0] for ent in sort_original_entities if ent] ) 
        except Exception as e:
            entities.append([])

    
    # Detectar KEYWORDS with neighboards y keywords mas frecuentes
    k_w_n, keyword_single = keywords_with_neighboards(keywords_spa)
    
    # filtramos las que al menos se repiten una vez
    filtered_k_w_n = [ [tupla[0] for tupla in sublista if tupla[1] > 1] for sublista in k_w_n ]
    
    # Si un keyword unigrama coincide en los bigramas elegidos se descarta.
    # la cantidad de keywords se obtiene utilizando la media como umbral de corte
    
    values = [value for sublist in keyword_single for _, value in sublist]
    threshold = np.mean(values) # Umbral

    for i, sublist in enumerate(keyword_single):
        lista_k_w_n = list(set([word for sentence in filtered_k_w_n[i] for word in sentence.split()]))
        for tupla in sublist:
            if tupla[1] >= threshold and tupla[0] not in lista_k_w_n:
                filtered_k_w_n[i].append(tupla[0])

    keywords = filtered_k_w_n 

    # Grabar noticias en la base
    st.write("Actualizando noticias en OpenSearch...")

    response = save_news(data, df, entities, keywords)  

    if response['errors']:
        st.write("Errores encontrados al insertar los documentos")
    else:
        st.write("Actualizado.")

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
        process     = doc.to_dict()['process'] 

        db_news.append([index, title, style_tags(keywords), style_tags(entities), author, created_at, process])

    # Convertir a DataFrame
    df = pd.DataFrame(db_news, columns=["indice", "titulo", "keywords", "entidades", "autor", "creado", "procesado" ])
    

    # Mostrar el DataFrame como una grilla en Streamlit
    st.title("Base de Noticias")
    
    # Configurar opciones de la grilla
    gb = GridOptionsBuilder.from_dataframe(df)

    # Definir anchos de columna específicos
    gb.configure_column('indice', width=70) 
    gb.configure_column('titulo', width=400) 
    gb.configure_column('keywords', width=200) 
    gb.configure_column('entidades', width=200) 
    gb.configure_column('autor', width=100)   
    gb.configure_column('creado', width=100) 
    gb.configure_column('procesado', width=100) 
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

        st.text_area(":orange[Entidades de la noticia seleccionada]", style_tags(response['_source']['entities']), height=100)

        # Boton para estimar tópico
        if st.button("Estimar Tópico"):

            try:
                topic_model = BERTopic.load(PATH+"modelos/bertopic_model_app")
                new_doc_embedding = topic_model.embedding_model.embed(response['_source']['news'])

                # Buscamos en la base a que topico pertenece el nuevo documento
                knn_query = {
                    "size": 1,
                    "query": {
                        "knn": {
                            "vector": {
                                "vector": new_doc_embedding,
                                "k" : 3
                            }
                        }
                    }
                }
                response = os_client.search(index='topic', body=knn_query)

                if response['hits']['total']['value'] > 0:
                    st.write(f" Topico: {response['hits']['hits'][0]['_source']['index']} - {response['hits']['hits'][0]['_source']['name']}")

                    estim = round(response['hits']['hits'][0]['_score'],4)
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
                else:
                    st.warning("No se puede estimar el Tópico")
            except:
                st.warning("Error al estimar el Tópico")
        
    else:
        st.write("No hay filas seleccionadas.")

    return

#----------------------------------------------------------------------------------
def topic_process(openai_api_key):
    """
    Proceso para la generación de tópicos
    """
    db_news = get_news()

    # Crear un diccionario para agrupar los registros por fecha (solo día, mes y año)
    fechas_dict = defaultdict(list)

    # Agrupar registros por fecha
    for registro in db_news:
        fecha_completa = registro[-1]
        fecha_solo_dia = fecha_completa.split('T')[0]  # Tomar solo el día, mes y año
        if fecha_solo_dia not in fechas_dict:
            fechas_dict[fecha_solo_dia] = 1
        else:
            fechas_dict[fecha_solo_dia] += 1


    # Convertir el diccionario en un DataFrame
    df = pd.DataFrame(list(fechas_dict.items()), columns=['Fecha', 'Cantidad de noticias'])

    st.subheader(" Elegir lote/s de noticias a procesar")
    # Mostrar el DataFrame en una grilla seleccionable
    st.dataframe(df)

    # Crear una lista de opciones basadas en el DataFrame
    options = df.index.tolist()
    selected_indices = st.multiselect('Seleccione las filas', options, format_func=lambda x: df.loc[x, 'Fecha'])

    # Filtrar el DataFrame basado en la selección
    if selected_indices:
        selected_rows = df.loc[selected_indices]
        
    else:
        st.write('No se han seleccionado filas.')

    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')

    # Boton para procesar el archivo
    if st.button("Iniciar Proceso"):
       
        client = OpenAI(api_key= openai_api_key)

        # Busqueda de todas las noticias de la base
        db_news = get_news()

        if db_news == []:
            st.write('<span style="color: yellow;">Sin nuevos datos a procesar...</span>', unsafe_allow_html=True)
            return
        
        df_news = pd.DataFrame(db_news , columns=["indice", "titulo", "noticia", "keywords", "entidades"])

        id_data    = list(df_news['indice'])
        title_data = list(df_news['titulo'])
        data       = list(df_news['noticia'])
        entities   = list(df_news['entidades'])
        keywords   = list(df_news['keywords'])

        st.write("Datos recuperados de la base...")
    
        # Vocabulario - Unificar Entities + Keywords
        vocab = list(set().union(*entities, *keywords))
           
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
        umap_model = UMAP(n_neighbors=15, n_components=6, min_dist=0.0, metric='cosine', random_state=42)
        # Step 3 - Cluster reduced embeddings
        hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        # Step 4 - Tokenize topics
        vectorizer_model = tfidf_vectorizer
        # Step 5 - Create topic representation
        ctfidf_model = ClassTfidfTransformer()
        # Step 6 - (Optional) Fine-tune topic representations with a `bertopic.representation` model
        # representation_model = KeyBERTInspired()

        # All steps together
        topic_model = BERTopic(
            embedding_model=embedding_model,                # Step 1 - Extract embeddings
            umap_model=umap_model,                          # Step 2 - Reduce dimensionality
            hdbscan_model=hdbscan_model,                    # Step 3 - Cluster reduced embeddings
            vectorizer_model=vectorizer_model,              # Step 4 - Tokenize topics
            ctfidf_model=ctfidf_model,                      # Step 5 - Extract topic words
            # representation_model=representation_model,    # Step 6 - (Optional) Fine-tune topic represenations
            # language='multilingual',                      # This is not used if embedding_model is used.
            # calculate_probabilities=True,                 
            verbose=True,
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


        # Crear un espacio para la barra de progreso
        st.write("Actualizando tópicos encontrados...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        

        # Grabar todos los topicos en la base
        for i, topic in enumerate(topic_model.get_topics().keys()):

            # Actualizar la barra de progreso
            progress = (i+1) / len(set(topics))
            progress_bar.progress(progress)
            status_text.text(f'Procesando topico {i} de {len(set(topics))-1}')
            
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
                    from_date = parse('2024-04-01'),
                    to_date = parse('2024-04-02'),         
                    keywords = topic_keywords_top,
                    entities = topic_entities_top,
                    id_best_doc = id_best_doc,
                    title_best_doc = title_best_doc,
                    best_doc = best_doc,
                ) 
                
                # Grabar topico
                topic_doc.save()

        st.write("Actualizando base de noticias...")           
        # Marcar registros de noticias como procesados y grabar sus embeddings
        # update_news([ int(ID) for ID in topic_documents_ids.keys() ], docs_embedding)

        st.write("Proceso finalizado.")

    return
#----------------------------------------------------------------------------------
def view_all_topics():
    
    try:
        load_dotenv()
        PATH=os.environ.get('PATH_LOCAL')
        topics = np.load(PATH+"modelos/topics_app.npy")
        probs = np.load(PATH+"modelos/probs_app.npy")

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
            

            db_topics.append([index, name, round(similarity_threshold, 4), create_at, from_date, to_date, title_best_doc, id_best_doc])
            data_topics[index] = [doc.to_dict()['name'],
                                  doc.to_dict()['title_best_doc'],
                                  doc.to_dict()['best_doc'],
                                  doc.to_dict()['entities'],
                                  doc.to_dict()['keywords'],
                                  doc.to_dict()['similarity_threshold']
                                  ]

        # Convertir a DataFrame
        df = pd.DataFrame(db_topics, columns=["indice", "nombre", "umbral", "creado", "desde", "hasta", "titulo noticia mas cercana", "id noticia"])
        
        # Mostrar el DataFrame como una grilla en Streamlit
        st.title("Tópicos de noticias")

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

            # Obtener noticias de la base
            db_news = get_news()

            # Obtener los idx de las noticias del topico
            topic_docs_idx = [i for i, topic in enumerate(topics) if topic == idx]

            # Preparar datos de otros documentos
            others_docs = []
            for i in topic_docs_idx:
                others_docs.append([db_news[i][0], db_news[i][1], round(probs[i],4)] )
            
            df_other_docs = pd.DataFrame(others_docs, columns=['indice', 'titulo', 'probs']) 

            # Ordenar docs y filtrar por umbral del topico
            df_other_docs_sorted = df_other_docs.sort_values('probs', ascending=False)
            df_other_docs_filtered = df_other_docs_sorted[df_other_docs_sorted['probs'] > float(data_topics[idx[0]][5])]
            

            # Configurar opciones de la grilla
            gbo = GridOptionsBuilder.from_dataframe(df_other_docs_filtered)

            gbo.configure_pagination(paginationAutoPageSize=True)
            gbo.configure_column('indice', width=120) 
            gbo.configure_column('titulo', width=1280)
            gbo.configure_selection('single')
            grid_options_o = gbo.build()

            # Mostrar la grilla con AgGrid
            response_2 = AgGrid(
                df_other_docs_filtered,
                gridOptions=grid_options_o,
                enable_enterprise_modules=True,
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=350,
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
                             f" | {response['hits']['hits'][0]['_source']['created_at'][:10]} ", response['hits']['hits'][0]['_source']['news'], height=400)
                
                st.text_area(f":orange[Keywords de la noticia]", " | ".join([ent for ent in response['hits']['hits'][0]['_source']['keywords']]) ) 
                st.text_area(f":orange[Entidades de la noticia]", " | ".join([ent for ent in response['hits']['hits'][0]['_source']['entities']]) ) 
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

   