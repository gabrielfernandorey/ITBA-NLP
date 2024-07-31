try:
        # Obtener todos los documentos de un topico
        topic_docs_idx = [i for i, (_, topic) in enumerate(zip(list(df_news.index), topics)) if topic == topic_id]
        df_data = pd.DataFrame(np.array(topic_docs_idx).reshape(-1,1), columns=["idx"])

        # Similitud coseno entre el embedding del topico y los documentos del topico
        #s_coseno = []
        #for i in topic_docs_idx:
        #    s_coseno.append(cosine_similarity([topic_model.topic_embeddings_[topic_id + 1]], [docs_embedding[i]])[0][0])

        # Indices
        #idx_coseno_sort = np.argsort(s_coseno)[::-1]

        # Ordenado por mayor similitud
        #docs_per_topics_unsort = docs_per_topics[topic]

        # Entidades de documentos ordenados por similitud del topico elelgido
        #entities_topic = []
        #for doc in docs_per_topics_unsort[idx_coseno_sort]:
        #    entities_topic.append(entities_clean[doc][:n_entities])


        # Similitud coseno entre el embedding del topico y los documentos del topico
        s_coseno = []
        for i in topic_docs_idx:
            s_coseno.append(cosine_similarity([topic_model.topic_embeddings_[topic_id + 1]], [docs_embedding[i]])[0][0])
        
        df_data['umbral'] = s_coseno
        
        # Ordenado por mayor similitud
        df_filtered = df_data[df_data["umbral"] > threshold].sort_values("umbral", ascending=False)

        # Entidades de documentos ordenados por similitud del topico elelgido
        entities_topic = []
        for doc in list(df_filtered["idx"]):
            entities_topic.append(entities[doc][:n_entities])

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
        umbral_e = np.array(freq_e).mean() + 1

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
        
        if topic_entities_top == {}:
            topic_entities_top = sorted_frequencies[:3]

        return topic_entities_top
    except:
        return {}