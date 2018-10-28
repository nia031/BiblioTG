from django.test import TestCase
import csv
import time
import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

def inicio():
	startTime = time.time()
	conn = psycopg2.connect(host="localhost", dbname="bibliocarvajaldb", user="postgres", password="1234", port="5433")

	query_users= """ 
		select * from 
		(select distinct(usuario), titleno from transacciones) table_alias 
		ORDER BY random() limit 10;
	"""
	data_users = pd.read_sql(query_users, conn)
	data_users.columns=['user', 'titleno']

	matriz_rec = []

	for i in data_users.index:
		print(data_users['titleno'][i])
		matriz_rec.append(rec_contenido(data_users['titleno'][i]))
	

	'''sum_k =[0]*10
	sum_s =[0]*10

	for i in range(0,len(matriz_rec)):
		for j in range(0,len(matriz_rec)):
			sum_k[i] = float(((matriz_rec[j])[1])[i]) + sum_k[i]
			sum_s[i] = float(((matriz_rec[j])[2])[i]) + sum_s[i]
		
		sum_k[i] = sum_k[i]/10
		sum_s[i] = sum_s[i]/10
			
	print(sum_k)
	print(sum_s)'''	
			
	print ('The script inicio took {0} second !'.format(time.time() - startTime)) 

def rec_contenido(title):
	conn = psycopg2.connect(host="localhost", dbname="bibliocarvajaldb", user="postgres", password="1234", port="5433")

	query = """
	  SELECT A.titleno, A.title, A.subject, A.note
	    FROM(  Select titles.titleno,title,subject,note,
	            ROW_NUMBER() OVER(PARTITION BY titles.titleno ORDER BY title_mat.id_subject) AS RN
	            From titles 
	            left join title_mat on titles.titleno=title_mat.titleno
	            join notas on notas.titleno=titles.titleno
	            join materias on materias.id_subject=title_mat.id_subject
	            ) A
	    WHERE A.RN = 1;
	"""

	startTime = time.time()
	data = pd.read_sql(query, conn)

	data['note'] = data['note'].fillna('')
	data['description'] = data['subject'] + " "+ data['note']
	data['description'] = data['description'].fillna('')

	tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english',norm='l2') 
	#transforms text to feature vectors that can be used as input to estimator.
	######ngram (1,3) can be explained as follows#####
	#ngram(1,3) encompasses uni gram, bi gram and tri gram    
	tfidf_matrix = tf.fit_transform(data['description'])

	lin_kernel = linear_kernel(tfidf_matrix, tfidf_matrix)  
	cosine_similarities = cosine_similarity(tfidf_matrix,tfidf_matrix)      
	
	datos = data.reset_index()
	indices = pd.Series(datos.index, index=datos['titleno'])
	idx = indices[title]

	#libro = data.iloc[idx]
	#libro_target =libro.title

	libros_k={} 
	score_k=[]         
	k_scores = list(enumerate(lin_kernel[idx]))
	k_scores = sorted(k_scores, key=lambda x: x[1], reverse=True)
	k_scores = k_scores[1:11]
	k_indices = [i[0] for i in k_scores]
	#libros_k=data.loc[k_indices]['titleno']
	#libros_k['score'] = [str(i[1]) for i in k_scores]   
	#libros_k=libros_k.reset_index().T.to_dict('list')
	score_k= [str(i[1]) for i in k_scores] 

	libros_s={}
	score_s=[]
	sim_scores = list(enumerate(cosine_similarities[idx]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	sim_scores = sim_scores[1:11]
	sim_indices = [i[0] for i in sim_scores]
	#libros_s=data.iloc[sim_indices]['titleno']
	#libros_s['score'] = [str(i[1]) for i in sim_scores]        
	#libros_s=libros_s.reset_index().T.to_dict('list')   
	score_s = [str(i[1]) for i in sim_scores] 

	print ('rec_contenid took {0} second !'.format(time.time() - startTime)) 

	with open("/home/stefania/Web/TG-Dev/backup/biblioteca/kernel.csv", 'a', newline='') as lista_k:
		wr = csv.writer(lista_k, quoting=csv.QUOTE_ALL)
		wr.writerow(score_k)

	with open("/home/stefania/Web/TG-Dev/backup/biblioteca/cosine.csv", 'a', newline='') as lista_s:
		wr = csv.writer(lista_s, quoting=csv.QUOTE_ALL)
		wr.writerow(score_s)

	return title, score_k,score_s

    
inicio()