from django.test import TestCase
import csv
from biblioteca.models import Transacciones
import time
import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

def crearmatriz():
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
	datos_libros = pd.read_sql(query, conn)
    
    datos_libros['note'] = datos_libros['note'].fillna('')
    datos_libros['description'] = datos_libros['subject'] + " "+ datos_libros['note']
    datos_libros['description'] = datos_libros['description'].fillna('')


	return datos_libros

def inicio():
	startTime = time.time()

	datos_libros = crearmatriz()

	random = datos_libros.sample(n=5)
	print(random)

	matriz_rec = []

	for i in random.index:
		print(random['titleno'][i])
		matriz_rec.append(rec_contenido(random['titleno'][i]))	
			
	print ('The script inicio took {0} second !'.format(time.time() - startTime)) 

def rec_contenido(title):
	startTime = time.time()
	datos_libros = crearmatriz()

	tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english',norm='l2') 
    
    matriz_tfidf = tfidf.fit_transform(datos_libros['description'])

	linearkernel = linear_kernel(matriz_tfidf, matriz_tfidf)  
    similaridad_coseno = cosine_similarity(matriz_tfidf,matriz_tfidf)      
	
	datos = datos_libros.reset_index()
    indices = pd.Series(datos.index, index=datos['titleno']) 
	ind_libro = indices[title]  

	libros_k={}          
    k_scores = list(enumerate(linearkernel[ind_libro]))
    k_scores = sorted(k_scores, key=lambda x: x[1], reverse=True)
    k_scores = k_scores[1:11]
    k_indices = [i[0] for i in k_scores]
	score_k= [str(i[1]) for i in k_scores] 

	libros_s={}
    sim_scores = list(enumerate(similaridad_coseno[ind_libro]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    sim_indices = [i[0] for i in sim_scores]  
	score_s = [str(i[1]) for i in sim_scores] 

	print ('El rec_contenido tomó {0} segundos'.format(time.time() - startTime))    

	with open("/home/stefania/Web/TG-Dev/backup/biblioteca/kernel.csv", 'a', newline='') as lista_k:
		wr = csv.writer(lista_k, quoting=csv.QUOTE_ALL)
		wr.writerow(score_k)

	with open("/home/stefania/Web/TG-Dev/backup/biblioteca/cosine.csv", 'a', newline='') as lista_s:
		wr = csv.writer(lista_s, quoting=csv.QUOTE_ALL)
		wr.writerow(score_s)

	return title, score_k,score_s
    
inicio()