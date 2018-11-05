from django.test import TestCase
from bx_library.models import BxUsers, BxBooks, BxBookRatings
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import sklearn
import time
import warnings
import csv
from sklearn.decomposition import TruncatedSVD

	# Create your tests here.

def crear_matriz():
	usuarios =pd.DataFrame.from_records(BxUsers.objects.all().values())
    libros =pd.DataFrame.from_records(BxBooks.objects.all().values())
    ratings =pd.DataFrame.from_records(BxBookRatings.objects.all().values())

    libros_ratings = pd.merge(ratings, libros, right_on='isbn', left_on='isbn_id')
    columns= ['prediction','isbn_id','id', 'imageurll', 'imageurlm', 'imageurls']
    libros_ratings = libros_ratings.drop(columns, axis=1)
    
    libros_ratings = libros_ratings.dropna(axis=0, subset=['isbn'])
    cuenta_rating_libro = (libros_ratings.groupby(by = ['isbn'])['book_rating'].count().
        reset_index().rename(columns={'book_rating': 'cuentaTotalRatings'})[['isbn','cuentaTotalRatings']])
    
    totales_ratings = libros_ratings.merge(cuenta_rating_libro, 
        left_on = 'isbn', right_on = 'isbn', how = 'left')
    
    ratings_minimo = 50
    libros_mas_populares = totales_ratings.query('cuentaTotalRatings >= @ratings_minimo')
    
    populares_usuarios = libros_mas_populares.merge(usuarios, left_on = 'user_id', right_on = 'user_id', how = 'left')

    ratings_us_canada = populares_usuarios[populares_usuarios['location'].str.contains("usa|canada")]
    ratings_us_canada=ratings_us_canada.drop('age', axis=1)    

    if not ratings_us_canada[ratings_us_canada.duplicated(['user_id', 'isbn'])].empty:        
        filas_iniciales = ratings_us_canada.shape[0]
        ratings_us_canada = ratings_us_canada.drop_duplicates(['user_id', 'isbn'])
        filas_actuales = ratings_us_canada.shape[0]
        print('Removed {0} rows'.format(filas_iniciales - filas_actuales))

    return ratings_us_canada, libros

def inicio():
	startTime = time.time()

	ratings_us_canada, libros = crear_matriz()

	seleccion_random = ratings_us_canada.sample(n=10)

	matriz_col = []
	for i in seleccion_random.index:
		print(seleccion_random['isbn'][i])		
		matriz_col.append(colaborativo(seleccion_random['isbn'][i]))	
	
			
	print ('El script inicio tomó {0} segundos!'.format(time.time() - startTime))


def colaborativo(query_book):

	startTime2 = time.time()
	ratings_us_canada, libros = crear_matriz()

	matriz_ratings = ratings_us_canada.pivot(index = 'isbn', columns = 'user_id', values = 'book_rating').fillna(0)
	###################### Técnica Matriz de Factorizacion SVD #################################

	X = matriz_ratings.values
    SVD = TruncatedSVD(n_components=12, random_state=17)
    matriz_svd = SVD.fit_transform(X)     
    warnings.filterwarnings("ignore",category =RuntimeWarning)
    matriz_corr = np.corrcoef(matriz_svd) 
    isbn_matriz_libros = matriz_ratings.index
    lista_isbn_libros = list(isbn_matriz_libros)
    rec_svd=[]

	###################### Técnica vecinos cercanos #################################
	matriz_ratings_comprimida = csr_matrix(matriz_ratings.values)
    modelol_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    modelo_knn.fit(matriz_ratings_comprimida)
    rec_knn = []
    
    ind_libro_target = lista_isbn_libros.index(isbn_book)
    lista_libros_corr_target = list(matriz_corr[ind_libro_target]) #calificaciones de libros respecto a target

	max_indices=[]
	prueba=[]
	for i in lista_libros_corr_target:
	    if (i<1.0) & (i>0.8):
	        max_ind= lista_libros_corr_target.index(i)
	        max_indices.append(max_ind)
	        isbn_ind = isbn_matriz_libros[max_ind]
	        prueba.append(i)

	rec_svd= sorted(prueba, key=lambda x:x, reverse=True)[1:11]

	k=10

	distancias, indices = modelo_knn.kneighbors(matriz_ratings.iloc[ind_libro_target, :].values.
        reshape(1, -1), n_neighbors = k+1)
    
    distancias =distancias.flatten()
	distancias=sorted(distancias, key=lambda x:x, reverse=True)
    lista_knn = list(matriz_ratings.index[indices.flatten()[1:]])

	n=1
	for isbn_k in lista_knn:
	    rec_knn.append(distancias[n])
	    n=n+1

	nombre_libro = libros.iloc[libros.index[libros['isbn']==isbn_book]]['book_title'].tolist()[0]

	print ('El script tomó {0} segundos'.format(time.time() - startTime2))

	with open("/home/stefania/Web/TG-Dev/backup/bx_library/knn.csv", 'a', newline='') as lista_knn:
		wr = csv.writer(lista_knn, quoting=csv.QUOTE_ALL)
		wr.writerow(rec_knn)

	with open("/home/stefania/Web/TG-Dev/backup/bx_library/svd.csv", 'a', newline='') as lista_svd:
		wr = csv.writer(lista_svd, quoting=csv.QUOTE_ALL)
		wr.writerow(rec_svd)
	

	return nombre_libro, rec_svd, rec_knn

inicio()