from django.shortcuts import render
from .models import BxUsers, BxBooks, BxBookRatings
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import sklearn
from django.db.models import Q
import time
import warnings
from sklearn.decomposition import TruncatedSVD

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

def colaborativo(request, isbn_book):

    startTime2 = time.time()

    ratings_us_canada, libros = crear_matriz()

    matriz_ratings = ratings_us_canada.pivot(index = 'isbn', columns = 'user_id', 
        values = 'book_rating').fillna(0)

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
            nombre_titulo =libros.iloc[libros.index[libros['isbn']==isbn_ind]]['book_title'].tolist()[0]      
            info_libro = libros.iloc[libros.index[libros['isbn']==isbn_ind]]
            prueba.append((i, isbn_ind, nombre_titulo, info_libro['book_author'].tolist()[0], 
                info_libro['year_publication'].tolist()[0], info_libro['publisher'].tolist()[0]))

    rec_svd= sorted(prueba, key=lambda x:x[0], reverse=True)[1:10]
    
    k=10
    distancias, indices = modelo_knn.kneighbors(matriz_ratings.iloc[ind_libro_target, :].values.
        reshape(1, -1), n_neighbors = k+1)
    
    lista_knn = list(matriz_ratings.index[indices.flatten()[1:]])
    
    n=1
    for isbn_k in lista_knn:
        nombre_k =libros.iloc[libros.index[libros['isbn']==isbn_k]]['book_title'].tolist()[0]
        info_libro_k = libros.iloc[libros.index[libros['isbn']==isbn_k]]
        rec_knn.append((distancias.flatten()[n], isbn_k, nombre_k, info_libro_k['book_author'].tolist()[0], 
            info_libro_k['year_publication'].tolist()[0], info_libro_k['publisher'].tolist()[0]))
        n=n+1

    nombre_libro = libros.iloc[libros.index[libros['isbn']==isbn_book]]['book_title'].tolist()[0]
    print ('El script tomó {0} segundos'.format(time.time() - startTime2))

    return render(request, 'bx_library/colaborativo.html',{'book':nombre_libro, 'svd':rec_svd, 'knn':rec_knn})

    
def home(request):     

    if request.method == 'POST':
        if request.POST['titulo']:
            result = buscar(request, request.POST['titulo'])
            return render(request, 'bx_library/buscar.html',{'biblioteca':result})        
    else:
        return render(request, 'bx_library/home.html')
    
def buscar(request, query):

    startTime = time.time()  
    matriz, libros = crear_matriz()  
    
    query = query.lower()
    matriz['book_title'] = matriz['book_title'].str.lower()
    matriz['book_author'] = matriz['book_author'].str.lower()
    result_query = matriz[matriz['book_title'].str.contains(query) | matriz['book_author'].str.contains(query)]    
    unicos = result_query.isbn.unique()

    lista_libros =[]
    for i in unicos:
        lista_libros.append(list(libros[libros['isbn']==i].values))

    print ('Termina busqueda en {0} second !'.format(time.time() - startTime))

    return lista_libros

    


    