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
	users =pd.DataFrame.from_records(BxUsers.objects.all().values())
	books =pd.DataFrame.from_records(BxBooks.objects.all().values())
	ratings =pd.DataFrame.from_records(BxBookRatings.objects.all().values())

	combine_book_rating = pd.merge(ratings, books, right_on='isbn', left_on='isbn_id')
	columns= ['prediction','isbn_id','id', 'imageurll', 'imageurlm', 'imageurls']
	combine_book_rating = combine_book_rating.drop(columns, axis=1)

	combine_book_rating = combine_book_rating.dropna(axis=0, subset=['isbn'])
	book_ratingCount = (combine_book_rating.groupby(by = ['isbn'])['book_rating'].count().
	    reset_index().rename(columns={'book_rating': 'totalRatingCount'})[['isbn','totalRatingCount']])

	rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, 
	    left_on = 'isbn', right_on = 'isbn', how = 'left')

	popularity_threshold = 50
	rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

	combined = rating_popular_book.merge(users, left_on = 'user_id', right_on = 'user_id', how = 'left')

	us_canada_user_rating = combined[combined['location'].str.contains("usa|canada")]
	us_canada_user_rating=us_canada_user_rating.drop('age', axis=1)    

	if not us_canada_user_rating[us_canada_user_rating.duplicated(['user_id', 'isbn'])].empty:        
	    initial_rows = us_canada_user_rating.shape[0]
	    us_canada_user_rating = us_canada_user_rating.drop_duplicates(['user_id', 'isbn'])
	    current_rows = us_canada_user_rating.shape[0]
	    print('Removed {0} rows'.format(initial_rows - current_rows))

	return us_canada_user_rating, books

def inicio():
	startTime = time.time()

	us_canada_user_rating, books = crear_matriz()

	seleccion_random = us_canada_user_rating.sample(n=10)

	matriz_col = []
	for i in seleccion_random.index:
		print(seleccion_random['isbn'][i])		
		matriz_col.append(colaborativo(seleccion_random['isbn'][i]))	
	
								
	'''sum_k =[0]*12
	sum_s =[0]*12

	for i in range(0,len(matriz_col)):
		for j in range(0,len(matriz_col)):							
			sum_k[i] = float(((matriz_col[j])[2])[i]) + sum_k[i]
			sum_s[i] = float(((matriz_col[j])[1])[i]) + sum_s[i]
		
		sum_k[i] = sum_k[i]/10
		sum_s[i] = sum_s[i]/10
			
	print(sum_k)
	print(sum_s)'''
			
	print ('The script inicio took {0} second !'.format(time.time() - startTime))


def colaborativo(query_book):

	startTime2 = time.time()
	us_canada_user_rating, books = crear_matriz()

	us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'isbn', columns = 'user_id', values = 'book_rating').fillna(0)

	###################### Técnica Matriz de Factorizacion SVD #################################
	#X = us_canada_user_rating_pivot2.values.T
	X = us_canada_user_rating_pivot.values
	SVD = TruncatedSVD(n_components=12, random_state=17)
	matrix = SVD.fit_transform(X)     
	warnings.filterwarnings("ignore",category =RuntimeWarning)
	corr = np.corrcoef(matrix) 
	us_canada_book_isbn = us_canada_user_rating_pivot.index
	us_canada_book_list = list(us_canada_book_isbn)
	rec_svd=[]

	###################### Técnica vecinos cercanos #################################
	us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)
	model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
	model_knn.fit(us_canada_user_rating_matrix)
	rec_knn = []

	target_book_index = us_canada_book_list.index(query_book)
	corr_target_book_list  = list(corr[target_book_index]) #calificaciones de libros respecto a target

	max_indices=[]
	prueba=[]
	for i in corr_target_book_list:
	    if (i<1.0) & (i>0.8):
	        max_ind= corr_target_book_list.index(i)
	        max_indices.append(max_ind)
	        isbn_ind = us_canada_book_isbn[max_ind]
	        #title_name =books.iloc[books.index[books['isbn']==isbn_ind]]['book_title'].tolist()[0]      
	        #all_book = books.iloc[books.index[books['isbn']==isbn_ind]]
	        prueba.append(i)#(, isbn_ind))#, title_name, all_book['book_author'].tolist()[0], 
	            #all_book['year_publication'].tolist()[0], all_book['publisher'].tolist()[0]))

	rec_svd= sorted(prueba, key=lambda x:x, reverse=True)[1:11]

	k=10

	distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[target_book_index, :].values.
	    reshape(1, -1), n_neighbors = k+1)

	distances =distances.flatten()
	distances=sorted(distances, key=lambda x:x, reverse=True)

	list_knn = list(us_canada_user_rating_pivot.index[indices.flatten()[1:]])

	n=1
	for isbn_k in list_knn:
	    #name_k =books.iloc[books.index[books['isbn']==isbn_k]]['book_title'].tolist()[0]
	    #all_book_k = books.iloc[books.index[books['isbn']==isbn_k]]
	    rec_knn.append(distances[n])#(, isbn_k))#, name_k, all_book['book_author'].tolist()[0], 
	        #all_book['year_publication'].tolist()[0], all_book['publisher'].tolist()[0]))
	    n=n+1

	book_name = books.iloc[books.index[books['isbn']==query_book]]['book_title'].tolist()[0]

	print ('The script colaborativo took {0} second !'.format(time.time() - startTime2))

	#print(rec_svd)
	#print(rec_knn)

	with open("/home/stefania/Web/TG-Dev/backup/bx_library/knn.csv", 'a', newline='') as lista_knn:
		wr = csv.writer(lista_knn, quoting=csv.QUOTE_ALL)
		wr.writerow(rec_knn)

	with open("/home/stefania/Web/TG-Dev/backup/bx_library/svd.csv", 'a', newline='') as lista_svd:
		wr = csv.writer(lista_svd, quoting=csv.QUOTE_ALL)
		wr.writerow(rec_svd)
	

	return query_book, rec_svd, rec_knn

inicio()