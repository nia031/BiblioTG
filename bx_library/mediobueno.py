from django.shortcuts import render
from .models import BxUsers, BxBooks, BxBookRatings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import sklearn
from fuzzywuzzy import fuzz
import time
import warnings
from sklearn.decomposition import TruncatedSVD

def colaborativo(request, query_book):

    startTime2 = time.time()

    users =pd.DataFrame.from_records(BxUsers.objects.all().values())
    books =pd.DataFrame.from_records(BxBooks.objects.all().values())
    ratings =pd.DataFrame.from_records(BxBookRatings.objects.all().values())

    combine_book_rating = pd.merge(ratings, books, right_on='isbn', left_on='isbn_id')
    columns= ['prediction','isbn_id','id','year_publication', 'publisher', 'book_author', 'imageurll', 'imageurlm', 'imageurls']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    combine_book_rating = combine_book_rating.dropna(axis=0, subset=['book_title'])
    book_ratingCount = (combine_book_rating.groupby(by = ['book_title'])['book_rating'].count().
        reset_index().rename(columns={'book_rating': 'totalRatingCount'})[['book_title','totalRatingCount']])

    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, 
        left_on = 'book_title', right_on = 'book_title', how = 'left')

    '''pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(book_ratingCount['totalRatingCount'].describe())
    print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))'''

    popularity_threshold = 50
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    
    combined = rating_popular_book.merge(users, left_on = 'user_id', right_on = 'user_id', how = 'left')

    us_canada_user_rating = combined[combined['location'].str.contains("usa|canada")]
    us_canada_user_rating=us_canada_user_rating.drop('age', axis=1)

    if not us_canada_user_rating[us_canada_user_rating.duplicated(['user_id', 'book_title'])].empty:
        initial_rows = us_canada_user_rating.shape[0]

        #print('Initial dataframe shape {0}'.format(us_canada_user_rating.shape))
        us_canada_user_rating = us_canada_user_rating.drop_duplicates(['user_id', 'book_title'])
        current_rows = us_canada_user_rating.shape[0]
        #print('New dataframe shape {0}'.format(us_canada_user_rating.shape))
        print('Removed {0} rows'.format(initial_rows - current_rows))
        
    us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'book_title', columns = 'user_id', 
        values = 'book_rating').fillna(0)

###################### Técnica Matriz de Factorizacion SVD #################################
    #X = us_canada_user_rating_pivot2.values.T
    X = us_canada_user_rating_pivot.values
    SVD = TruncatedSVD(n_components=12, random_state=17)
    matrix = SVD.fit_transform(X)     
    warnings.filterwarnings("ignore",category =RuntimeWarning)
    corr = np.corrcoef(matrix) 
    us_canada_book_title = us_canada_user_rating_pivot.index
    us_canada_book_list = list(us_canada_book_title)
    rec_svd=[]

###################### Técnica vecinos cercanos #################################
    us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(us_canada_user_rating_matrix)
    rec_knn = []
    
    k=10
    book_name = None
    ratio_tuples = []
    m=0
    for name in us_canada_user_rating_pivot.index:
        ratio = fuzz.ratio(name.lower(), query_book.lower())
        isbn = books.iloc[books.index[books['book_title']==name]]['isbn'].tolist()[0]
        m=m+1        
        if ratio >= 70:
            current_query_index = us_canada_user_rating_pivot.index.tolist().index(name)
            ratio_tuples.append((isbn, name, ratio, current_query_index))
    
    print('Possible matches: {0}:\n'.format([(x[0], x[1], x[2]) for x in ratio_tuples]))
    
    try:
        book_name = max(ratio_tuples, key = lambda x: x[2])[1] # get the name of the best artist match in the data
        book_index = max(ratio_tuples, key = lambda x: x[2])[3] # get the index of the best artist match in the data
    except:
        book_name = 'No se encontraron coincidencias'
        print('Your artist didn\'t match any artists in the data. Try again')
        return book_name, rec_svd, rec_knn

    
    target_book_index = us_canada_book_list.index(book_name)
    corr_target_book_list  = list(corr[target_book_index]) #calificaciones de libros respecto a target
    
    max_indices=[]
    prueba=[]
    for i in corr_target_book_list:
        if (i<1.0) & (i>0.8):
            max_ind= corr_target_book_list.index(i)
            max_indices.append(max_ind)
            title_name = us_canada_book_title[max_ind]
            isbn_ind =books.iloc[books.index[books['book_title']==title_name]]['isbn'].tolist()[0]      
            all_book = books.iloc[books.index[books['isbn']==isbn_ind]]
            prueba.append((i, isbn_ind, title_name, all_book['book_author'].tolist()[0], 
                all_book['year_publication'].tolist()[0], all_book['publisher'].tolist()[0]))

    rec_svd= sorted(prueba, key=lambda x:x[0], reverse=True)[1:10]
    #print(rec_svd)
    
    distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[book_index, :].values.
        reshape(1, -1), n_neighbors = k+1)
    
    list_knn = list(us_canada_user_rating_pivot.index[indices.flatten()[1:]])

    
    n=1
    for i in list_knn:
        isbn_ind_k =books.iloc[books.index[books['book_title']==i]]['isbn'].tolist()[0]
        all_book_k = books.iloc[books.index[books['isbn']==isbn_ind_k]]
        rec_knn.append((distances.flatten()[n], isbn_ind_k, i, all_book['book_author'].tolist()[0], 
            all_book['year_publication'].tolist()[0], all_book['publisher'].tolist()[0]))
        n=n+1

    #print(rec_knn)

    print ('The script took {0} second !'.format(time.time() - startTime2))

    return book_name,rec_svd, rec_knn

    
def home(request):
    #titles=BxBooks.objects.filter()[:50]
    #return render(request, 'bx_library/home.html',{'biblioteca':titles})

    titles=BxBooks.objects.filter()[:50]

    if request.method == 'POST':
        if request.POST['titulo']:
            book_name, rec_svd, rec_knn = colaborativo(request, request.POST['titulo'])
            return render(request, 'bx_library/home.html',{'book':book_name, 'svd':rec_svd, 'knn': rec_knn})        
    else:
        return render(request, 'bx_library/home.html',{'biblioteca':titles})
    
    