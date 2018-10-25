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


# Create your views here.
def slopenone(request):
	#print(BxUsers.objects.filter()[:10])

    users = {}
    books = []

    for user in BxUsers.objects.filter()[:100]:#all():
        ratingsUser ={}
        for rating in user.BxBookRatings.filter(book_rating=0, prediction=0)[:100]:
            ratingsUser[str(rating.isbn)]=rating.book_rating
    
        users[str(user.user_id)]=ratingsUser

    for book in BxBooks.objects.all():
        books.append(str(book.isbn))

    print(users)
    print(books)

	
    def averageDeviations():
	    deviations = {}
	    
	    num = len(books)
	    for i in range(num):
	        for j in range(i + 1, num):
	            item1 = books[i]
	            item2 = books[j]

	            r = 0.0
	            n = 0
	            for key in users.keys():
	                user = users[key]
	                if item1 in user and item2 in user:
	                    r += user[item2] - user[item1]
	                    n += 1

	            if n > 0:
	                r /= float(n)

	            deviations[(item1, item2)] = (r, n)
	            deviations[(item2, item1)] = (-r, n)


    def predict(userRatings, item):
	    if item in userRatings:
	        return userRatings[item]

	    if len(userRatings) == 0:
	        return 0

	    r1 = 0.0
	    r2 = 0.0
	    for key in userRatings.keys():
	        dev, n = deviations[(key, item)]
	        r1 += (dev + userRatings[key]) * n
	        r2 += n

	    try:
	        return r1 / r2
	    except:
	        return 0

    def recommends(userRatings):
	    averageDeviations()

	    booksNotRated = [book for book in books if book not in userRatings]

	    result = []
	    for booksNotRated in booksNotRated:
	        suggest = predict(userRatings,booksNotRated)
	        result.append((booksNotRated, suggest))

	    return result

    def execute():
        saves = []
        for userKey in users.keys():
    	    suggests = recommends(users[userKey])
    	    user = User.objects.get(user_id=userKey)
    	    
    	    # for suggest in suggests:
    	    #     prediction = suggest[1]
    	    #     if(prediction > 0):
    	    #         book = BxBooks.objects.get(isbn=suggest[0])
    	    #         try:
    	    #             rating = BxBookRatings.objects.get(user_id=user, isbn=book, book_rating=0)
    	    #         except BxBookRatings.DoesNotExist:
    	    #             BxBookRatings.objects.create(user_id=user, isbn=book, book_rating=0, prediction=prediction)
    	    #             continue

    	    #         BxBookRatings.prediction = prediction 
    	    #         BxBookRatings.save()

        print(suggests)

    execute()
    	
    return render(request, 'bx_library/home.html')

def correlation(request):
    
    users =pd.DataFrame.from_records(BxUsers.objects.all().values())

    books =pd.DataFrame.from_records(BxBooks.objects.all().values())

    ratings =pd.DataFrame.from_records(BxBookRatings.objects.all().values())
 

    rating_count = pd.DataFrame(ratings.groupby('isbn_id')['book_rating'].count())
    rating_count.sort_values('book_rating',ascending=False).head() #Los mas calificados

    average_rating  = pd.DataFrame(ratings.groupby('isbn_id')['book_rating'].mean())
    average_rating['ratingCount'] =rating_count
    average_rating.sort_values('ratingCount', ascending=False).head() #Calificacion de acuerdo a ratingcount

    counts_u = ratings['user_id'].value_counts()
    ratings = ratings[ratings['user_id'].isin(counts_u[counts_u>=200].index)]
    counts_b = ratings['book_rating'].value_counts()
    ratings = ratings[ratings['book_rating'].isin(counts_b[counts_b>=100].index)]

    ratings_pivot = ratings.pivot(index='user_id', columns='isbn_id').book_rating
    userID = ratings_pivot.index
    ISBN = ratings_pivot.columns
    print(ratings_pivot.shape)
    #ratings_pivot.head()

    bones_ratings = ratings_pivot['0316666343']
    similar_to_bones = ratings_pivot.corrwith(bones_ratings)
    corr_bones = pd.DataFrame(similar_to_bones, columns=['pearsonR'])
    corr_bones.dropna(inplace=True)
    corr_summary = corr_bones.join(average_rating['ratingCount'])
    corr_summary[corr_summary['ratingCount']>=300].sort_values('pearsonR', ascending=False).head(10)

    books_corr_to_bones = pd.DataFrame(['0312291639', '0316601950', '0446610038', '0446672211', '0385265700', '0345342968', '0060930535', '0375707972', '0684872153'], 
                                  index=np.arange(9), columns=['isbn_id'])
    corr_books = pd.merge(books_corr_to_bones, books, on='isbn_id')
    print(corr_books)

def colaborativo_vecinos(request, query_book):

    startTime = time.time()

    users =pd.DataFrame.from_records(BxUsers.objects.all().values())
    #print(users.columns)
    books =pd.DataFrame.from_records(BxBooks.objects.all().values())
    ratings =pd.DataFrame.from_records(BxBookRatings.objects.all().values())

    combine_book_rating = pd.merge(ratings, books, right_on='isbn', left_on='isbn_id')
    columns= ['prediction','isbn_id','id','year_publication', 'publisher', 'book_author', 'imageurll', 'imageurlm', 'imageurls']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    #print(combine_book_rating.head())

    combine_book_rating = combine_book_rating.dropna(axis=0, subset=['book_title'])
    book_ratingCount = (combine_book_rating.groupby(by = ['book_title'])['book_rating'].count().
        reset_index().rename(columns={'book_rating': 'totalRatingCount'})[['book_title','totalRatingCount']])
    #print(book_ratingCount.head())

    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, 
        left_on = 'book_title', right_on = 'book_title', how = 'left')
    #print(rating_with_totalRatingCount.head())

    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(book_ratingCount['totalRatingCount'].describe())
    print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

    popularity_threshold = 50
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    print(rating_popular_book.head())

    combined = rating_popular_book.merge(users, left_on = 'user_id', right_on = 'user_id', how = 'left')

    us_canada_user_rating = combined[combined['location'].str.contains("usa|canada")]
    us_canada_user_rating=us_canada_user_rating.drop('age', axis=1)
    #print(us_canada_user_rating.head())

    if not us_canada_user_rating[us_canada_user_rating.duplicated(['user_id', 'book_title'])].empty:
        initial_rows = us_canada_user_rating.shape[0]

        print('Initial dataframe shape {0}'.format(us_canada_user_rating.shape))
        us_canada_user_rating = us_canada_user_rating.drop_duplicates(['user_id', 'book_title'])
        current_rows = us_canada_user_rating.shape[0]
        print('New dataframe shape {0}'.format(us_canada_user_rating.shape))
        print('Removed {0} rows'.format(initial_rows - current_rows))

    us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'book_title', columns = 'user_id', 
        values = 'book_rating').fillna(0)
    us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(us_canada_user_rating_matrix)

    #query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
    #query_artist ='smoke jumper'
    k=10
    query_index = None
    ratio_tuples = []
    
    for i in us_canada_user_rating_pivot.index:
        ratio = fuzz.ratio(i.lower(), query_book.lower())
        if ratio >= 75:
            current_query_index = us_canada_user_rating_pivot.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
    
    print('Possible matches: {0}:\n'.format([(x[0], x[1]) for x in ratio_tuples]))
    
    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[2] # get the index of the best artist match in the data
    except:
        print('Your artist didn\'t match any artists in the data. Try again')
        return None

    distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.
        reshape(1, -1), n_neighbors = k+1)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.
                flatten()[i]], distances.flatten()[i]))

    print ('The script took {0} second !'.format(time.time() - startTime))

def colaborativo_SVD(request, query_book):

    startTime2 = time.time()

    users =pd.DataFrame.from_records(BxUsers.objects.all().values())
    #print(users.columns)
    books =pd.DataFrame.from_records(BxBooks.objects.all().values())
    ratings =pd.DataFrame.from_records(BxBookRatings.objects.all().values())

    combine_book_rating = pd.merge(ratings, books, right_on='isbn', left_on='isbn_id')
    columns= ['prediction','isbn_id','id','year_publication', 'publisher', 'book_author', 'imageurll', 'imageurlm', 'imageurls']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    #print(combine_book_rating.head())

    combine_book_rating = combine_book_rating.dropna(axis=0, subset=['book_title'])
    book_ratingCount = (combine_book_rating.groupby(by = ['book_title'])['book_rating'].count().
        reset_index().rename(columns={'book_rating': 'totalRatingCount'})[['book_title','totalRatingCount']])
    #print(book_ratingCount.head())

    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, 
        left_on = 'book_title', right_on = 'book_title', how = 'left')
    #print(rating_with_totalRatingCount.head())

    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(book_ratingCount['totalRatingCount'].describe())
    print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

    popularity_threshold = 50
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    print(rating_popular_book.head())

    combined = rating_popular_book.merge(users, left_on = 'user_id', right_on = 'user_id', how = 'left')

    us_canada_user_rating = combined[combined['location'].str.contains("usa|canada")]
    us_canada_user_rating=us_canada_user_rating.drop('age', axis=1)
    #print(us_canada_user_rating.head())

    if not us_canada_user_rating[us_canada_user_rating.duplicated(['user_id', 'book_title'])].empty:
        initial_rows = us_canada_user_rating.shape[0]

        print('Initial dataframe shape {0}'.format(us_canada_user_rating.shape))
        us_canada_user_rating = us_canada_user_rating.drop_duplicates(['user_id', 'book_title'])
        current_rows = us_canada_user_rating.shape[0]
        print('New dataframe shape {0}'.format(us_canada_user_rating.shape))
        print('Removed {0} rows'.format(initial_rows - current_rows))
    
    us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'book_title', columns = 'user_id', 
        values = 'book_rating').fillna(0)
    us_canada_user_rating_pivot2 = us_canada_user_rating.pivot(index = 'user_id', columns = 'book_title', 
        values = 'book_rating').fillna(0)
    X = us_canada_user_rating_pivot2.values.T
    SVD = TruncatedSVD(n_components=12, random_state=17)
    matrix = SVD.fit_transform(X) 
    
    warnings.filterwarnings("ignore",category =RuntimeWarning)
    corr = np.corrcoef(matrix)
    
    us_canada_book_title = us_canada_user_rating_pivot2.columns
    us_canada_book_list = list(us_canada_book_title)

    query_index = None
    ratio_tuples = []
    
    
    for i in us_canada_user_rating_pivot.index:
        ratio = fuzz.ratio(i.lower(), query_book.lower())
        if ratio >= 75:
            current_query_index = us_canada_user_rating_pivot.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
    print(ratio_tuples)
    
    print('Possible matches: {0}:\n'.format([(x[0], x[1]) for x in ratio_tuples]))
    
    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[0] # get the index of the best artist match in the data
    except:
        print('Your artist didn\'t match any artists in the data. Try again')
        return None

    print(query_index)
    
    target_book = us_canada_book_list.index(query_index)
    print(target_book)
    
    corr_target_book  = corr[target_book]
    rec = (list(us_canada_book_title[(corr_target_book<1.0) & (corr_target_book>0.8)]))
    
    
    print ('The script took {0} second !'.format(time.time() - startTime2))

    return rec

	
def home(request):
    '''titles=BxBooks.objects.filter()[:100]
    
    username = None
    if request.user.is_authenticated:
        username = request.user.get_username()'''
    if request.method == 'POST':
        if request.POST['titulo']:
            #rec = colaborativo_SVD(request, request.POST['titulo'])
            rec = ['A Light in the Window (The Mitford Years)', 'A Long Fatal Love Chase', 'A Maidens Grave', 'A New Song (Mitford Years (Paperback))', 'A Patchwork Planet', 'After All These Years', 'At Home in Mitford (The Mitford Years)', 'Back When We Were Grownups : A Novel (Ballantine Readers Circle)', 'Bad Love (Alex Delaware Novels (Paperback))', 'Big Stone Gap', 'Birds of Prey: A Novel of Suspense', 'Black and Blue', 'Blood Work', 'Cane River (Oprahs Book Club (Paperback))', 'Colony', 'Coming Home', 'Dangerous to Know', 'Dr. Atkins New Diet Revolution', 'Dust to Dust', 'East of Eden (Oprahs Book Club)', 'Evening Class', 'Fortunes Rocks: A Novel', 'Gap Creek: The Story Of A Marriage', 'Gone For Good', 'Good in Bed', 'Here on Earth (Oprahs Book Club)', 'I Dont Know How She Does It: The Life of Kate Reddy, Working Mother', 'Icy Sparks', 'In Pursuit of the Proper Sinner', 'Ladder of Years', 'Light a Penny Candle', 'London Transports', 'Manhattan Hunt Club', 'Memoirs of a Geisha', 'Mother of Pearl (Oprahs Book Club (Paperback))', 'Out to Canaan (The Mitford Years)', 'Outer Banks', 'Personal Injuries', 'Plantation: A Lowcountry Tale', 'Portrait of a Killer: Jack the Ripper__ Case Closed (Berkley True Crime)', 'Promises', 'Quentins', 'Ralphs Party', 'Revenge of the Middle_Aged Woman', 'Roses Are Red', 'SEAT OF THE SOUL', 'Salem Falls', 'September', 'Serpents Tooth : A Peter Decker/Rina Lazarus Novel (Peter Decker &amp; Rina Lazarus Novels (Paperback))', 'Spencerville', 'Sticks &amp; Scones', 'Still Waters', 'Suzannes Diary for Nicholas', 'Switcheroo : A Novel', 'Tell Me Lies (Tell Me Lies)', 'The Art of Happiness: A Handbook for Living', 'The Associate', 'The Burning Man', 'The Coffin Dancer (Lincoln Rhyme Novels (Paperback))', 'The Copper Beech', 'The Last Time They Met : A Novel', 'The Lilac Bus: Stories', 'The Mermaids Singing', 'The Pilots Wife : A Novel Tag: Author of the Weight of Water (Oprahs Book Club (Hardcover))', 'The Return Journey', 'The Shelters of Stone (Earths Children, Book 5)', 'The Valley of Horses', 'The Weight of Water', 'The Weight of Water : A Novel Tag _ Author of Resistance and Strange Fits of Passion', 'These High, Green Hills (The Mitford Years)', 'ThursdayS At Eight', 'Watermelon', 'Where You Belong', 'Where or When  : A Novel', 'Winter Solstice']

            return render(request, 'bx_library/home.html',{'biblioteca':rec})
    else:
        #return render(request, 'bx_library/home.html',{'biblioteca':titles,'user':username})
        return render(request, 'bx_library/home.html')









 us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'isbn', columns = 'user_id', 
        values = 'book_title').fillna(0)
    us_canada_user_rating_pivot2 = us_canada_user_rating.pivot(index = 'user_id', columns = 'isbn', 
        values = 'book_rating').fillna(0)
    X = us_canada_user_rating_pivot2.values.T
    SVD = TruncatedSVD(n_components=12, random_state=17)
    matrix = SVD.fit_transform(X) 
    
    warnings.filterwarnings("ignore",category =RuntimeWarning)
    corr = np.corrcoef(matrix)
    
    us_canada_book_title = us_canada_user_rating_pivot2.columns
    us_canada_book_list = list(us_canada_book_title)

    query_index = None
    ratio_tuples = []
    
    for i in us_canada_user_rating_pivot.index:
        #libro = BxBooks.objects.filter(isbn=i)
        ratio = fuzz.ratio(i.lower(), query_book.lower())
        if ratio >= 75:
            current_query_index = us_canada_user_rating_pivot.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
    
    print(ratio_tuples)
    
    print('Possible matches: {0}:\n'.format([(x[0], x[1]) for x in ratio_tuples]))
    
    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[0] # get the index of the best artist match in the data
    except:
        print('Your artist didn\'t match any artists in the data. Try again')
        return None

    print(query_index)
    
    target_book = us_canada_book_list.index(query_index)
    print(target_book)
    
    corr_target_book  = corr[target_book]
    rec = (list(us_canada_book_title[(corr_target_book<1.0) & (corr_target_book>0.8)]))
    
    
    print ('The script took {0} second !'.format(time.time() - startTime2))

    return rec

    
def home(request):
    '''titles=BxBooks.objects.filter()[:100]
    
    username = None
    if request.user.is_authenticated:
        username = request.user.get_username()'''
    if request.method == 'POST':
        if request.POST['titulo']:
            rec = colaborativo_SVD(request, request.POST['titulo'])
            #rec = ['A Light in the Window (The Mitford Years)', 'A Long Fatal Love Chase', 'A Maidens Grave', 'A New Song (Mitford Years (Paperback))', 'A Patchwork Planet', 'After All These Years', 'At Home in Mitford (The Mitford Years)', 'Back When We Were Grownups : A Novel (Ballantine Readers Circle)', 'Bad Love (Alex Delaware Novels (Paperback))', 'Big Stone Gap', 'Birds of Prey: A Novel of Suspense', 'Black and Blue', 'Blood Work', 'Cane River (Oprahs Book Club (Paperback))', 'Colony', 'Coming Home', 'Dangerous to Know', 'Dr. Atkins New Diet Revolution', 'Dust to Dust', 'East of Eden (Oprahs Book Club)', 'Evening Class', 'Fortunes Rocks: A Novel', 'Gap Creek: The Story Of A Marriage', 'Gone For Good', 'Good in Bed', 'Here on Earth (Oprahs Book Club)', 'I Dont Know How She Does It: The Life of Kate Reddy, Working Mother', 'Icy Sparks', 'In Pursuit of the Proper Sinner', 'Ladder of Years', 'Light a Penny Candle', 'London Transports', 'Manhattan Hunt Club', 'Memoirs of a Geisha', 'Mother of Pearl (Oprahs Book Club (Paperback))', 'Out to Canaan (The Mitford Years)', 'Outer Banks', 'Personal Injuries', 'Plantation: A Lowcountry Tale', 'Portrait of a Killer: Jack the Ripper__ Case Closed (Berkley True Crime)', 'Promises', 'Quentins', 'Ralphs Party', 'Revenge of the Middle_Aged Woman', 'Roses Are Red', 'SEAT OF THE SOUL', 'Salem Falls', 'September', 'Serpents Tooth : A Peter Decker/Rina Lazarus Novel (Peter Decker &amp; Rina Lazarus Novels (Paperback))', 'Spencerville', 'Sticks &amp; Scones', 'Still Waters', 'Suzannes Diary for Nicholas', 'Switcheroo : A Novel', 'Tell Me Lies (Tell Me Lies)', 'The Art of Happiness: A Handbook for Living', 'The Associate', 'The Burning Man', 'The Coffin Dancer (Lincoln Rhyme Novels (Paperback))', 'The Copper Beech', 'The Last Time They Met : A Novel', 'The Lilac Bus: Stories', 'The Mermaids Singing', 'The Pilots Wife : A Novel Tag: Author of the Weight of Water (Oprahs Book Club (Hardcover))', 'The Return Journey', 'The Shelters of Stone (Earths Children, Book 5)', 'The Valley of Horses', 'The Weight of Water', 'The Weight of Water : A Novel Tag _ Author of Resistance and Strange Fits of Passion', 'These High, Green Hills (The Mitford Years)', 'ThursdayS At Eight', 'Watermelon', 'Where You Belong', 'Where or When  : A Novel', 'Winter Solstice']

            return render(request, 'bx_library/home.html',{'biblioteca':rec})
    else:
        #return render(request, 'bx_library/home.html',{'biblioteca':titles,'user':username})
        return render(request, 'bx_library/home.html')