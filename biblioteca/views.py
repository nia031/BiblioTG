from django.shortcuts import render, redirect, get_object_or_404, render_to_response
from django.contrib.auth.decorators import login_required
from .models import Titles
import time
import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


def home(request):
    titles=Titles.objects.filter()[:10]
    
    username = None
    if request.user.is_authenticated:
        username = request.user.get_username()

    return render(request, 'biblioteca/home.html',{'biblioteca':titles,'user':username})

def rec_contenido(request,title):

    conn = psycopg2.connect(host="localhost", dbname="bibliocarvajaldb", user="postgres", password="1234", port="5433")
    query = """
      SELECT A.titleno, A.title, A.subtitle, A.subject, A.note, A.fname,A.sname
        FROM(  Select titles.titleno,title,subtitle,subject,note,fname,sname,
                ROW_NUMBER() OVER(PARTITION BY titles.titleno ORDER BY title_mat.id_subject) AS RN
                From titles 
                left join title_mat on titles.titleno=title_mat.titleno
                join notas on notas.titleno=titles.titleno
                join materias on materias.id_subject=title_mat.id_subject
                join title_author on title_author.titleno=titles.titleno
                join authors on authors.authorno=title_author.authorno
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
    print(indices)
    print("AQUI")
    print(title)
    idx = indices[title]  
    print(idx)
    libro = data.iloc[idx]
    libro_target =libro.title
    
    libros_k={}          
    k_scores = list(enumerate(lin_kernel[idx]))
    k_scores = sorted(k_scores, key=lambda x: x[1], reverse=True)
    k_scores = k_scores[1:11]
    k_indices = [i[0] for i in k_scores]
    libros_k=data.iloc[k_indices]
    libros_k['score'] = [str(i[1]) for i in k_scores]   
    libros_k=libros_k.reset_index().T.to_dict('list')    

    libros_s={}
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    sim_indices = [i[0] for i in sim_scores]
    libros_s=data.iloc[sim_indices]
    libros_s['score'] = [str(i[1]) for i in sim_scores]        
    libros_s=libros_s.reset_index().T.to_dict('list')    

    print ('The script took {0} second !'.format(time.time() - startTime))    
    return render(request, 'biblioteca/rec_contenido.html',{'target': libro_target,'titulo': title, 'libros_k':libros_k, 'libros_s':libros_s})

    