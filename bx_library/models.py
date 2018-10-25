from django.db import models

# Create your models here.
class BxBooks(models.Model):
    isbn = models.CharField(primary_key=True, max_length=13)
    book_title = models.CharField(max_length=255, blank=True, null=True)
    book_author = models.CharField(max_length=255, blank=True, null=True)
    year_publication = models.IntegerField(blank=True, null=True)
    publisher = models.CharField(max_length=255, blank=True, null=True)
    imageurls = models.CharField(max_length=255, blank=True, null=True)
    imageurlm = models.CharField(max_length=255, blank=True, null=True)
    imageurll = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'bx_books'


class BxUsers(models.Model):
    user_id = models.IntegerField(primary_key=True)
    location = models.CharField(max_length=250, blank=True, null=True)
    age = models.CharField(max_length=250, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'bx_users'

class BxBookRatings(models.Model):
    user = models.ForeignKey('BxUsers', models.DO_NOTHING, db_column='user_id',related_name='BxBookRatings')
    isbn = models.ForeignKey('BxBooks', models.DO_NOTHING, db_column='isbn',related_name='BxBookRatings')
    book_rating = models.IntegerField()
    prediction = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'bx_book_ratings'