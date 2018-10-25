# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Titles(models.Model):
    titleno = models.TextField(primary_key=True)
    title = models.TextField(blank=True, null=True)
    subtitle = models.TextField(blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'titles'

    def __str__(self):
        return self.title


class Authors(models.Model):
    authorno = models.CharField(primary_key=True, max_length=10)
    fname = models.CharField(max_length=250, blank=True, null=True)
    sname = models.CharField(max_length=250, blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'authors'

    def __str__(self):
        return self.fname+' '+self.sname

class Estanterias(models.Model):
    id_shelf = models.CharField(primary_key=True, max_length=10)
    shelf = models.CharField(max_length=250, blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'estanterias'

    def __str__(self):
        return self.shelf

class Localizaciones(models.Model):
    id_loc = models.CharField(primary_key=True, max_length=10)
    loc = models.CharField(max_length=250, blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'localizaciones'

    def __str__(self):
        return self.loc

class Materias(models.Model):
    id_subject = models.CharField(primary_key=True, max_length=10)
    subject = models.CharField(max_length=250, blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'materias'

    def __str__(self):
        return self.subject

class TipoTrans(models.Model):
    id_ctrans = models.CharField(primary_key=True, max_length=10)
    ctrans = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'tipo_trans'

    def __str__(self):
        return self.ctrans


class TitleAuthor(models.Model):
    id = models.BigIntegerField(primary_key=True)
    titleno = models.ForeignKey('Titles', models.DO_NOTHING, db_column='titleno')
    authorno = models.ForeignKey(Authors, models.DO_NOTHING, db_column='authorno')

    class Meta:
        managed = True
        db_table = 'title_author'


class TitleMat(models.Model):
    id = models.BigIntegerField(primary_key=True)
    titleno = models.ForeignKey('Titles', models.DO_NOTHING, db_column='titleno')
    id_subject = models.ForeignKey(Materias, models.DO_NOTHING, db_column='id_subject')

    class Meta:
        managed = True
        db_table = 'title_mat'

class Notas(models.Model):
    id = models.BigIntegerField(primary_key=True)
    titleno = models.ForeignKey('Titles', models.DO_NOTHING, db_column='titleno')
    note = models.TextField(blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'notas'

class Copies(models.Model):
    copyno = models.CharField(primary_key=True, max_length=10)
    titleno = models.ForeignKey('Titles', models.DO_NOTHING, db_column='titleno')
    barcode = models.CharField(max_length=20, blank=True, null=True)
    title = models.TextField(blank=True, null=True)
    signatura = models.CharField(max_length=50, blank=True, null=True)
    id_shelf = models.ForeignKey('Estanterias', models.DO_NOTHING, db_column='id_shelf', blank=True, null=True)
    id_loc = models.ForeignKey('Localizaciones', models.DO_NOTHING, db_column='id_loc', blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'copies'

    def __str__(self):
        return self.copyno

class Transacciones(models.Model):
    id = models.BigIntegerField(primary_key=True)
    fecha = models.CharField(max_length=10, blank=True, null=True)
    id_ctrans = models.ForeignKey(TipoTrans, models.DO_NOTHING, db_column='id_ctrans', blank=True, null=True)
    usuario = models.TextField(blank=True, null=True)
    id_loc = models.ForeignKey(Localizaciones, models.DO_NOTHING, db_column='id_loc', blank=True, null=True)
    titleno = models.ForeignKey(Titles, models.DO_NOTHING, db_column='titleno')
    copyno = models.ForeignKey(Copies, models.DO_NOTHING, db_column='copyno')

    class Meta:
        managed = True
        db_table = 'transacciones'

