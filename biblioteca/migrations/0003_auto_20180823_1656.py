# Generated by Django 2.1 on 2018-08-23 21:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('biblioteca', '0002_auto_20180823_1654'),
    ]

    operations = [
        migrations.AlterField(
            model_name='titles',
            name='titleno',
            field=models.TextField(primary_key=True, serialize=False),
        ),
    ]