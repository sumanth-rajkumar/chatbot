# Generated by Django 5.1.3 on 2024-12-04 20:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('file_manager', '0002_powerpointfile_extracted_text'),
    ]

    operations = [
        migrations.AddField(
            model_name='powerpointfile',
            name='chunks_path',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='powerpointfile',
            name='faiss_index_path',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
