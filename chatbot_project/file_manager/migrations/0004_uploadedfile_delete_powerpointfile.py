# Generated by Django 5.1.3 on 2024-12-05 16:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('file_manager', '0003_powerpointfile_chunks_path_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('file', models.FileField(upload_to='uploads/')),
                ('file_type', models.CharField(choices=[('pptx', 'PowerPoint'), ('pdf', 'PDF')], default='pptx', max_length=10)),
                ('extracted_text', models.TextField(blank=True, null=True)),
                ('faiss_index_path', models.CharField(blank=True, max_length=255, null=True)),
                ('chunks_path', models.CharField(blank=True, max_length=255, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.DeleteModel(
            name='PowerPointFile',
        ),
    ]
