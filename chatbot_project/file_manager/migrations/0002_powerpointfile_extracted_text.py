# Generated by Django 5.1.3 on 2024-12-03 14:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('file_manager', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='powerpointfile',
            name='extracted_text',
            field=models.TextField(blank=True, null=True),
        ),
    ]
