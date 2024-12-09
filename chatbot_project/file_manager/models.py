from django.db import models

class UploadedFile(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    file_type = models.CharField(max_length=10, choices=[('pptx', 'PowerPoint'), ('pdf', 'PDF')], default='pptx')
    extracted_text = models.TextField(blank=True, null=True)
    faiss_index_path = models.CharField(max_length=255, blank=True, null=True)
    chunks_path = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return str(self.name)
