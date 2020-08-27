from django.db import models
# Create your models here.

class feature(models.Model):
    img_name = models.TextField()
    phash = models.TextField()
    group = models.IntegerField()
    class Meta:
        db_table = "feature"

class upload_record(models.Model):
    user = models.TextField()
    pdf = models.TextField()
    timestamp = models.DateTimeField()
    # timestamp = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = "upload_record"
