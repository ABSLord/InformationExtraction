from django.db import models


class ExtractionModel(models.Model):

    name = models.CharField(max_length=200, help_text="extraction model name", unique=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "ExtractionModel"
