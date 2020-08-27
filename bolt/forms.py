from django import forms

import os
import re

def validate_file_extension(value):
    pattern = "[a-zA-Z0-9]+_\d_.+"
    name = value.name.split('.')[0]
    ext = value.name.split('.')[-1].lower()
    valid_extensions = ['pdf']
    if ext not in valid_extensions:
        raise forms.ValidationError('請上傳PDF檔!')
    if not re.fullmatch(pattern, name):
        raise forms.ValidationError('檔名不符合："工程編號_公司代號_PDF檔名"')

class FileFieldForm(forms.Form):
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}), validators=[validate_file_extension])

class DataForm(forms.Form):
    phash = forms.CharField(label='phash', widget=forms.HiddenInput())
    group = forms.CharField(label='group', widget=forms.HiddenInput())
    # phash = forms.CharField(label='phash')
    # group = forms.CharField(label='group')