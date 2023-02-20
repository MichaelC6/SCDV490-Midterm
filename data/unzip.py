import zipfile
with zipfile.ZipFile('euroleague.zip', 'r') as zip_ref:
    zip_ref.extractall('')