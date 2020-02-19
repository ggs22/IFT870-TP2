# %%
import numpy as np
import pandas as pd

# %%
package_file = 'package.csv'
product_file = 'product.csv'

product_data = pd.read_csv(product_file, sep=';', encoding='latin1')
package_data = pd.read_csv(package_file, sep=';')

# %%
"""
# 1. Auscultation
## Etude des données du fichier 'package'
"""

# %%
package_data.head()

# %%
package_data.isnull().sum().sort_values()

# %%
"""
### Colonne PRODUCTID
On remarque que, pour chaque objet, la valeur du 'PRODUCTNDC' y est incluse telle un préfixe. Dans la documentation NDC, il est précisé que c'est pour prévenir le duplicata de lignes. Il y a 41 valeurs nulles.
"""

# %%
"""
### Colonne PACKAGEDESCRIPTION
Cette colonne est sous forme de phrase et contient de multiples informations. Le volume, son unité, le nombre de contenant et son type. S'il existe plusieurs contenants pour un objet, ils sont concaténés avec un séparateur '>' de manière hiérarchique. Aussi, on remarque aussi que la valeur du 'NDCPACKAGECODE' de l'objet est incluse entre paranthèses. 
"""

# %%
"""
### Colonne STARTMARKETINGDATE
Les valeurs de cette colonne sont de type date. Elles ont l'air séquentielles. 
"""

# %%
"""
### Colonne ENDMARKETINGDATE
Les valeurs de cette colonne sont de type date. Il existe beaucoup de valeurs manquantes.
"""

# %%
"""
### Colonne NDC_EXCLUDE_FLAG
"""

# %%
package_data['NDC_EXCLUDE_FLAG'].unique()

# %%
"""
Tous les objets possèdent la valeur N et ne présente aucune valeur manquante. 
"""

# %%
"""
### Colonne SAMPLE_PACKAGE
"""

# %%
package_data['SAMPLE_PACKAGE'].value_counts()

# %%
"""
Les valeurs possibles sont 'Y' ou 'N'. Il y a une majorité de 'N' et aucune valeur manquante.
"""

# %%
"""
## Etude des données du fichier 'product'
"""

# %%
product_data.head()

# %%
product_data.isnull().sum().sort_values()

# %%
"""
### Colonne PRODUCTID
"""

# %%
product_data['PRODUCTTYPENAME'].value_counts()

# %%
"""
On remarque que, pour chaque objet, la valeur du 'PRODUCTNDC' y est incluse telle un préfixe. Dans la documentation NDC, il est précisé que c'est pour prévenir le duplicata de lignes.
"""

# %%
"""
### Colonne PRODUCTTYPENAME
Il y a 7 valeurs possibles textuelles catégorielles dans cette colonne.
"""

# %%
"""
### Colonne PROPRIETARYNAME
"""

# %%
product_data['PROPRIETARYNAME'].nunique()

# %%
product_data['PROPRIETARYNAME'][393:401]

# %%
"""
Dans cette colonne, il existe un grand nombre de valeurs différentes. Les mêmes valeurs peuvent être présentes sous 
différentes formes notamment en minuscules ou majuscules, il existe donc une inconsistance entre les valeurs.
"""

# %%
"""
### Colonne PROPRIETARYNAMESUFFIX
"""

# %%
product_data['PROPRIETARYNAMESUFFIX'].nunique()

# %%
"""
Dans cette colonne, il y a un nombre important de valeurs manquantes. Ces valeurs sont textuelles et il existe un nombre 
élevé de valeurs différentes.
"""

# %%
"""
### Colonne NONPROPRIETARYNAME
"""

# %%
product_data['NONPROPRIETARYNAME'].nunique()

# %%
product_data['NONPROPRIETARYNAME'][2:6]

# %%
"""
Cette colonne présente seulement 4 valeurs manquantes. Ceux sont des données textuelles inconsistantes, par exemple pouvant représenter la même valeur en caratères minuscules ou majuscules. Il y a un nombre très important de valeurs différentes.
"""

# %%
"""
### Colonne DOSAGEFORMNAME
"""

# %%
product_data['DOSAGEFORMNAME'].value_counts()

# %%
"""
Cette colonne contient 134 différentes valeurs textuelles. Comme on peut le voir, différentes catégories peuvent être affectées au même objet. La colonne ne présente aucune valeur manquante.
"""

# %%
"""
### Colonne ROUTENAME
"""

# %%
product_data['ROUTENAME'].value_counts()

# %%
"""
Cette colonne contient 180 différentes valeurs textuelles. Comme on peut le voir, différentes catégories peuvent être affectées au même objet. La colonne présente 1932 valeurs manquantes.
"""

# %%
