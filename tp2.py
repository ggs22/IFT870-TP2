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
count_missing_values_package = package_data.isnull().sum().sort_values()

# %%
"""
### Colonne PACKAGEDESCRIPTION
Cette colonne est présentée sous forme de phrase et contient de multiples informations. Le volume, son unité, le nombre 
de contenant et son type. S'il existe plusieurs contenants pour un objet, ils sont concaténés par un séparateur '>' de 
manière hiérarchique.
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
count_missing_values_product = product_data.isnull().sum().sort_values()

# %%
"""
### Colonne PRODUCTTYPENAME
"""

# %%
product_data['PRODUCTTYPENAME'].value_counts()

# %%
"""
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
Cette colonne présente seulement 4 valeurs manquantes. Ceux sont des données textuelles inconsistantes, par exemple 
pouvant représenter la même valeur en caratères minuscules ou majuscules. Il y a un nombre très important de valeurs 
différentes.
"""

# %%
"""
### Colonne DOSAGEFORMNAME
"""

# %%
product_data['DOSAGEFORMNAME'].value_counts()

# %%
"""
Cette colonne contient 134 différentes valeurs textuelles. Comme on peut le voir, différentes catégories peuvent être 
affectées au même objet. La colonne ne présente aucune valeur manquante.
"""

# %%
"""
### Colonne ROUTENAME
"""

# %%
product_data['ROUTENAME'].value_counts()

# %%
"""
Cette colonne contient 180 différentes valeurs textuelles. Comme on peut le voir, différentes catégories peuvent être 
affectées au même objet. La colonne présente 1932 valeurs manquantes.
"""

# %%
"""
### Colonne STARTMARKETINGDATE
Les valeurs sont de type date, il n'y a aucune valeur manquante. 
"""

# %%
"""
### Colonne ENDMARKETINGDATE
Les valeurs sont de type date, il y a un grand nombre de valeurs manquantes. 
"""

# %%
"""
### Colonne MARKETINGCATEGORYNAME
"""

# %%
product_data['MARKETINGCATEGORYNAME'].value_counts()
# %%
"""
Les valeurs sont de type textuelles, il y a 26 catégories différentes et ne présente aucune valeur manquante.  
"""
# %%
"""
### Colonne APPLICATIONNUMBER
"""
# %%
product_data['APPLICATIONNUMBER'].nunique()
# %%
"""
Cette colonne spécifie le numéro de série de la catégorie marketing. Le nombre de valeurs manquantes est élevé, et 
comme il y a un numéro de série pour chaque objet dans une catégorie, le nombre de valeurs différentes est également 
important.
"""

# %%
"""
### Colonne LABELERNAME
"""

# %%
product_data['LABELERNAME'].nunique()

# %%
product_data['LABELERNAME'][7291:7293]
# %%
"""
La colonne présente peu de valeurs manquantes (557). Si on remarque qu'il existe un nombre important de valeurs 
différentes, les données sont cependant inconsistantes.
"""

# %%
"""
### Colonne SUBSTANCENAME
"""

# %%
product_data['SUBSTANCENAME'].nunique()

# %%
product_data['SUBSTANCENAME'][727:735]

# %%
"""
Cette colonne présente un nombre assez important de données manquantes (2309), ceux sont des données textuelles 
catégorielles. Or le nombre de catégories parait élevé, comme le montre le nombre de valeurs uniques. Chaque objet 
peut cependant présenté plusieurs catégories séparées par un ';'.
"""

# %%
"""
### Colonne ACTIVE_NUMERATOR_STRENGTH
"""

# %%
product_data['ACTIVE_NUMERATOR_STRENGTH'][725:731]

# %%
"""
Ceux sont des données numériques qui paraissent dupliquées pour le même objet.  
"""

# %%
"""
### Colonne ACTIVE_INGRED_UNIT
"""

# %%
product_data['ACTIVE_INGRED_UNIT'][725:731]

# %%
"""
Ceux sont des données textuelles catégorielles qui présentent l'unité de la colonne 'ACTIVE_INGRED_UNIT'. Les données
paraissent également dupliquées pour le même objet.
"""

# %%
"""
### Colonne PHARM_CLASSES
"""

# %%
product_data['PHARM_CLASSES'].nunique()

# %%
product_data['PHARM_CLASSES'][725]

# %%
"""
Ceux sont des données textuelles catégorielles présentant plusieurs catégories pour un même objet. Il y a un grand 
nombre de valeurs manquantes.
"""
# %%
"""
### Colonne DEASCHEDULE
"""

# %%
product_data['DEASCHEDULE'].value_counts()

# %%
"""
Cette colonne présente un nombre important de données manquantes. Ceux sont des données catégorielles, présentant 
seulement 4 catégories.
"""

# %%
"""
### Colonne NDC_EXCLUDE_FLAG
"""

# %%
product_data['NDC_EXCLUDE_FLAG'].value_counts()

# %%
"""
Cette colonne présente seulement une catégorie 'N'.
"""

# %%
"""
# 2. Relations entre attributs
## Informations communes
Les colonnes 'PRODUCTID' des tables 'package' et 'product' contiennent deux informations concaténées: l'id du produit 
ainsi que le contenu de leur colonne 'PRODUCTNDC', le code label et le code segment produit.  
Dans la documentation NDC, il est précisé que c'est pour prévenir le duplicata de lignes.

La colonne 'NDCPACKAGECODE' de la table 'package' contient deux informations concaténées: le code segment du package et 
le contenu de la colonne 'PRODUCTNDC', le code label et le code segment produit.

La colonne 'PACKAGEDESCRIPTION' de la table 'package' contient plusieurs informations concaténées. En plus des 
informations propres à la description du package, il y a dans la majorité des objets la valeur 'NDCPACKAGECODE' associée
.

La colonne 'APPLICATIONNUMBER' de la table 'product' présente la majorité du temps le contenu de la colonne 
'MARKETINGCATEGORYNAME' et spécifie son numéro de série.

Dans les deux tables, il existe des colonnes 'STARTMARKETINGDATE',  'ENDMARKETINGDATE' et 'NDCEXLUDEDFLAG'. 
Elles semblent présenter les mêmes informations.

## Corrélation

"""