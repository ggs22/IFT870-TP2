# %%
import datetime

import numpy as np
import pandas as pd
import re
import datetime
from sklearn.preprocessing import OneHotEncoder

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
Pour la table 'product':
Il semble pouvoir exister une corrélation entre les attributs 'ROUTENAME' et 'DOSAGEFORMNAME' qui présentent des idées 
d'administration similaires. 
On peut également considérer l'existance d'une corrélation entre les modes d'administration
et les dosages du médicament, donc les attributs 'ROUTENAME', 'DOSAGEFORMNAME' et ceux 'ACTIVE_NUMERATOR_STRENGTH', 
'ACTIVE_INGRED_UNIT'.
L'attribut 'PHARM_CLASS' semble pouvoir être corrélé à l'attribut 'SUBSTANCENAME'.
"""

# %%
"""
# 3. Correction des incohérences
On élimine dans un premier temps les duplicata de valeurs dans les attributs 'ACTIVE_NUMERATOR_STRENGTH', 
'ACTIVE_INGRED_UNIT' de la table 'product'. 
## Table 'product'
"""

# %%

# TODO: keep most frequent value
dupl_val_cols = ['ACTIVE_NUMERATOR_STRENGTH', 'ACTIVE_INGRED_UNIT']
for c in dupl_val_cols:
    product_data[c] = product_data[c].replace(to_replace=r'\;.*', value='', regex=True)

# %%
""""
On tranforme toutes les valeurs textuelles insconsistantes (majuscule/minuscule) des différents attributs.
"""

# %%
inconsistant_cols = ['PROPRIETARYNAME', 'PROPRIETARYNAMESUFFIX', 'NONPROPRIETARYNAME', 'LABELERNAME', 'SUBSTANCENAME',
                     'ACTIVE_INGRED_UNIT', 'PHARM_CLASSES']
for c in inconsistant_cols:
    product_data[c] = product_data[c].str.lower()

# %%
""""
Il existerait également une incohérence si l'attribut 'ENDMARKETINGDATE' est moins récent que le 'STARTMARKETINGDATE'.
On vérifie s'il en existe dans les tables 'product' et 'package'.
"""

# %%

# conversion to datetime format
date_cols = ['STARTMARKETINGDATE', 'ENDMARKETINGDATE', 'LISTING_RECORD_CERTIFIED_THROUGH']
for c in date_cols:
    product_data[c] = pd.to_datetime(product_data[c], errors='coerce', format='%Y%m%d')

# compare STARTMARKETINGDATE and ENDMARKETINGDATE
# replace ENDMARKETINGDATE to NaT when incoherence
product_data.loc[
    (product_data['STARTMARKETINGDATE'] > product_data['ENDMARKETINGDATE']), 'ENDMARKETINGDATE'] = pd.NaT

# %%
"""
## Table 'package'
La colonne 'PACKAGEDESCRIPTION' contient beaucoup trop d'informations pour être exploitable. Tout d'abord, on garde 
seulement l'information du package le plus informatif (le dernier) car spécifie le volume le plus précis.
On supprime l'information dupliquée du 'NDCPACKAGECODE'. 
Enfin, on crée une colonne pour chaque information: 'PACKAGESIZE', 'PACKAGEUNIT' et 'PACKAGETYPE'.
On peut retirer la colonne 'PACKAGEDESCRIPTION' de la table.
"""
# %%

# keep only most informative packaging and remove duplicate info NDCPACKAGECODE
package_data['PACKAGEDESCRIPTION'] = package_data['PACKAGEDESCRIPTION'].replace(to_replace=r'.*(\>|\*\ ) |\(.*',
                                                                                value='', regex=True)

# split info into multiple columns
search = {0: [], 1: [], 2: []}
for values in package_data['PACKAGEDESCRIPTION']:
    s = re.search(r'(^\.?[0-9\.]+)\ (.*)\ in\ 1\ (.*)', values)
    for i in range(3):
        search[i].append(s.group(i + 1))

for i, n in enumerate(['PACKAGESIZE', 'PACKAGEUNIT', 'PACKAGETYPE']):
    package_data[n] = search[i]

package_data = package_data.drop(columns=['PACKAGEDESCRIPTION'])

# %%
"""
Traimtement des colonnes 'STARTMARKETINGDATE', 'ENDMARKETINGDATE' similairement à la table 'product'.
"""

# %%
# conversion to datetime format
date_cols = ['STARTMARKETINGDATE', 'ENDMARKETINGDATE']
for c in date_cols:
    package_data[c] = pd.to_datetime(package_data[c], errors='coerce', format='%Y%m%d')

# compare STARTMARKETINGDATE and ENDMARKETINGDATE
# replace ENDMARKETINGDATE to NaT when incoherence
package_data.loc[
    (package_data['STARTMARKETINGDATE'] > package_data['ENDMARKETINGDATE']), 'ENDMARKETINGDATE'] = pd.NaT

# %%
"""
# 4. Données manquantes
## Table 'package'
"""

# %%

# TODO Gab: missing 'PRODUCTID', 'PRODUCTNDC', 'NDCPACKAGECODE' in 'package'

# %%
"""
Il existe des valeurs manquantes pour les colonnes 'STARTMARKETINGDATE' et 'ENDMARKETINGDATE' dans la table 'package'
mais on choisit de ne pas les compléter car on ne peut effectuer d'estimation précise. 
"""

# %%
"""
## Table 'product'
"""

# %%
"""
# Transformation en données numériques
## Table 'package'
"""

# %%

transf_package_data = package_data

# TODO: change get_dummies to OneHotEncoder
# transform PACKAGEUNIT and PACKAGETYPE categorial columns (multiple values) to one hot
transf_package_data = pd.concat([transf_package_data, transf_package_data['PACKAGEUNIT']
                                .str.get_dummies(sep=', ')
                                .add_prefix('PACKAGEUNIT')], axis=1)
transf_package_data = transf_package_data.drop(columns=['PACKAGEUNIT'])
transf_package_data = pd.concat([transf_package_data, transf_package_data['PACKAGETYPE']
                                .str.get_dummies(sep=', ')
                                .add_prefix('PACKAGETYPE')], axis=1)
transf_package_data = transf_package_data.drop(columns=['PACKAGETYPE'])

# %%

# convert PACKAGESIZE to proper numerical value
transf_package_data['PACKAGESIZE'] = pd.to_numeric(transf_package_data['PACKAGESIZE'])

# %%

# TODO: change get_dummies to OneHotEncoder
# convert NDC_EXCLUDE_FLAG and SAMPLE_PACKAGE
transf_package_data = pd.get_dummies(data=transf_package_data, columns=['NDC_EXCLUDE_FLAG', 'SAMPLE_PACKAGE'])

# %%
"""
## Table 'product'
"""