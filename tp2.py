# %%

import numpy as np
import pandas as pd
import re
import csv
import os
import pickle

from _datetime import datetime
from sklearn.preprocessing import OneHotEncoder

# %%

product_headers_to_encode = ['PRODUCTTYPENAME', 'ROUTENAME', 'DOSAGEFORMNAME', 'MARKETINGCATEGORYNAME',
                             'ACTIVE_NUMERATOR_STRENGTH', 'ACTIVE_INGRED_UNIT']
package_headers_to_encode = ['PACKAGEUNIT', 'PACKAGETYPE']

target_encoding = 'utf-8'
# separ = '\t'
separ = '|'
custom_sep = ' ?[|,;:<>] ?|^ | $'

product_file = 'Product2.csv'
package_file = 'Package2.csv'

encoder_dir = 'encoders/'
encoding_dir = 'enconding_dic/'

encoded_product_file = 'transformed_product_data.csv'
encoded_package_file = 'transformed_package_data.csv'

product_encode_file_exist = False
package_encode_file_exist = False


# TODO incohernce entre dates
# TODO incohernce entre routname / forme
# TODO incohernce entre valeurs numeric abberantes (ordre de grandeur)
# TODO incohernce entre valeurs phase et l'emballage
# TODO tester imputatin itérative

def assert_table_completeness(table):
    empty_cells = table.shape[0] - table.count(axis=0)
    unique_values = table.nunique(axis=0)

    print('Empty cells:\n{}\n'.format(empty_cells))
    print('Unique values:\n{}\n'.format(unique_values))


def assert_product_id_completeness(table, header):
    empty_cells = table.shape[0] - table.count(axis=0)
    unique_values = table.nunique(axis=0)

    if empty_cells[header] == 0:
        print('No empty values in the {} column'.format(header))
    else:
        print('There are {} empty values in the {} column'.format(empty_cells[header], header))

    if unique_values[header] == table.shape[0]:
        print('No duplicat values in the {} column'.format(header))
    else:
        print(
            'There are {} duplicat values in the {} column\n\n'.format(table.shape[0] - unique_values[header], header))


def get_unique_values(table, headers=''):
    uniques = {}
    if headers == '':
        cols = table.columns.values
        for n, c in enumerate(cols):
            uniques[c] = pd.unique(table[c])
    elif type(headers) is list:
        for header in headers:
            uniques[header] = pd.unique(table[header])
    elif type(headers) is str:
        uniques[headers] = pd.unique(table[headers])
    return uniques


def df_to_lower(table, columns='all'):
    cols = table.columns.values if columns == 'all' else columns
    for c in cols:
        try:
            table[c] = table[c].str.lower()
        except:
            pass


def get_decomposed_uniques(table, header):
    decomposed_uniques = {}
    if type(header) is str:
        for unique_header, uniques in get_unique_values(table, header).items():
            tmp_lst = []
            for val in uniques:
                if type(val) is str:
                    for decomposed in re.split(custom_sep, val):
                        if decomposed != '' and not decomposed in tmp_lst:
                            tmp_lst.append(decomposed)

            tmp_lst.sort()
            decomposed_uniques[unique_header] = tmp_lst
    else:
        raise TypeError('header should be a string representing a column header')

    return pd.DataFrame.from_dict(decomposed_uniques)


def get_onehot_encoders(table, cols):
    encoder_dict = {}
    for col in cols:
        uniques_vals = get_decomposed_uniques(table, header=col)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit_transform(uniques_vals)
        encoder_dict[col] = enc
    return encoder_dict


def onehot_encode(table, header):
    # Create onehot codes for the specidfied column
    lst = []
    encoder_dict = get_onehot_encoders(table, [header])

    count = 0
    for index in range(table.shape[0]):
        _tmp = np.zeros([1, len(encoder_dict[header].categories_[0])], dtype=int)
        if type(table.loc[index, header]) is str:
            for decomposed in re.split(custom_sep, table.loc[index, header]):
                _tmp |= np.int_(encoder_dict[header].transform([[decomposed]]).toarray())
            lst.append(_tmp)

        # Update loading bar
        if count == 1000:
            progress(index, table.shape[0])
            count = 0
        count += 1

    print(" -> Done", flush=True)

    # Replace dataframe column by encoded values
    table.loc[:, header] = pd.Series(lst)

    # return the encoder associated to that particular header
    return encoder_dict[header]


def time_methode(methode, status='', **kwargs):
    print('Timing {}'.format(methode.__name__))
    if status != '':
        print(status)
    start_time = datetime.now()
    print('Start time: {}'.format(start_time))
    ret = methode(**kwargs)
    end_time = datetime.now()
    print('End time: {}'.format(end_time))
    print('{} took: {}'.format(methode.__name__, (end_time - start_time)))
    if ret != '':
        return ret
    else:
        ret = 0
    return ret


def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))
    _str = ''
    percents = np.ceil(100.0 * count / float(total))
    bar = '=' * filled_len + ':' * (bar_len - filled_len)

    if status == '':
        _str = '|{}| {}%'.format(bar, percents)
    else:
        _str = '|{}| {}% - {}'.format(bar, percents, status)

    print('\r', end='', flush=True)
    print(_str, end='', flush=True)


"""
Load data:
    Will look for existing files to deserialize prior encoding data. If the files are not found
    it will proceed with the original data through encoding.
"""
product_encode_file_exist = os.path.isfile(encoded_product_file)
package_encode_file_exist = os.path.isfile(encoded_package_file)

enc_dic = {}

original_product_data = pd.read_csv(product_file, sep=';', encoding='latin1')
original_package_data = pd.read_csv(package_file, sep=';', encoding='latin1')

# %%

if product_encode_file_exist:
    print('Loading encoded product data from existing file...')
    product_data = pd.read_csv(encoded_product_file, sep=separ, encoding=target_encoding)
    # Populate onehot encoders dictionnary
    for header in product_headers_to_encode:
        enc_dic[header] = pickle.load(open(encoder_dir + '{}_data_encoder.pkl'.format(header), 'rb'))
else:
    product_data = original_product_data

if package_encode_file_exist:
    print('Loading encoded package data from existing file...')
    package_data = pd.read_csv(encoded_package_file, sep=separ, encoding=target_encoding)

    # Populate onehot encoders dictionnary
    for header in package_headers_to_encode:
        enc_dic[header] = pickle.load(open(encoder_dir + '{}_data_encoder.pkl'.format(header), 'rb'))
else:
    package_data = original_package_data

# Make everything lower characters in both tables
df_to_lower(product_data)
df_to_lower(package_data)

if product_encode_file_exist:
    print('Get unique values for ROUTENAME column of PRODUCT table')
    product_unique_values = get_decomposed_uniques(original_product_data, 'ROUTENAME')
    print(product_unique_values)
    print(enc_dic['ROUTENAME'].categories_[0])

# drop useless index columns
product_data = product_data.drop(product_data.columns[0], axis=1)
package_data = package_data.drop(package_data.columns[0], axis=1)

# %%
"""
# 1. Auscultation
Nous avons déjà prétraitées les données (passage en minuscules des données textuelles) afin de minimiser l'inconsistance
entre les valeurs.

## Etude des données du fichier 'package'
"""

# %%
print('Assessing completeness of packaging data table')
assert_table_completeness(package_data)

# %%
"""
La colonne PRODUCTID présente des valeurs manquantes, qui paraissent bloquantes. Les valeurs manquantes des colonnes 
STARTMARKETINGDATE et ENDMARKETINGDATE sont plus nombreuses mais semblent être non bloquantes. Ces deux dernières 
colonnes sont de type date et les valeurs de STARTMARKETINGDATE ont l'air séquentielles.

La colonne PACKAGEDESCRIPTION est présentée sous forme de phrase et contient de multiples informations: le type de 
volume, sa valeur et son unité. S'il existe plusieurs contenants pour un objet, ils sont concaténés par un séparateur 
'>' de manière hiérarchique.

Les colonnes NDC_EXCLUDE_FLAG et SAMPLE_PACKAGE, présentant peu de valeurs différentes, et ont l'air facilement 
traitables numériquement.
"""
# %%
"""
## Etude des données du fichier 'product'
"""

# %%
print('Assessing completeness product data table')
assert_table_completeness(product_data)

# %%

# %%
"""
Dans la colonne PRODUCTTYPENAME, on remarque 7 valeurs possibles textuelles catégorielles dans cette colonne et aucune
valeur manquante. Cette colonne sera donc facilement numérisable. 

La colonne PROPRIETARYNAME dispose d'un grand nombre de valeurs différentes, de type textuelle. Ces valeurs sont assez 
variables (phrase, simple mot) décrivant plus ou moins le produit. 
La colonne PROPRIETARYNAMESUFFIX est du même type que PROPRIETARYNAME, cependant elle présente beaucoup de valeurs 
nulles, et apporte des informations variantes aux objets. La documentation précise ne pas reconnaître de standard.

La colonne NONPROPRIETARYNAME présente seulement 4 valeurs manquantes mais un nombre très important de valeurs 
textuelles différentes. Elle indique les ingrédients actifs du produit, donc présente ses valeurs sous forme de liste
(inconsistante dans sa représentation). Les valeurs manquantes paraissent difficilement remplissables.

La colonne DOSAGEFORMNAME présente des données du standard FDA. En les étudiant, on se rend compte que nous pourrions 
simplifier notre utilisation du standard. En effet, celui-ci apporte une information principale sur le mode 
d'administration et présente certaines caractéristiques plus spécifique au mode. Ces dernières pourraient être omises 
pour notre utilisation car trop spécifiques et pouvant être globalisés en gardant seulement l'information principale
du mode d'administration.

La colonne ROUTENAME présente des données du standard FDA. Chaque objet a la possibilité d'en contenir plusieurs. On 
remarque que la représentation de données multiples est consistante, via un séparateur ';'. Il y a un nombre conséquent
de données manquantes qui seront à priori difficiles à compléter.

Les colonnes STARTMARKETINGDATE et ENDMARKETINGDATE sont similaires à celle présentes dans la table 'package'. 
Cependant, dans cette table, il n'y a aucune valeur manquante pour la colonne STARTMARKETINGDATE.
nt de type date, il y a un grand nombre de valeurs manquantes. 

La colonne MARKETINGCATEGORYNAME présente des données du standard FDA. Il n'y a pas de valeur manquante et seulement 
10 catégories différentes, la colonne sera donc numérisables facilement. 

La colonne APPLICATIONNUMBER spécifie pour chaque objet le numéro de catégorie marketing associée (présente dans la 
colonne MARKETINGCATEGORYNAME). Il y a un nombre important de valeurs manquantes. 

La colonne LABELERNAME présente des données textuelles très inconsistantes réflétant donc le nombre important de valeurs
différentes. Cette colonne parait difficilement numérisables et les valeurs manquantes (557) non complétables. 
"""

# %%
print(product_data['LABELERNAME'][7253:7256])

# %%
"""

La colonne SUBSTANCENAME présente des données du standard FDA, celui-ci est composé de 108 227 catégories différentes.
Chaque objet peut présenter plusieurs catégories, la représentation de valeurs multiples est consistante via le 
séparateur ';'. Cela pourrait expliquer le nombre important de valeurs différentes. Le nombre de valeurs manquantes
est important et les valeurs seront difficilement complétables.

Les colonnes ACTIVE_NUMERATOR_STRENGTH et ACTIVE_INGRED_UNIT présentent des valeurs liées. Il existe des valeurs
multiples et une consistance dans leur représentation via le séparateur ';'. Le nombre de valeurs manquantes est égale 
pour les deux colonnes. Elles paraissent assez facilement numérisables mais difficilement complétables.

La colonne PHARM_CLASSES présente des données du standard FDA, cependant il y en a un extrêmement important. Chaque 
objet peut disposer de plusieurs valeurs, la représentation de multiples valeurs semblent être consistante via le 
séparateur ','. 
Comme précisé par la FDA, ces données sont les catégories pharmaceutiques correspondants aux substances
du produit (valeurs contenues dans la colonne SUBSTANCENAME). On sait cependant qu'il existe un nombre assez important 
de valeurs de noms de substances manquantes. 

La colonne DEASCHEDULE présente des données du standard FDA. Ces données semblent être facilement numérisables, il y
cependant un nombre important de données manquantes qui seront difficilement complétables car nécessite de les traiter
un à un par un expert.

La colonne NDC_EXCLUDE_FLAG présente seulement la catégorie N pour notre jeu de données, comme précisé dans la 
documentation. Il n'y a pas de valeur manquante.
"""

# %%
print(product_data['NDC_EXCLUDE_FLAG'].value_counts())

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
# dupl_val_cols = ['ACTIVE_NUMERATOR_STRENGTH', 'ACTIVE_INGRED_UNIT']
# for c in dupl_val_cols:
#     product_data[c] = product_data[c].replace(to_replace=r'\;.*', value='', regex=True)


# %%
"""
Il existerait également une incohérence si l'attribut 'ENDMARKETINGDATE' est moins récent que le 'STARTMARKETINGDATE'.
On vérifie s'il en existe dans les tables 'product' et 'package'.
"""


# %%

# conversion to datetime format
def date_convert():
    date_cols = ['STARTMARKETINGDATE', 'ENDMARKETINGDATE', 'LISTING_RECORD_CERTIFIED_THROUGH']
    for c in date_cols:
        product_data[c] = pd.to_datetime(product_data[c], errors='coerce', format='%Y%m%d')


if not product_encode_file_exist:
    time_methode(date_convert)

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

if not package_encode_file_exist:
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

# %%
"""
Traitement des colonnes 'STARTMARKETINGDATE', 'ENDMARKETINGDATE' similairement à la table 'product'.
"""

# %%

if not package_encode_file_exist:
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
On s'intéresse aux données manquantes dans les colonnes PRODUCTID, PRODUCTNDC, NDCPACKAGECODE.
"""

# %%
if not package_encode_file_exist:
    package_missing_ndcpackagecode = package_data.iloc[np.where(pd.isnull(package_data['NDCPACKAGECODE']))]
    values = package_missing_ndcpackagecode['PACKAGEDESCRIPTION'].str.extract(r'\((.*?)\).*')
    for index, row in values.iterrows():
        package_data.loc[index, 'NDCPACKAGECODE'] = row[0]

# %%
if not package_encode_file_exist:
    package_missing_productndc = package_data.iloc[np.where(pd.isnull(package_data['PRODUCTNDC']))]
    values = package_missing_productndc['NDCPACKAGECODE'].str.extract(r'^([\w]+-[\w]+)')
    for index, row in values.iterrows():
        package_data.loc[index, 'PRODUCTNDC'] = row[0]

# %%

if not package_encode_file_exist:
    # TODO : find a way to retrieve PRODUCTID from 'product' table
    package_missing_ndcproductid = package_data.iloc[np.where(pd.isnull(package_data['PRODUCTID']))]

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
# 5. Duplications données
"""

# %%

# TODO: drop column PACKAGEDESCRIPTION

# %%
"""
# Transformation en données numériques (après question 8)
## Table 'package'
"""

# %%
"""
## Table 'product'
"""

# %%

# %%
# TODO: hash PROPRIETARYNAME NONPROPRIETARYNAME LABELERNAME PROPRIETARYNAMESUFFIX
# TODO: separate and hash SUBSTANCENAME PHARM_CLASSES
# TODO : split ACTIVE_INGRED_UNIT by '/' (nan others), then one hot each col

# TODO: ideas?? APPLICATIONNUMBER
# %%
# TODO : analysis ratio per category

# %%
"""
## Encodage onehot
"""

# %%

# Call and time onehot encoding for all predefined columns
if not os.path.isdir(encoder_dir):
    os.mkdir(encoder_dir)
if not product_encode_file_exist:
    for header in product_headers_to_encode:
        enc_dic[header] = time_methode(onehot_encode, header, **(dict(table=product_data, header=header)))
        pickle.dump(enc_dic[header], open(encoder_dir + '{}_data_encoder.pkl'.format(header), 'wb'),
                    pickle.HIGHEST_PROTOCOL)

if not package_encode_file_exist:
    for header in package_headers_to_encode:
        enc_dic[header] = time_methode(onehot_encode, header, **(dict(table=package_data, header=header)))
        pickle.dump(enc_dic[header], open(encoder_dir + '{}_data_encoder.pkl'.format(header), 'wb'),
                    pickle.HIGHEST_PROTOCOL)

if not os.path.isdir(encoding_dir):
    os.mkdir(encoding_dir)
# Prints out encding of each category for a given column in a txt file
for header, enc in enc_dic.items():
    file = open(encoding_dir + 'Encoding_{}.txt'.format(header), 'w')
    for category in enc.categories_[0]:
        tmp_str = str(enc.transform([[category]]).toarray())
        tmp_str = category + ' ' * (40 - len(category)) + tmp_str.replace('\n', '\n' + ' ' * 40) + '\n'
        file.write(tmp_str)
    file.close()

# Save transformed data to file
if not product_encode_file_exist:
    time_methode(product_data.to_csv, **(dict(path_or_buf=encoded_product_file,
                                              index=False,
                                              sep=separ,
                                              encoding=target_encoding,
                                              quoting=csv.QUOTE_NONNUMERIC)))

if not product_encode_file_exist:
    time_methode(package_data.to_csv, **(dict(path_or_buf=encoded_package_file,
                                              index=False,
                                              sep=separ,
                                              encoding=target_encoding,
                                              quoting=csv.QUOTE_NONNUMERIC)))

# %%
"""
## Résultats
"""

# %%
print('Encoded product data:')
print(product_data)
product_data

# %%
print('Encoded packaging data:')
print(package_data)
package_data
