# %%

import numpy as np
import pandas as pd
import re
import csv
import threading
import multiprocessing
import os
import pickle

from _datetime import datetime
from sklearn.preprocessing import OneHotEncoder

# %%

product_headers_to_encode = ['PRODUCTTYPENAME', 'ROUTENAME', 'DOSAGEFORMNAME', 'MARKETINGCATEGORYNAME']
package_headers_to_encode = ['PACKAGEUNIT', 'PACKAGETYPE']

target_encoding = 'utf-8'
separ = '\t'

product_file = 'product.csv'
package_file = 'package.csv'

encoded_product_file = 'transformed_product_data.csv'
encoded_package_file = 'transformed_package_data.csv'

product_encode_file_exist = False
package_encode_file_exist = False


# TODO incohernce entre dates
# TODO incohernce entre routname / forme
# TODO incohernce entre valeurs numeric abberantes (ordre de grandeur)
# TODO incohernce entre valeurs phase et l'emballage
# TODO tester imputatin itérative
# TODO utiliser le one hot de sk learn au lieu de dummies de pandas

def assert_table_completeness(table):
    empty_cells = table.shape[0] - table.count(axis=0)
    unique_values = table.nunique(axis=0)

    print('Empty cells:\n{}\n'.format(empty_cells))
    print('Unique values:\n{}\n'.format(unique_values))


def assert_product_id_completeness(table, header):
    empty_cells = table.shape[0] - table.count(axis=0)
    unique_values = table.nunique(axis=0)

    try:
        assert empty_cells[header] == 0
        print('No empty values in the {} column'.format(header))
    except:
        print('There are {} empty values in the {} column'.format(empty_cells[header], header))
    try:
        assert unique_values[header] == table.shape[0]
        print('No duplicat values in the {} column'.format(header))
    except:
        print(
            'There are {} duplicat values in the {} column\n\n'.format(table.shape[0] - unique_values[header], header))


def get_unique_values(table, headers=''):
    uniques = {}
    if headers == '':
        cols = table.columns.values
        for n, c in enumerate(cols):
            uniques[c] = pd.unique(table[c])
    # else:
    #     uniques[headers] = pd.unique(table[headers])
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
                    for decomposed in re.split(' ?[_|,;:<>/;] ?|^ | $', val):
                        if decomposed != '' and \
                                not decomposed in tmp_lst:
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


def onehot_encode(table, header, threaded=False):
    # Create onehot codes for the specidfied column
    lst = []
    encoder_dict = get_onehot_encoders(table, [header])

    if threaded:
        threads = []

        class _onehot_encode_thread(threading.Thread):
            def __init__(self, threadID, name, start_index, end_index):
                threading.Thread.__init__(self)
                self.threadID = threadID
                self.name = name
                self.start_index = start_index
                self.end_index = end_index

            def run(self):
                _int_onehot_encode(self.start_index, self.end_index)

        def _int_onehot_encode(start_index, end_index):
            for index in np.int_(np.linspace(start_index, end_index, (end_index - start_index) + 1)):
                _tmp = np.zeros([1, len(encoder_dict[header].categories_[0])], dtype=int)
                if type(table.loc[index, header]) is str:
                    for decomposed in re.split('[_|,;:<>/;] ?|^ ', table.loc[index, header]):
                        _tmp |= np.int_(encoder_dict[header].transform([[decomposed]]).toarray())
                    lst.append(_tmp)

        cpu_count = multiprocessing.cpu_count()
        step = int(np.floor(table.shape[0] / cpu_count))
        rem = table.shape[0] % cpu_count;
        strt = 0

        for i in range(cpu_count):
            end = step * (i + 1)
            if i == cpu_count - 1:
                end += rem - 1
            thrd = _onehot_encode_thread(i, str(i), strt, end)
            thrd.start()
            threads.append(thrd)
            strt = end + 1

        count = 1
        for thrd in threads:
            thrd.join()
            # Update loading bar
            progress(1, 1, 'Thread {} finished'.format(count))
            print('')
            count += 1
    else:
        count = 0
        for index in range(table.shape[0]):
            _tmp = np.zeros([1, len(encoder_dict[header].categories_[0])], dtype=int)
            if type(table.loc[index, header]) is str:
                for decomposed in re.split('[_|,;:<>/;] ?|^ ', table.loc[index, header]):
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
    bar = '|' * filled_len + '_' * (bar_len - filled_len)

    if status == '':
        _str = '[{}] {}%'.format(bar, percents)
    else:
        _str = '[{}] {}% - {}'.format(bar, percents, status)

    print('\r', end='', flush=True)
    print(_str, end='', flush=True)


product_encode_file_exist = os.path.isfile(encoded_product_file)
package_encode_file_exist = os.path.isfile(encoded_package_file)

tmp_dic = {}

original_product_data = pd.read_csv(product_file, sep=';', encoding='latin1')
original_package_data = pd.read_csv(package_file, sep=';', encoding='latin1')

if product_encode_file_exist:
    print('Loading encoded product data from existing file...')
    product_data = pd.read_csv(encoded_product_file, sep=separ, encoding=target_encoding)

    # Populate onehot encoders dictionnary
    for header in product_headers_to_encode:
        tmp_dic[header] = pickle.load(open('{}_data_encoder.pkl'.format(header), 'rb'))
else:
    product_data = original_product_data

if package_encode_file_exist:
    print('Loading encoded package data from existing file...')
    package_data = pd.read_csv(encoded_package_file, sep=separ, encoding=target_encoding)

    # Populate onehot encoders dictionnary
    for header in package_headers_to_encode:
        tmp_dic[header] = pickle.load(open('{}_data_encoder.pkl'.format(header), 'rb'))
else:
    package_data = original_package_data

# Make everything lower characters in both tables
df_to_lower(product_data)
df_to_lower(package_data)

print('Get unique values for ROUTENAME column of PRODUCT table')
product_unique_values = get_decomposed_uniques(product_data, 'ROUTENAME')
print(product_unique_values)
print('Get unique values for each column of PACKAGING table')
package_unique_values = get_unique_values(package_data)

# %%
"""
# 1. Auscultation
## Etude des données du fichier 'package'
"""

# %%
package_data.head()

# %%
print('Assessing completnes of packaging data table')
assert_table_completeness(package_data)

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
print('Assessing completnes product data table')
assert_table_completeness(product_data)

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
# product_data['PROPRIETARYNAME'][393:401]

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

# Call and time onehot encoding for a column
if not product_encode_file_exist:
    for header in product_headers_to_encode:
        kwargs = dict(table=product_data, header=header)
        tmp_dic[header] = time_methode(onehot_encode, header, **kwargs)
        pickle.dump(tmp_dic[header], open('{}_data_encoder.pkl'.format(header), 'wb'), pickle.HIGHEST_PROTOCOL)

if not package_encode_file_exist:
    for header in package_headers_to_encode:
        kwargs = dict(table=package_data, header=header)
        tmp_dic[header] = time_methode(onehot_encode, header, **kwargs)
        pickle.dump(tmp_dic[header], open('{}_data_encoder.pkl'.format(header), 'wb'), pickle.HIGHEST_PROTOCOL)

for header, enc in tmp_dic.items():
    file = open('Encoding_{}.txt'.format(header), 'w')
    for category in enc.categories_[0]:
        tmp_str = str(enc.transform([[category]]).toarray())
        tmp_str = category + ' ' * (40 - len(category)) + tmp_str.replace('\n', '\n' + ' ' * 40) + '\n'
        file.write(tmp_str)
    file.close()

# Save transformed data to file
if not product_encode_file_exist:
    kwargs = dict(path_or_buf= encoded_product_file, index= False, sep= separ, encoding= target_encoding,
              quoting= csv.QUOTE_NONNUMERIC)
    time_methode(product_data.to_csv, **kwargs)

if not product_encode_file_exist:
    kwargs = dict(path_or_buf=encoded_package_file, index=False, sep=separ, encoding=target_encoding,
                  quoting=csv.QUOTE_NONNUMERIC)
    time_methode(package_data.to_csv, **kwargs)

# %%
"""
## Résultats
"""

# %%
print('Encoded product data:')
product_data

# %%
print('Encoded packaging data:')
package_data