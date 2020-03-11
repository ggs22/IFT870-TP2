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
package_headers_to_encode = ['PACKAGEDESCRIPTION']

date_cols = ['STARTMARKETINGDATE', 'ENDMARKETINGDATE', 'LISTING_RECORD_CERTIFIED_THROUGH']

standard_dosageformname = {"AEROSOL": "AEROSOL", "AEROSOL, FOAM": "AEROSOL", "AEROSOL, METERED": "AEROSOL",
                           "AEROSOL, POWDER": "AEROSOL", "AEROSOL, SPRAY": "AEROSOL", "BAR": "BAR",
                           "BAR, CHEWABLE": "BAR", "BEAD": "BEAD", "CAPSULE": "CAPSULE", "CAPSULE, COATED": "CAPSULE",
                           "CAPSULE, COATED PELLETS": "CAPSULE", "CAPSULE, COATED, EXTENDED RELEASE": "CAPSULE",
                           "CAPSULE, DELAYED RELEASE": "CAPSULE", "CAPSULE, DELAYED RELEASE PELLETS": "CAPSULE",
                           "CAPSULE, EXTENDED RELEASE": "CAPSULE", "CAPSULE, FILM COATED, EXTENDED RELEASE": "CAPSULE",
                           "CAPSULE, GELATIN COATED": "CAPSULE", "CAPSULE, LIQUID FILLED": "CAPSULE",
                           "CELLULAR SHEET": "CELLULAR SHEET", "CHEWABLE GEL": "CHEWABLE GEL", "CLOTH": "CLOTH",
                           "CONCENTRATE": "CONCENTRATE", "CREAM": "CREAM", "CREAM, AUGMENTED": "CREAM",
                           "CRYSTAL": "CRYSTAL", "DISC": "DISC", "DOUCHE": "DOUCHE", "DRESSING": "DRESSING",
                           "ELIXIR": "ELIXIR", "EMULSION": "EMULSION", "ENEMA": "ENEMA", "EXTRACT": "EXTRACT",
                           "FIBER, EXTENDED RELEASE": "FIBER", "FILM": "FILM", "FILM, EXTENDED RELEASE": "FILM",
                           "FILM, SOLUBLE": "FILM", "FOR SOLUTION": "FOR SOLUTION", "FOR SUSPENSION": "FOR SUSPENSION",
                           "FOR SUSPENSION, EXTENDED RELEASE": "FOR SUSPENSION", "GAS": "GAS", "GEL": "GEL",
                           "GEL, DENTIFRICE": "GEL", "GEL, METERED": "GEL", "GLOBULE": "GLOBULE", "GRANULE": "GRANULE",
                           "GRANULE, DELAYED RELEASE": "GRANULE", "GRANULE, EFFERVESCENT": "GRANULE",
                           "GRANULE, FOR SOLUTION": "GRANULE", "GRANULE, FOR SUSPENSION": "GRANULE",
                           "GRANULE, FOR SUSPENSION, EXTENDED RELEASE": "GRANULE", "GUM": "GUM", "GUM, CHEWING": "GUM",
                           "IMPLANT": "IMPLANT", "INHALANT": "INHALANT", "INJECTABLE FOAM": "INJECTABLE FOAM",
                           "INJECTABLE": "INJECTABLE", "INJECTABLE, LIPOSOMAL": "INJECTABLE", "INJECTION": "INJECTION",
                           "INJECTION, EMULSION": "INJECTION", "INJECTION, LIPID COMPLEX": "INJECTION",
                           "INJECTION, POWDER, FOR SOLUTION": "INJECTION",
                           "INJECTION, POWDER, FOR SUSPENSION": "INJECTION",
                           "INJECTION, POWDER, FOR SUSPENSION, EXTENDED RELEASE": "INJECTION",
                           "INJECTION, POWDER, LYOPHILIZED, FOR LIPOSOMAL SUSPENSION": "INJECTION",
                           "INJECTION, POWDER, LYOPHILIZED, FOR SOLUTION": "INJECTION",
                           "INJECTION, POWDER, LYOPHILIZED, FOR SUSPENSION": "INJECTION",
                           "INJECTION, POWDER, LYOPHILIZED, FOR SUSPENSION, EXTENDED RELEASE": "INJECTION",
                           "INJECTION, SOLUTION": "INJECTION", "INJECTION, SOLUTION, CONCENTRATE": "INJECTION",
                           "INJECTION, SUSPENSION": "INJECTION", "INJECTION, SUSPENSION, EXTENDED RELEASE": "INJECTION",
                           "INJECTION, SUSPENSION, LIPOSOMAL": "INJECTION",
                           "INJECTION, SUSPENSION, SONICATED": "INJECTION", "INSERT": "INSERT",
                           "INSERT, EXTENDED RELEASE": "INSERT", "INTRAUTERINE DEVICE": "INTRAUTERINE DEVICE",
                           "IRRIGANT": "IRRIGANT", "JELLY": "JELLY", "KIT": "KIT", "LINIMENT": "LINIMENT",
                           "LIPSTICK": "LIPSTICK", "LIQUID": "LIQUID", "LIQUID, EXTENDED RELEASE": "LIQUID",
                           "LOTION": "LOTION", "LOTION, AUGMENTED": "LOTION", "LOTION/SHAMPOO": "LOTION/SHAMPOO",
                           "LOZENGE": "LOZENGE", "MOUTHWASH": "MOUTHWASH", "NOT APPLICABLE": "NOT APPLICABLE",
                           "OIL": "OIL", "OINTMENT": "OINTMENT", "OINTMENT, AUGMENTED": "OINTMENT", "PASTE": "PASTE",
                           "PASTE, DENTIFRICE": "PASTE", "PASTILLE": "PASTILLE", "PATCH": "PATCH",
                           "PATCH, EXTENDED RELEASE": "PATCH",
                           "PATCH, EXTENDED RELEASE, ELECTRICALLY CONTROLLED": "PATCH", "PELLET": "PELLET",
                           "PELLET, IMPLANTABLE": "PELLET", "PELLETS, COATED, EXTENDED RELEASE": "PELLETS",
                           "PILL": "PILL", "PLASTER": "PLASTER", "POULTICE": "POULTICE", "POWDER": "POWDER",
                           "POWDER, DENTIFRICE": "POWDER", "POWDER, FOR SOLUTION": "POWDER",
                           "POWDER, FOR SUSPENSION": "POWDER", "POWDER, METERED": "POWDER", "RING": "RING",
                           "RINSE": "RINSE", "SALVE": "SALVE", "SHAMPOO": "SHAMPOO", "SHAMPOO, SUSPENSION": "SHAMPOO",
                           "SOAP": "SOAP", "SOLUTION": "SOLUTION", "SOLUTION, CONCENTRATE": "SOLUTION",
                           "SOLUTION, FOR SLUSH": "SOLUTION", "SOLUTION, GEL FORMING / DROPS": "SOLUTION",
                           "SOLUTION, GEL FORMING, EXTENDED RELEASE": "SOLUTION", "SOLUTION/ DROPS": "SOLUTION",
                           "SPONGE": "SPONGE", "SPRAY": "SPRAY", "SPRAY, METERED": "SPRAY",
                           "SPRAY, SUSPENSION": "SPRAY", "STICK": "STICK", "STRIP": "STRIP",
                           "SUPPOSITORY": "SUPPOSITORY", "SUPPOSITORY, EXTENDED RELEASE": "SUPPOSITORY",
                           "SUSPENSION": "SUSPENSION", "SUSPENSION, EXTENDED RELEASE": "SUSPENSION",
                           "SUSPENSION/ DROPS": "SUSPENSION", "SWAB": "SWAB", "SYRUP": "SYRUP",
                           "SYSTEM": "SYSTEM", "TABLET": "TABLET", "TABLET, CHEWABLE": "TABLET",
                           "TABLET, CHEWABLE, EXTENDED RELEASE": "TABLET", "TABLET, COATED": "TABLET",
                           "TABLET, COATED PARTICLES": "TABLET", "TABLET, DELAYED RELEASE": "TABLET",
                           "TABLET, DELAYED RELEASE PARTICLES": "TABLET", "TABLET, EFFERVESCENT": "TABLET",
                           "TABLET, EXTENDED RELEASE": "TABLET", "TABLET, FILM COATED": "TABLET",
                           "TABLET, FILM COATED, EXTENDED RELEASE": "TABLET", "TABLET, FOR SOLUTION": "TABLET",
                           "TABLET, FOR SUSPENSION": "TABLET", "TABLET, MULTILAYER": "TABLET",
                           "TABLET, MULTILAYER, EXTENDED RELEASE": "TABLET", "TABLET, ORALLY DISINTEGRATING": "TABLET",
                           "TABLET, ORALLY DISINTEGRATING, DELAYED RELEASE": "TABLET", "TABLET, SOLUBLE": "TABLET",
                           "TABLET, SUGAR COATED": "TABLET", "TABLET WITH SENSOR": "TABLET",
                           "TAMPON": "TAMPON", "TAPE": "TAPE", "TINCTURE": "TINCTURE", "TROCHE": "TROCHE",
                           "WAFER": "WAFER"}
standard_routename = ["AURICULAR (OTIC)", "BUCCAL", "CONJUNCTIVAL", "CUTANEOUS", "DENTAL", "ELECTRO-OSMOSIS",
                      "ENDOCERVICAL", "ENDOSINUSIAL", "ENDOTRACHEAL", "ENTERAL", "EPIDURAL", "EXTRA-AMNIOTIC",
                      "EXTRACORPOREAL", "HEMODIALYSIS", "INFILTRATION", "INTERSTITIAL", "INTRA-ABDOMINAL",
                      "INTRA-AMNIOTIC", "INTRA-ARTERIAL", "INTRA-ARTICULAR", "INTRABILIARY", "INTRABRONCHIAL",
                      "INTRABURSAL", "INTRACANALICULAR", "INTRACARDIAC", "INTRACARTILAGINOUS", "INTRACAUDAL",
                      "INTRACAVERNOUS", "INTRACAVITARY", "INTRACEREBRAL", "INTRACISTERNAL", "INTRACORNEAL",
                      "INTRACORONAL, DENTAL", "INTRACORONARY", "INTRACORPORUS CAVERNOSUM", "INTRACRANIAL",
                      "INTRADERMAL", "INTRADISCAL", "INTRADUCTAL", "INTRADUODENAL", "INTRADURAL", "INTRAEPICARDIAL",
                      "INTRAEPIDERMAL", "INTRAESOPHAGEAL", "INTRAGASTRIC", "INTRAGINGIVAL", "INTRAHEPATIC",
                      "INTRAILEAL", "INTRALESIONAL", "INTRALINGUAL", "INTRALUMINAL", "INTRALYMPHATIC", "INTRAMAMMARY",
                      "INTRAMEDULLARY", "INTRAMENINGEAL", "INTRAMUSCULAR", "INTRANODAL", "INTRAOCULAR", "INTRAOMENTUM",
                      "INTRAOVARIAN", "INTRAPERICARDIAL", "INTRAPERITONEAL", "INTRAPLEURAL", "INTRAPROSTATIC",
                      "INTRAPULMONARY", "INTRARUMINAL", "INTRASINAL", "INTRASPINAL", "INTRASYNOVIAL", "INTRATENDINOUS",
                      "INTRATESTICULAR", "INTRATHECAL", "INTRATHORACIC", "INTRATUBULAR", "INTRATUMOR", "INTRATYMPANIC",
                      "INTRAUTERINE", "INTRAVASCULAR", "INTRAVENOUS", "INTRAVENTRICULAR", "INTRAVESICAL",
                      "INTRAVITREAL", "IONTOPHORESIS", "IRRIGATION", "LARYNGEAL", "NASAL", "NASOGASTRIC",
                      "NOT APPLICABLE", "OCCLUSIVE DRESSING TECHNIQUE", "OPHTHALMIC", "ORAL", "OROPHARYNGEAL",
                      "PARENTERAL", "PERCUTANEOUS", "PERIARTICULAR", "PERIDURAL", "PERINEURAL", "PERIODONTAL", "RECTAL",
                      "RESPIRATORY (INHALATION)", "RETROBULBAR", "SOFT TISSUE", "SUBARACHNOID", "SUBCONJUNCTIVAL",
                      "SUBCUTANEOUS", "SUBGINGIVAL", "SUBLINGUAL", "SUBMUCOSAL", "SUBRETINAL", "TOPICAL", "TRANSDERMAL",
                      "TRANSENDOCARDIAL", "TRANSMUCOSAL", "TRANSPLACENTAL", "TRANSTRACHEAL", "TRANSTYMPANIC",
                      "URETERAL", "URETHRAL", "VAGINAL"]
standard_marketingcategoryname = ["ANADA", "ANDA", "Approved Drug Product Manufactured Under Contract", "BLA",
                                  "Bulk ingredient", "Bulk Ingredient For Animal Drug Compounding",
                                  "Bulk Ingredient For Human Prescription Compounding", "Conditional NADA", "Cosmetic",
                                  "Dietary Supplement", "Drug for Further Processing", "Exempt device", "Export only",
                                  "Humanitarian Device Exemption", "IND", "Medical Food",
                                  "Legally Marketed Unapproved New Animal Drugs for Minor Species", "NADA", "NDA",
                                  "NDA authorized generic", "OTC Monograph Drug Product Manufactured Under Contract",
                                  "OTC monograph final", "OTC monograph not final", "Premarket Application",
                                  "Premarket Notification", "Unapproved drug for use in drug shortage",
                                  "Unapproved drug other", "Unapproved Drug Product Manufactured Under Contract",
                                  "Unapproved homeopathic", "Unapproved medical gas"]
standard_deaschedule = ["CI", "CII", "CIII", "CIV", "CV"]
standard_ndcexcludeflag = ["N"]

target_encoding = 'utf-8'
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


def date_convert(table, dc):
    for c in dc:
        table[c] = pd.to_datetime(table[c], errors='coerce', format='%Y%m%d')


def date_convert_back(table, dc):
    for c in dc:
        for index, _ in table[c].items():
            table[c][index] = pd.Timestamp(table[c][index])


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
    product = pd.read_csv(encoded_product_file, sep=separ, encoding=target_encoding)

    # # Convert back date from string to timestamp
    # TODO date conversion cause conflict when loading back data
    # time_methode(date_convert_back, **dict(table=product, dc=date_cols))

    # Populate onehot encoders dictionnary
    for header in product_headers_to_encode:
        enc_dic[header] = pickle.load(open(encoder_dir + '{}_data_encoder.pkl'.format(header), 'rb'))
else:
    product = original_product_data

if package_encode_file_exist:
    print('Loading encoded package data from existing file...')
    package = pd.read_csv(encoded_package_file, sep=separ, encoding=target_encoding)

    # # Populate onehot encoders dictionnary
    # for header in package_headers_to_encode:
    #     enc_dic[header] = pickle.load(open(encoder_dir + '{}_data_encoder.pkl'.format(header), 'rb'))
else:
    package = original_package_data

# Make everything lower characters in both tables
df_to_lower(product)
df_to_lower(package)

# %%
"""
# 1. Auscultation
Nous avons déjà prétraitées les données (passage en minuscules des données textuelles) afin de minimiser l'inconsistance
entre les valeurs.

## Etude des données du fichier 'package'
"""

# %%
print('Assessing completeness of packaging data table')
assert_table_completeness(package)

# %%
"""
La colonne PRODUCTID ne présente pas de valeurs manquantes. Celle-ci fournit les valeurs concaténées de 
code produit NDC et de l'identifiant SPL. Cependant, la colonne PRODUCTNDC présente quant à elle 1500 valeurs manquantes
. On remarque également des valeurs aberrantes dans ses valeurs.

Les valeurs manquantes des colonnes STARTMARKETINGDATE et ENDMARKETINGDATE sont plus nombreuses mais semblent être non 
bloquantes. Ces deux dernières colonnes sont de type date.

La colonne PACKAGEDESCRIPTION est présentée sous forme de phrase et contient de multiples informations: le type de 
volume, sa valeur et son unité. S'il existe plusieurs contenants pour un objet, ils sont concaténés par un séparateur 
'>' de manière hiérarchique.

Les colonnes NDC_EXCLUDE_FLAG et SAMPLE_PACKAGE, présentant peu de valeurs différentes, et sont facilement 
traitables numériquement.
"""

# %%
"""
## Etude des données du fichier 'product'
"""

# %%
print('Assessing completeness product data table')
assert_table_completeness(product)

# %%
"""
On remarque que la colonne PRODUCTID présente 1560 valeurs manquantes. La colonne PRODUCTNDC quant à elle présente 
certaines valeurs aberrantes.
"""

# %%
print(product['PRODUCTNDC'][159:161])

# %%
"""
Dans la colonne PRODUCTTYPENAME, on remarque 7 valeurs possibles textuelles catégorielles dans cette colonne et aucune
valeur manquante. Cette colonne sera donc facilement numérisable. 

La colonne PROPRIETARYNAME dispose d'un grand nombre de valeurs différentes, de type textuelle. Ces valeurs sont assez 
variables (phrase, simple mot) décrivant plus ou moins le produit. 
La colonne PROPRIETARYNAMESUFFIX est du même type que PROPRIETARYNAME, cependant elle présente beaucoup de valeurs 
nulles, et apporte des informations variantes aux objets. La documentation précise ne reconnait pas de standard.

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

Il appert également que, toutes colonnes confondues, les valeurs uniques sont parfois simplement des permutations de
"sous-valeurs" séparées par des charctères de ponctuation ou des charctères spéciaux. Il convient donc de
décortiquer d'avantage ces données pour réduire le nombre de catégories au maximum pour les dimensions concernées.
Par exemple, les 1932 valeurs uniques de la colonne ROUTENAME peuvent en fait être réduite à 180 "vraies" valeurs uniques
lorsqu'on fait abstraction des permutations:
"""

# %%
product_unique_values = get_decomposed_uniques(original_product_data, 'ROUTENAME')
print(product_unique_values)

# %%
product_unique_values = print(product['LABELERNAME'][7252:7255])
print(product_unique_values)

# %%
print(product['LABELERNAME'][7252:7255])

# %%
"""
La colonne SUBSTANCENAME présente des données du standard FDA, celui-ci est composé de 108 227 catégories différentes.
Chaque objet peut présenter plusieurs catégories, la représentation de valeurs multiples est consistante via le 
séparateur ';'. Cela pourrait expliquer le nombre important de valeurs différentes. Le nombre de valeurs manquantes
est important et les valeurs seront difficilement complétables.

Les colonnes ACTIVE_NUMERATOR_STRENGTH et ACTIVE_INGRED_UNIT présentent des valeurs liées à la substance. 
Il existe des valeurs multiples et une consistance dans leur représentation via le séparateur ';'. 
Le nombre de valeurs manquantes est égal pour les deux colonnes. Elles paraissent assez facilement numérisables mais 
difficilement complétables.

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
print(product['NDC_EXCLUDE_FLAG'].value_counts())

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
## Table 'product'
Il y a de nombreux points à vérifier pour la table 'product'. 
Tout d'abord, on peut s'intéresser aux colonnes date STARTMARKETINGDATE, ENDMARKETINGDATE et 
LISTING_RECORD_CERTIFIED_THROUGH. 
On se rend compte de l'existence de données aberrantes que l'on décide d'ignorer et de supprimer leur valeur.
"""


# %%
# conversion to datetime format
def date_convert(table, dc):
    for c in dc:
        table[c] = pd.to_datetime(table[c], errors='coerce', format='%Y%m%d')


# %%

# TODO date conversion cause conflict when loading back data
date_cols = ['STARTMARKETINGDATE', 'ENDMARKETINGDATE', 'LISTING_RECORD_CERTIFIED_THROUGH']
if not product_encode_file_exist:
    date_convert(product, date_cols)

# %%
"""
Aussi, il existerait une incohérence si la date de fin de mise sur le marché est moins récente que la date de début de 
mise sur le marché.
"""

# %%
# compare STARTMARKETINGDATE and ENDMARKETINGDATE
nb = product[product['STARTMARKETINGDATE'] > product['ENDMARKETINGDATE']].shape[0]
print(f"Nombre d'incohérences entre STARTMARKETINGDATE et ENDMARKETINGDATE: {nb}")

# %%
"""
La colonne LISTING_RECORD_CERTIFIED_THROUGH permet de savoir si la certification du produit est expiré. On considère 
donc que le produit n'est plus à jour (et donc à supprimer de notre dataset) si la date précisée dans cette 
colonne est passée. En l'occurence, il n'y a uncun produit dont la date d'échéance est antérieure au 31 décembre 2021.
"""

# %%
# TODO #1 there is no LISTING_RECORD_CERTIFIED_THROUGH prior to 31-12-2021 and date conversion cause a conflict when loading back files...
# TODO #2 check fix
nb = (product['LISTING_RECORD_CERTIFIED_THROUGH'] < datetime.now()).sum()
print(f'Nombre d\'incohérences pour l\'attribut LISTING_RECORD_CERTIFIED_THROUGH: {nb}')

# %%
"""
La colonne NDC_EXCLUDE_FLAG ne devrait présenter que des valeurs de la catégorie 'N' pour notre dataset, comme le 
précise la documentation FDA. On le vérifie simplement:
"""

# %%
print(product['NDC_EXCLUDE_FLAG'].value_counts())

# %%
"""
Les colonnes SUBSTANCENAME, ACTIVE_NUMERATOR_STRENGTH et ACTIVE_INGRED_UNIT présentent des valeurs multiples liées. Leur
nombre dans chacune des colonnes doit donc être égal. On les vérifie deux à deux.
"""

# %%
values_count = lambda row, col: len(re.sub(r"(\().*?;?.*?(\))", '', row[col]).split(';')) if isinstance(row[col],
                                                                                                        str) else 0
check = lambda row: values_count(row, "SUBSTANCENAME") == values_count(row, "ACTIVE_NUMERATOR_STRENGTH") \
                    == values_count(row, "ACTIVE_INGRED_UNIT")
nb_valid = len(product.apply(check, axis=1))
print(f"Nombre d'incohérences entre ces 3 colonnes: {product.shape[0] - nb_valid}")

# TODO: print outliers PRODUCTNDC dans product

# %%
"""
La colonne PRODUCTNDC présente certaines valeurs aberrantes que nous décidons de récupérer de la première
partie de la valeur du PRODUCTID associée. En effet, celuLISTING_RECORD_CERTIFIED_THROUGHi-ci étant un duplicata, celui-ci peut être considéré comme 
correct.
"""


# %%


def replace_outliers_productndc(table):
    outliers = table['PRODUCTNDC'][~table['PRODUCTNDC'].str.contains(r'\d{4,5}-\d{3,4}', regex=True, na=False)]
    id_outliers = table.iloc[outliers.index.values.tolist()]['PRODUCTID']
    for (io, i) in zip(id_outliers, outliers.index.values.tolist()):
        if not pd.isnull(io):
            table.at[i, 'PRODUCTNDC'] = re.match('(^[^_]+)', io).group(0)


replace_outliers_productndc(product)

# %%
"""
Certaines colonnes représentent des standards FDA, afin d'assurer aucune incohérence dans leurs valeurs, 
nous décidons de vérifier que leurs valeurs sont incluses dans les standards fournis par la FDA (disponible 
https://www.fda.gov/industry/fda-resources-data-standards/structured-product-labeling-resources). 
On s'intéressera donc aux colonnes: DOSAGEFORMNAME, ROUTENAME, MARKETINGCATEGORYNAME, DEASCHEDULE, NDC_EXCLUDE_FLAG 
Les colonnes SUBSTANCENAME et PHARM_CLASSES représentent également des standards FDA, cependant, le nombre de valeurs
possibles fournis par la FDA est extrêment important. Nous décidons, par mesure de possibilité, ne pas les traiter.
"""


# %%

def check_categories(table, column_name, standard):
    categories = pd.Series(table[column_name].unique()).dropna()
    lowercase_standard = map(str.lower, pd.Series(standard))
    return categories.isin(lowercase_standard).any().any()


def check_dict_categories(table, column_name, standard):
    categories = pd.Series(table[column_name].unique()).dropna()
    lowercase_standard = dict((k.lower(), v.lower()) for k, v in standard.items())
    return categories.isin(list(lowercase_standard.values())).any().any()


# %%
cols = ['DEASCHEDULE', 'NDC_EXCLUDE_FLAG', 'ROUTENAME', 'MARKETINGCATEGORYNAME']
standards = [standard_deaschedule, standard_ndcexcludeflag, standard_routename, standard_marketingcategoryname]
for (col_name, stand) in zip(cols, standards):
    check = check_categories(product, col_name, stand)
    print(f'Toutes les valeurs de la colonne {col_name} correspondent au stardard FDA: {check}')

check = check_dict_categories(product, 'DOSAGEFORMNAME', standard_dosageformname)
print(f'Toutes les valeurs de la colonne DOSAGEFORMNAME correspondent au stardard FDA: {check}')


# %%

def check_format_standard(table, cols, reg):
    for (c, r) in zip(cols, reg):
        check = table[c].str.contains(r, regex=True, na=True).sum() == table.shape[0]
        print(f'La colonne {c} répond au format de la standardisation: {check}')


check_format_standard(product, ['PRODUCTNDC', 'PRODUCTID'], [r'\d{4,5}-\d{3,4}', r'\d{4,5}-\d{3,4}_[A-Za-z0-9\-]+'])

# %%
"""
## Table 'package'

Traitement des colonnes STARTMARKETINGDATE et ENDMARKETINGDATE similairement à la table 'product'.
"""

# %%

date_cols = ['STARTMARKETINGDATE', 'ENDMARKETINGDATE']
# TODO data conversion causes conflist when loading data back
if not package_encode_file_exist:
    date_convert(package, date_cols)

# compare STARTMARKETINGDATE and ENDMARKETINGDATE
nb = package[package['STARTMARKETINGDATE'] > package['ENDMARKETINGDATE']].shape[0]
print(f"Nombre d'incohérences entre STARTMARKETINGDATE et ENDMARKETINGDATE: {nb}")
print(package[package['STARTMARKETINGDATE'] > package['ENDMARKETINGDATE']][['STARTMARKETINGDATE', 'ENDMARKETINGDATE']])

# %%
"""
Ces anomalies semblent être valeurs aberrantes, et pourraient résulter d'une erreur manuelle.
On décide de les remplacer par des valeurs nulles.
"""

# %%
package[package['STARTMARKETINGDATE'] > package['ENDMARKETINGDATE']] = pd.NaT

# %%
"""
La colonne NDC_EXCLUDE_FLAG représente un stardard FDA que l'on vérifie comme pour la table 'product'.

"""

# %%
cols = ['NDC_EXCLUDE_FLAG']
standards = [standard_ndcexcludeflag]
for (col_name, stand) in zip(cols, standards):
    check = check_categories(package, col_name, stand)
    print(f'Toutes les valeurs de la colonne {col_name} correspondent au stardard FDA: {check}')

# %%
"""
La colonne PRODUCTNDC présente également des données aberrantes du même type que l'on avait trouvé dans la table product
.
"""

# %%

replace_outliers_productndc(package)

# %%
"""
Les colonnes PRODUCTID, PRODUCTNDC et NDCPACKAGECODE suivent un format spécifié :
- PRODUCTNDC doit répondre à une structure de digits telle que {3-5}, {3-4}, {4-4}, {4-5}.
- PRODUCTID concatène la valeur du PRODUCTNDC et un identifiant SPL séparé par un '_'.
- NDCPACKAGECODE concatène la valeur du PRODUCTNDC et un code segment de 2 digits séparé par '-'.
"""

# %%
cols = ['PRODUCTNDC', 'PRODUCTID', 'NDCPACKAGECODE']
reg = [r'\d{4,5}-\d{3,4}', r'\d{4,5}-\d{3,4}_[A-Za-z0-9\-]+', r'\d{4,5}-\d{3,4}-\d{1,2}']
check_format_standard(package, cols, reg)

# %%
"""
On remarque que les valeurs de la colonne NDCPACKAGECODE ne répondent pas toutes au format de la standardisation.
"""

# %%
val_bad_formatting = package[~package['NDCPACKAGECODE'].str.contains(r'\d{4,5}-\d{3,4}-\d{1,2}', regex=True, na=False)]
val_bad_formatting['NDCPACKAGECODE']

# %%
"""
On remarque effectivement que certaines valeurs sont incorrectes et ne correspondent pas à des code package. On sait que
l'attribut PACKAGEDESCRIPTION contient, pour le premier contenant, la valeur du code package. On va pouvoir le récupérer
de cette manière.
"""

# %%

correct_val = val_bad_formatting['PACKAGEDESCRIPTION'].str.extract(r'\((.*?)\).*')
for (i, v) in correct_val.iterrows():
    package.at[i, 'NDCPACKAGECODE'] = v[0]

# %%
"""
# 4. Données manquantes
## Table 'package'
Il y a des valeurs manquantes dans les colonnes NDCPACKAGECODE et PRODUCTNDC.
Pour la colonne NDCPACKAGECODE, on peut récupérer cette information dans PACKAGEDESCRIPTION. Celle-ci se retrouve
concaténée et associée au premier contenant du produit.
"""


# %%

def replace_missing_values(table, col_name_1, col_name_2, regex):
    missing_values = package.iloc[np.where(pd.isnull(table[col_name_1]))]
    values = missing_values[col_name_2].str.extract(regex)
    for index, row in values.iterrows():
        package.loc[index, col_name_1] = row[0]


if not package_encode_file_exist:
    replace_missing_values(package, 'NDCPACKAGECODE', 'PACKAGEDESCRIPTION', r'\((.*?)\).*')

# %%
if not package_encode_file_exist:
    replace_missing_values(package, 'PRODUCTNDC', 'NDCPACKAGECODE', r'^([\w]+-[\w]+)')

# %%

assert_table_completeness(package)

# %%
"""
Il existe des valeurs manquantes pour les colonnes 'STARTMARKETINGDATE' et 'ENDMARKETINGDATE' dans la table 'package'
mais on choisit de ne pas les compléter car on ne peut effectuer d'estimation précise.
"""

# %%
"""
## Table 'product'
Un nombre conséquent de colonnes présente des valeurs manquantes. On choisit de seulement traiter la colonne PRODUCTID,
qui sera utile lors de l'intégration des deux tables. Les autres colonnes présentent des valeurs très difficiles à
estimer.

Pour la colonne PRODUCTID, on va devoir utiliser la table package afin de pouvoir récupérer les bonnes valeurs. En
effet, les deux tables présentent les mêmes attributs PRODUCTID et PRODUCTNDC, on peut donc se baser là-dessus pour
retrouver les bonnes informations. Les valeurs de PRODUCTNDC étant quasiment toutes uniques, on peut considérer son
utilisation.
"""

# %%
missing_val = product[product['PRODUCTID'].isnull()]['PRODUCTNDC']
values = package.loc[package['PRODUCTNDC'].isin(missing_val)]['PRODUCTID'].drop_duplicates()
for (v, i) in zip(values, missing_val.index.values.tolist()):
    product.at[i, 'PRODUCTID'] = v

# %%
assert_table_completeness(package)

# %%
"""
# 5. Duplicata des objets
## Table package
On s'intéresse à la colonne NDCPACKAGECODE afin de déterminer les duplicata. En effet plusieurs packages peuvent être
associés à un produit (PRODUCTID), cependant les code package doivent être uniques.
"""

# %%

package_duplicated = package[package.duplicated(['NDCPACKAGECODE'], keep=False)]

# %%
"""
Celle-ci montre que nous avons 2 objets présentant un duplicata de code package. On les étudie deux à deux.
### Premier duplicata
"""

# %%
# NaN values are set to 0 to not compromise test
print(package_duplicated.fillna(0).iloc[0] == package_duplicated.fillna(0).iloc[1])

# %%
"""
On remarque que seules les valeurs de PRODUCTID sont différentes entre elles.
On confirme que ces deux valeurs de PRODUCTID sont également présentes dans la table product.
"""

# %%
d = product.loc[product['PRODUCTID'].isin(package_duplicated.iloc[0:2]['PRODUCTID'])]
print(d.fillna(0).iloc[0] == d.fillna(0).iloc[1])

# %%
"""
Dans la table product, ces deux objets sont également différentes que pour l'attribut PRODUCTID.
On choisit donc d'éliminer un des duplicata dans les deux tables (le premier par simplicité).
"""

# %%

product = product.drop(d.index[1])
package = package.drop(package_duplicated.index[1])

# %%
"""
### Deuxième duplicata
"""

# %%
print(package_duplicated.fillna(0).iloc[2] == package_duplicated.fillna(0).iloc[3])
print(package_duplicated.iloc[2:4]['STARTMARKETINGDATE'])

# %%
"""
On remarque que ces objets présentant des NDCPACKAGECODE dupliqués ont des valeurs de STARTMARKETINGDATE divergent.
On décide de comparer avec les objets correspondants dans la table product.
"""

# %%

d = product.loc[product['PRODUCTID'].isin(package_duplicated.iloc[2:3]['PRODUCTID'])]
print(d['STARTMARKETINGDATE'])

# %%
"""
La valeur de STARTMARKETINGDATE confirme un des objets dupliqués dans package. On décide alors d'éliminer l'objet
présentant une valeur différente de STARTMARKETINGDATE dans package.
"""

# %%

package = package.drop(package_duplicated.index[3])

# %%
d = package[package.duplicated(['NDCPACKAGECODE'], keep=False)]
print(f'Nombre d\'objets dupliqués dans package par rapport à NDCPACKAGECODE: {len(d)} ')

# %%
"""
## Table product
Pour la table product, on s'intéresse aux duplicata de l'attribut PRODUCTID qui devrait être unique.
"""

# %%
d = product[product.duplicated(['PRODUCTID'], keep=False)]
print(f'Nombre d\'objets dupliqués dans product par rapport à PRODUCTID: {len(d)}')

# %%
"""
Les valeurs de l'attribut PRODUCTNDC devraient également être uniques entre elles.
"""
# %%

d = product[product.duplicated(['PRODUCTNDC'], keep=False)]
print(f'Nombre d\'objets dupliqués dans product par rapport à PRODUCTNDC: {len(d)}')
# %%
"""
# 6. Intégration des tables
On se rend compte qu'un objet dans la table package ne dispose pas de son équivalent dans la table product. 
"""

# %%
d = package[~package['PRODUCTID'].isin(product['PRODUCTID'])]['PRODUCTID'].values[0]
print(f'Objet de package dont PRODUCTID est manquant dans product: {d}')

# %%
"""
On décide d'éliminer cet objet de package lors du merge, car celui-ci ne servira pas lors de l'entraînement pour le
modèle de prédiction.
"""

# %%
unified_tables = pd.merge(product, package, on='PRODUCTID')

print(unified_tables)
print(assert_table_completeness(unified_tables))

# %%
"""
Les colonnes appartenant dans les deux tables unifiées sont donc à traiter. 
Pour l'attribut STARTMARKETINGDATE, on regarde que la table originale package présentait 243 valeurs manquantes, alors 
que la table originale product n'en contenait aucune. Cela se reporte donc sur les colonnes  STARTMARKETINGDATE_x et 
STARTMARKETINGDATE_y, où la deuxième présente ces valeurs manquantes. On choisit alors aisément d'éliminer 
STARTMARKETINGDATE_y au profit de l'utilisation de STARTMARKETINGDATE_x.
Pour l'attribut ENDMARKETINGDATE, on remarque que c'est l'inverse. Par souci de logique, on choisit donc d'éliminer 
ENDMARKETINGDATE_x au profit de l'utilisation de ENDMARKETINGDATE_y.
"""

# %%

unified_tables = unified_tables.drop(['STARTMARKETINGDATE_y'], axis=1)

# %%

unified_tables = unified_tables.drop(['ENDMARKETINGDATE_x'], axis=1)
print(assert_table_completeness(unified_tables))

# %%
"""
On se souvient que tous les objets des attributs NDC_EXCLUDE_FLAG pour les deux tables sont établies à la même valeur: 'n
'. Le choix de la colonne à garder entre NDC_EXCLUDE_FLAG_x et NDC_EXCLUDE_FLAG_y est donc peu important.
"""

# %%
unified_tables = unified_tables.drop(['NDC_EXCLUDE_FLAG_y'], axis=1)
print(assert_table_completeness(unified_tables))

# %%
"""
On s'intéresse maintenant à l'attribut PRODUCTNDC des deux tables:
"""

# %%

check = (unified_tables['PRODUCTNDC_x'] == unified_tables['PRODUCTNDC_y']).all()
print(f'Les valeurs de l\'attribut PRODUCTNDC_x est égal ligne par ligne aux valeurs de l\'attribut PRODUCTNDC_y: '
      f'{check}')

# %%
"""
On peut ainsi éliminer l'une ou l'autre colonne sans soucis.
"""

# %%

unified_tables = unified_tables.drop(['PRODUCTNDC_y'], axis=1)
print(assert_table_completeness(unified_tables))

# %%

unified_tables = unified_tables.rename(columns={'STARTMARKETINGDATE_x': 'STARTMARKETINGDATE',
                                                'ENDMARKETINGDATE_y': 'ENDMARKETINGDATE',
                                                'NDC_EXCLUDE_FLAG_x': 'NDC_EXCLUDE_FLAG',
                                                'PRODUCTNDC_x': 'PRODUCTNDC'})

# %%
"""
Notre dataframe intitulé unified_tables possède maintenant des attributs uniques résumant au mieux les données des
 tables product et package originales.
"""

# %%
"""
7. Proposition d'un ensemble d'attributs éliminant redondance 

L'attribut PRODUCTID concatène les valeur du code produit NDC et de l'identifiant du SPL. On peut donc éliminer 
l'information code produit NDC (déjà présent dans l'attribut PRODUCTNDC) et ainsi spécifier un attribut pour 
l'identifiant du SPL que l'on nommera SPLID.

L'attribut PRODUCTNDC concatène les valeurs du code label et du code produit de segment. Ces valeurs ne sont pas 
dupliquées au sein de notre ensemble d'attributs. On peut considérer les garder également concaténées. 

L'attribut APPLICATIONNUMBER concatène les valeurs du nom de la catégorie marketing et de son nombre d'application.
Etant donné que l'attribut MARKETINGCATEGORYNAME spécifie déjà uniquement le nom de la catégorie marketing, on 
considère acceptable d'éliminer la valeur du nom de la catégorie marketing de l'attribut APPLICATIONNUMBER. Le nom de
cet attribut semble correspondre à nos nouvelles valeurs.

L'attribut NDCPACKAGECODE concatène les valeurs du code label, du code produit de segment et code package de segment.
Comme précédemment énoncé, on dispose déjà du code label et du code produit de segment dans l'attribut PRODUCTNDC. 
On peut donc éliminer ces valeurs de l'attribut NDCPACKAGECODE afin de garder seulement le code package de segment. 
Cet nouvel attribut sera nommé PACKAGECODE.

L'attribut PACKAGEDESCRIPTION intègre la description de la taille et du type de package pour chacun de ses contenants.
On y retrouve également des valeurs de NDCPACKAGECODE que l'on peut éliminer.
"""


# %%


def remove_content_from_attribute(attribute, regex):
    unified_tables[attribute] = unified_tables[attribute].replace(to_replace=regex, value='', regex=True)


# %%
cols = ['PRODUCTID', 'NDCPACKAGECODE', 'PACKAGEDESCRIPTION', 'APPLICATIONNUMBER']
reg = [r'\d{4,5}-\d{3,4}_', r'\d{4,5}-\d{3,4}-', r'\(\d{4,5}-\d{3,4}-\d{2}\) ', r'[a-zA-Z]']

for (c, r) in zip(cols, reg):
    remove_content_from_attribute(c, r)

# %%
print('Voici donc notre nouvel ensemble d\'attribut:')
print(unified_tables.head())

# %%
"""
8. Proposition d'un ensemble d'attributs pour la prédiction des classes pharmaceutiques
"""

# %%

# TODO: paragraphe répondant plus à la question 8
"""
Puisque l'objectif final est un modèle de classification entre les classes pharmaceutique, il n'est pas pertinent de
conserver les colonnes de la table où il manque beaucoup de valeurs, ou encore des valeurs qui n'ont aucun lien logique
avec la class pharmaceutiquel. On peut donc laisser tomber les colonnes PROPRIETARYNAMESUFFIX, ENDMARKETINGDATE_x,
APPLICATIONNUMBER, DEASCHEDULE, ENDMARKETINGDATE_y, NDC_EXCLUDE_FLAG_x, LISTING_RECORD_CERTIFIED_THROUGH, SAMPLE_PACKAGE,
NDC_EXCLUDE_FLAG_y
"""

# %%
"""
# Transformation en données numériques (après question 8)
## Encodage onehot
"""

# %%
# Call and time onehot encoding for all predefined columns
# if not os.path.isdir(encoder_dir):
#     os.mkdir(encoder_dir)
# if not product_encode_file_exist:
#     for header in product_headers_to_encode:
#         enc_dic[header] = time_methode(onehot_encode, header, **(dict(table=product, header=header)))
#         pickle.dump(enc_dic[header], open(encoder_dir + '{}_data_encoder.pkl'.format(header), 'wb'),
#                     pickle.HIGHEST_PROTOCOL)
#
# # if not package_encode_file_exist:
# #     for header in package_headers_to_encode:
# #         enc_dic[header] = time_methode(onehot_encode, header, **(dict(table=package, header=header)))
# #         pickle.dump(enc_dic[header], open(encoder_dir + '{}_data_encoder.pkl'.format(header), 'wb'),
# #                     pickle.HIGHEST_PROTOCOL)
#
# if not os.path.isdir(encoding_dir):
#     {os.mkdir(encoding_dir)}
# # Prints out encding of each category for a given column in a txt file
# for header, enc in enc_dic.items():
#     file = open(encoding_dir + 'Encoding_{}.txt'.format(header), 'w')
#     for category in enc.categories_[0]:
#         tmp_str = str(enc.transform([[category]]).toarray())
#         tmp_str = category + ' ' * (40 - len(category)) + tmp_str.replace('\n', '\n' + ' ' * 40) + '\n'
#         file.write(tmp_str)
#     file.close()
#
# # Save transformed data to file
# if not product_encode_file_exist:
#     time_methode(product.to_csv, **(dict(path_or_buf=encoded_product_file,
#                                          index=False,
#                                          sep=separ,
#                                          encoding=target_encoding,
#                                          quoting=csv.QUOTE_NONNUMERIC)))
#
# if not product_encode_file_exist:
#     time_methode(package.to_csv, **(dict(path_or_buf=encoded_package_file,
#                                          index=False,
#                                          sep=separ,
#                                          encoding=target_encoding,
#                                          quoting=csv.QUOTE_NONNUMERIC)))

# %%
"""
## Résultats
"""

# %%
# print('Encoded product data:')
# print(product)
# product
#
# # %%
# print('Encoded packaging data:')
# print(package)
# package
