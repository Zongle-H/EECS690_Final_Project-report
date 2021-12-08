```python
import pandas as pd 
from pandas import Series, DataFrame 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
import xgboost
from sklearn import metrics

```


```python
df = pd.read_csv(r"C:\Users\zl262\OneDrive\Desktop\input.file\dataset.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Player</th>
      <th>Opponent</th>
      <th>Position</th>
      <th>Champion</th>
      <th>Kills</th>
      <th>Deaths</th>
      <th>Assists</th>
      <th>Creep Score</th>
      <th>Gold Earned</th>
      <th>Champion Damage Share</th>
      <th>Kill Participation</th>
      <th>Wards Placed</th>
      <th>Wards Destroyed</th>
      <th>Ward Interactions</th>
      <th>Dragons For</th>
      <th>Dragons Against</th>
      <th>Barons For</th>
      <th>Barons Against</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UOL</td>
      <td>Boss</td>
      <td>GS</td>
      <td>Top</td>
      <td>Camille</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>188</td>
      <td>11107</td>
      <td>0.17</td>
      <td>0.78</td>
      <td>8</td>
      <td>8</td>
      <td>16</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GS</td>
      <td>Crazy</td>
      <td>UOL</td>
      <td>Top</td>
      <td>Gwen</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>217</td>
      <td>12201</td>
      <td>0.20</td>
      <td>0.52</td>
      <td>10</td>
      <td>7</td>
      <td>17</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>W</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UOL</td>
      <td>Ahahacik</td>
      <td>GS</td>
      <td>Jungle</td>
      <td>Trundle</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>156</td>
      <td>9048</td>
      <td>0.15</td>
      <td>0.78</td>
      <td>8</td>
      <td>14</td>
      <td>22</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>L</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GS</td>
      <td>Mojito</td>
      <td>UOL</td>
      <td>Jungle</td>
      <td>Talon</td>
      <td>5</td>
      <td>4</td>
      <td>10</td>
      <td>194</td>
      <td>11234</td>
      <td>0.23</td>
      <td>0.65</td>
      <td>12</td>
      <td>8</td>
      <td>20</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>W</td>
    </tr>
    <tr>
      <th>4</th>
      <td>UOL</td>
      <td>Nomanz</td>
      <td>GS</td>
      <td>Mid</td>
      <td>Leblanc</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>216</td>
      <td>9245</td>
      <td>0.29</td>
      <td>0.56</td>
      <td>6</td>
      <td>9</td>
      <td>15</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>L</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 220 entries, 0 to 219
    Data columns (total 20 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   Team                   220 non-null    object 
     1   Player                 220 non-null    object 
     2   Opponent               220 non-null    object 
     3   Position               220 non-null    object 
     4   Champion               220 non-null    object 
     5   Kills                  220 non-null    int64  
     6   Deaths                 220 non-null    int64  
     7   Assists                220 non-null    int64  
     8   Creep Score            220 non-null    int64  
     9   Gold Earned            220 non-null    int64  
     10  Champion Damage Share  220 non-null    float64
     11  Kill Participation     220 non-null    float64
     12  Wards Placed           220 non-null    int64  
     13  Wards Destroyed        220 non-null    int64  
     14  Ward Interactions      220 non-null    int64  
     15  Dragons For            220 non-null    int64  
     16  Dragons Against        220 non-null    int64  
     17  Barons For             220 non-null    int64  
     18  Barons Against         220 non-null    int64  
     19  Result                 220 non-null    object 
    dtypes: float64(2), int64(12), object(6)
    memory usage: 34.5+ KB
    


```python
df_copy = pd.read_csv(r"C:\Users\zl262\OneDrive\Desktop\input.file\dataset.csv")
```


```python
df['Result'] = df['Result'].map({'W':1,'L':0})
```


```python
df.drop(['Dragons Against','Barons Against'],axis= 1, inplace = True)
```


```python
df['Team'] = df['Team'].astype('category')
df['Player'] = df['Player'].astype('category')
df['Position'] = df['Position'].astype('category')
df['Champion'] = df['Champion'].astype('category')
df['Opponent'] = df['Opponent'].astype('category')
```


```python
Teams =  dict(enumerate(df['Team'].cat.categories))
Teams
```




    {0: 'BYG',
     1: 'C9',
     2: 'DFM',
     3: 'GS',
     4: 'HLE',
     5: 'INF',
     6: 'LNG',
     7: 'PCE',
     8: 'RED',
     9: 'UOL'}




```python
Players = dict(enumerate(df['Player'].cat.categories))
Players
```




    {0: 'Ackerman',
     1: 'Aegis',
     2: 'Ahahacik',
     3: 'Aladoric',
     4: 'Ale',
     5: 'Alive',
     6: 'Argonavt',
     7: 'Aria',
     8: 'Babip',
     9: 'Bapip',
     10: 'Blaber',
     11: 'Bolulu',
     12: 'Boss',
     13: 'Buggax',
     14: 'Bulcan',
     15: 'Chovy',
     16: 'Cody',
     17: 'Crazy',
     18: 'Deft',
     19: 'Doggo',
     20: 'Evi',
     21: 'Fudge',
     22: 'Gaeng',
     23: 'Grevthar',
     24: 'Guigo',
     25: 'Husha',
     26: 'Icon',
     27: 'Iwandy',
     28: 'Jojo',
     29: 'Kino',
     30: 'Leona',
     31: 'Liang',
     32: 'Light',
     33: 'Maoan',
     34: 'Mojito',
     35: 'Morgan',
     36: 'Nomanz',
     37: 'Perkz',
     38: 'Pk',
     39: 'Santas',
     40: 'Solidsnake',
     41: 'Steal',
     42: 'Tally',
     43: 'Tarzan',
     44: 'Titan',
     45: 'Violet',
     46: 'Vizicsacsi',
     47: 'Vsta',
     48: 'Vulcan',
     49: 'Whitelotus',
     50: 'Willer',
     51: 'Yutapon',
     52: 'Zergsting',
     53: 'Zersting',
     54: 'Zven'}




```python
df[df['Player'].isin(['Babip','Bapip'])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Player</th>
      <th>Opponent</th>
      <th>Position</th>
      <th>Champion</th>
      <th>Kills</th>
      <th>Deaths</th>
      <th>Assists</th>
      <th>Creep Score</th>
      <th>Gold Earned</th>
      <th>Champion Damage Share</th>
      <th>Kill Participation</th>
      <th>Wards Placed</th>
      <th>Wards Destroyed</th>
      <th>Ward Interactions</th>
      <th>Dragons For</th>
      <th>Barons For</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53</th>
      <td>PCE</td>
      <td>Babip</td>
      <td>LNG</td>
      <td>Jungle</td>
      <td>Zed</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>116</td>
      <td>6557</td>
      <td>0.18</td>
      <td>0.80</td>
      <td>4</td>
      <td>5</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>133</th>
      <td>PCE</td>
      <td>Babip</td>
      <td>INF</td>
      <td>Jungle</td>
      <td>Lillia</td>
      <td>8</td>
      <td>4</td>
      <td>7</td>
      <td>240</td>
      <td>14521</td>
      <td>0.23</td>
      <td>0.71</td>
      <td>5</td>
      <td>13</td>
      <td>18</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>152</th>
      <td>PCE</td>
      <td>Bapip</td>
      <td>HLE</td>
      <td>Jungle</td>
      <td>Lillia</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>162</td>
      <td>9729</td>
      <td>0.37</td>
      <td>0.71</td>
      <td>6</td>
      <td>11</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>182</th>
      <td>PCE</td>
      <td>Babip</td>
      <td>RED</td>
      <td>Jungle</td>
      <td>Nocturn</td>
      <td>2</td>
      <td>3</td>
      <td>12</td>
      <td>168</td>
      <td>10498</td>
      <td>0.16</td>
      <td>0.56</td>
      <td>3</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['Player'].isin(['Ale','Ale'])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Player</th>
      <th>Opponent</th>
      <th>Position</th>
      <th>Champion</th>
      <th>Kills</th>
      <th>Deaths</th>
      <th>Assists</th>
      <th>Creep Score</th>
      <th>Gold Earned</th>
      <th>Champion Damage Share</th>
      <th>Kill Participation</th>
      <th>Wards Placed</th>
      <th>Wards Destroyed</th>
      <th>Ward Interactions</th>
      <th>Dragons For</th>
      <th>Barons For</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>LNG</td>
      <td>Ale</td>
      <td>PCE</td>
      <td>Top</td>
      <td>Fiora</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>235</td>
      <td>13180</td>
      <td>0.24</td>
      <td>0.67</td>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>LNG</td>
      <td>Ale</td>
      <td>HLE</td>
      <td>Top</td>
      <td>Jax</td>
      <td>6</td>
      <td>0</td>
      <td>8</td>
      <td>395</td>
      <td>19128</td>
      <td>0.28</td>
      <td>0.78</td>
      <td>18</td>
      <td>12</td>
      <td>30</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>141</th>
      <td>LNG</td>
      <td>Ale</td>
      <td>RED</td>
      <td>Top</td>
      <td>Jayce</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>264</td>
      <td>14251</td>
      <td>0.33</td>
      <td>0.60</td>
      <td>12</td>
      <td>4</td>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>190</th>
      <td>LNG</td>
      <td>Ale</td>
      <td>INF</td>
      <td>Top</td>
      <td>Wukong</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>216</td>
      <td>13719</td>
      <td>0.24</td>
      <td>0.43</td>
      <td>6</td>
      <td>3</td>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['Player'].isin(['Icon','Icon'])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Player</th>
      <th>Opponent</th>
      <th>Position</th>
      <th>Champion</th>
      <th>Kills</th>
      <th>Deaths</th>
      <th>Assists</th>
      <th>Creep Score</th>
      <th>Gold Earned</th>
      <th>Champion Damage Share</th>
      <th>Kill Participation</th>
      <th>Wards Placed</th>
      <th>Wards Destroyed</th>
      <th>Ward Interactions</th>
      <th>Dragons For</th>
      <th>Barons For</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54</th>
      <td>LNG</td>
      <td>Icon</td>
      <td>PCE</td>
      <td>Mid</td>
      <td>Leblanc</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>230</td>
      <td>10241</td>
      <td>0.41</td>
      <td>0.73</td>
      <td>5</td>
      <td>5</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>75</th>
      <td>LNG</td>
      <td>Icon</td>
      <td>HLE</td>
      <td>Mid</td>
      <td>Gragas</td>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>359</td>
      <td>15853</td>
      <td>0.24</td>
      <td>0.67</td>
      <td>10</td>
      <td>1</td>
      <td>11</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>145</th>
      <td>LNG</td>
      <td>Icon</td>
      <td>RED</td>
      <td>Mid</td>
      <td>Zed</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>266</td>
      <td>11214</td>
      <td>0.18</td>
      <td>0.13</td>
      <td>14</td>
      <td>4</td>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>194</th>
      <td>LNG</td>
      <td>Icon</td>
      <td>INF</td>
      <td>Mid</td>
      <td>Lissandra</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>208</td>
      <td>9633</td>
      <td>0.21</td>
      <td>0.48</td>
      <td>9</td>
      <td>6</td>
      <td>15</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
Players = dict(enumerate(df['Player'].cat.categories))
Players
```




    {0: 'Ackerman',
     1: 'Aegis',
     2: 'Ahahacik',
     3: 'Aladoric',
     4: 'Ale',
     5: 'Alive',
     6: 'Argonavt',
     7: 'Aria',
     8: 'Babip',
     9: 'Bapip',
     10: 'Blaber',
     11: 'Bolulu',
     12: 'Boss',
     13: 'Buggax',
     14: 'Bulcan',
     15: 'Chovy',
     16: 'Cody',
     17: 'Crazy',
     18: 'Deft',
     19: 'Doggo',
     20: 'Evi',
     21: 'Fudge',
     22: 'Gaeng',
     23: 'Grevthar',
     24: 'Guigo',
     25: 'Husha',
     26: 'Icon',
     27: 'Iwandy',
     28: 'Jojo',
     29: 'Kino',
     30: 'Leona',
     31: 'Liang',
     32: 'Light',
     33: 'Maoan',
     34: 'Mojito',
     35: 'Morgan',
     36: 'Nomanz',
     37: 'Perkz',
     38: 'Pk',
     39: 'Santas',
     40: 'Solidsnake',
     41: 'Steal',
     42: 'Tally',
     43: 'Tarzan',
     44: 'Titan',
     45: 'Violet',
     46: 'Vizicsacsi',
     47: 'Vsta',
     48: 'Vulcan',
     49: 'Whitelotus',
     50: 'Willer',
     51: 'Yutapon',
     52: 'Zergsting',
     53: 'Zersting',
     54: 'Zven'}




```python
Pos = dict(enumerate(df['Position'].cat.categories))
Pos
```




    {0: 'Adc', 1: 'Jungle', 2: 'Mid', 3: 'Support', 4: 'Top'}




```python
Champ = dict(enumerate(df['Champion'].cat.categories))
Champ
```




    {0: 'Aatrox',
     1: 'Akali',
     2: 'Alistar',
     3: 'Amumu',
     4: 'Aphelios',
     5: 'Azir',
     6: 'Bard',
     7: 'Braum',
     8: 'Camille',
     9: 'Draven',
     10: 'Ezreal',
     11: 'Fiora',
     12: 'Galio',
     13: 'Gangplank',
     14: 'Gnar',
     15: 'Gragas',
     16: 'Graves',
     17: 'Gwen',
     18: 'Irelia',
     19: 'Jarvan',
     20: 'Jax',
     21: 'Jayce',
     22: 'Jhin',
     23: 'Kaisa',
     24: 'Kalista',
     25: 'Karma',
     26: 'Kennen',
     27: 'Leblanc',
     28: 'Lee Sin',
     29: 'Leona',
     30: 'Lillia',
     31: 'Lissandra',
     32: 'Lucian',
     33: 'Lulu',
     34: 'Malphite',
     35: 'Miss Fortune',
     36: 'Nami',
     37: 'Nautilus',
     38: 'Nocturn',
     39: 'Olaf',
     40: 'Orianna',
     41: 'Qiyana',
     42: 'Rakan',
     43: 'Rell',
     44: 'Renekton',
     45: 'Ryze',
     46: 'Sejuani',
     47: 'Senna',
     48: 'Sett',
     49: 'Sion',
     50: 'Sylas',
     51: 'Syndra',
     52: 'Tahm Kench',
     53: 'Taliyah',
     54: 'Talon',
     55: 'Thresh',
     56: 'Tristana',
     57: 'Trundle',
     58: 'Tryndamere',
     59: 'Twisted Fate',
     60: 'Urgot',
     61: 'Wukong',
     62: 'Xayah',
     63: 'Xin Zhao',
     64: 'Yone',
     65: 'Zed',
     66: 'Zoe'}




```python
Opp = dict(enumerate(df['Opponent'].cat.categories))
Opp
```




    {0: 'BYG',
     1: 'C9',
     2: 'DFM',
     3: 'GS',
     4: 'HLE',
     5: 'INF',
     6: 'LNG',
     7: 'PCE',
     8: 'RED',
     9: 'UOL'}




```python
df['Team'] = df['Team'].cat.codes
df['Player'] = df['Player'].cat.codes
df['Position'] = df['Position'].cat.codes
df['Champion'] = df['Champion'].cat.codes
df['Opponent'] = df['Opponent'].cat.codes
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Player</th>
      <th>Opponent</th>
      <th>Position</th>
      <th>Champion</th>
      <th>Kills</th>
      <th>Deaths</th>
      <th>Assists</th>
      <th>Creep Score</th>
      <th>Gold Earned</th>
      <th>Champion Damage Share</th>
      <th>Kill Participation</th>
      <th>Wards Placed</th>
      <th>Wards Destroyed</th>
      <th>Ward Interactions</th>
      <th>Dragons For</th>
      <th>Barons For</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>8</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>188</td>
      <td>11107</td>
      <td>0.17</td>
      <td>0.78</td>
      <td>8</td>
      <td>8</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>17</td>
      <td>9</td>
      <td>4</td>
      <td>17</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>217</td>
      <td>12201</td>
      <td>0.20</td>
      <td>0.52</td>
      <td>10</td>
      <td>7</td>
      <td>17</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>57</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>156</td>
      <td>9048</td>
      <td>0.15</td>
      <td>0.78</td>
      <td>8</td>
      <td>14</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>34</td>
      <td>9</td>
      <td>1</td>
      <td>54</td>
      <td>5</td>
      <td>4</td>
      <td>10</td>
      <td>194</td>
      <td>11234</td>
      <td>0.23</td>
      <td>0.65</td>
      <td>12</td>
      <td>8</td>
      <td>20</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>36</td>
      <td>3</td>
      <td>2</td>
      <td>27</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>216</td>
      <td>9245</td>
      <td>0.29</td>
      <td>0.56</td>
      <td>6</td>
      <td>9</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr = df.corr()
```


```python
plt.figure(figsize = (16,5))
sns.heatmap(corr,annot = True,cmap = "RdBu_r")
```




    <AxesSubplot:>




    
![png](output_20_1.png)
    



```python
X = df.loc[:,~df.columns.isin(['Team', 'Player','Opponent','Result'])]
y = df.Result
```


```python
X.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Position</th>
      <th>Champion</th>
      <th>Kills</th>
      <th>Deaths</th>
      <th>Assists</th>
      <th>Creep Score</th>
      <th>Gold Earned</th>
      <th>Champion Damage Share</th>
      <th>Kill Participation</th>
      <th>Wards Placed</th>
      <th>Wards Destroyed</th>
      <th>Ward Interactions</th>
      <th>Dragons For</th>
      <th>Barons For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
      <td>220.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.000000</td>
      <td>32.363636</td>
      <td>2.709091</td>
      <td>2.718182</td>
      <td>5.668182</td>
      <td>200.340909</td>
      <td>11008.159091</td>
      <td>0.200000</td>
      <td>0.608273</td>
      <td>19.454545</td>
      <td>8.704545</td>
      <td>28.159091</td>
      <td>2.090909</td>
      <td>0.545455</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.417439</td>
      <td>18.178078</td>
      <td>2.579673</td>
      <td>1.747179</td>
      <td>3.888149</td>
      <td>101.328153</td>
      <td>3198.806207</td>
      <td>0.095051</td>
      <td>0.190404</td>
      <td>15.713553</td>
      <td>5.101614</td>
      <td>18.051857</td>
      <td>1.381849</td>
      <td>0.657050</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>4714.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>156.000000</td>
      <td>8691.250000</td>
      <td>0.130000</td>
      <td>0.500000</td>
      <td>10.000000</td>
      <td>5.000000</td>
      <td>17.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>30.500000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>210.000000</td>
      <td>10454.500000</td>
      <td>0.205000</td>
      <td>0.625000</td>
      <td>14.000000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>45.250000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>266.250000</td>
      <td>13431.750000</td>
      <td>0.260000</td>
      <td>0.740000</td>
      <td>19.000000</td>
      <td>12.000000</td>
      <td>30.500000</td>
      <td>3.250000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>66.000000</td>
      <td>13.000000</td>
      <td>7.000000</td>
      <td>19.000000</td>
      <td>419.000000</td>
      <td>20546.000000</td>
      <td>0.470000</td>
      <td>1.000000</td>
      <td>92.000000</td>
      <td>30.000000</td>
      <td>112.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
```


```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Position</th>
      <th>Champion</th>
      <th>Kills</th>
      <th>Deaths</th>
      <th>Assists</th>
      <th>Creep Score</th>
      <th>Gold Earned</th>
      <th>Champion Damage Share</th>
      <th>Kill Participation</th>
      <th>Wards Placed</th>
      <th>Wards Destroyed</th>
      <th>Ward Interactions</th>
      <th>Dragons For</th>
      <th>Barons For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>68</th>
      <td>3</td>
      <td>36</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>21</td>
      <td>6744</td>
      <td>0.09</td>
      <td>0.83</td>
      <td>43</td>
      <td>18</td>
      <td>61</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>165</th>
      <td>2</td>
      <td>50</td>
      <td>6</td>
      <td>4</td>
      <td>9</td>
      <td>193</td>
      <td>11195</td>
      <td>0.35</td>
      <td>0.79</td>
      <td>6</td>
      <td>4</td>
      <td>10</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194</th>
      <td>2</td>
      <td>31</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>208</td>
      <td>9633</td>
      <td>0.21</td>
      <td>0.48</td>
      <td>9</td>
      <td>6</td>
      <td>15</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>160</th>
      <td>4</td>
      <td>13</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>186</td>
      <td>9567</td>
      <td>0.32</td>
      <td>0.73</td>
      <td>9</td>
      <td>4</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>203</th>
      <td>1</td>
      <td>39</td>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>225</td>
      <td>13610</td>
      <td>0.29</td>
      <td>0.83</td>
      <td>9</td>
      <td>14</td>
      <td>23</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>185</th>
      <td>2</td>
      <td>45</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>247</td>
      <td>10348</td>
      <td>0.22</td>
      <td>0.44</td>
      <td>14</td>
      <td>7</td>
      <td>21</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2</td>
      <td>50</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>205</td>
      <td>8493</td>
      <td>0.25</td>
      <td>0.57</td>
      <td>3</td>
      <td>6</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>172</th>
      <td>1</td>
      <td>30</td>
      <td>4</td>
      <td>0</td>
      <td>12</td>
      <td>215</td>
      <td>12194</td>
      <td>0.29</td>
      <td>0.73</td>
      <td>13</td>
      <td>16</td>
      <td>29</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>70</th>
      <td>4</td>
      <td>18</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>374</td>
      <td>14645</td>
      <td>0.16</td>
      <td>0.00</td>
      <td>15</td>
      <td>9</td>
      <td>24</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>4</td>
      <td>61</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>151</td>
      <td>8216</td>
      <td>0.21</td>
      <td>0.86</td>
      <td>7</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>132 rows × 14 columns</p>
</div>




```python
XGb_model = XGBClassifier()
XGb_model.fit(X_train,y_train)
```

    [20:18:55] WARNING: ..\src\learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    

    C:\Users\zl262\anaconda3\lib\site-packages\xgboost\sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                  gamma=0, gpu_id=-1, importance_type=None,
                  interaction_constraints='', learning_rate=0.300000012,
                  max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
                  monotone_constraints='()', n_estimators=100, n_jobs=16,
                  num_parallel_tree=1, predictor='auto', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)




```python
train_predictions = XGb_model.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))
```

    Accuracy: 95.4545%
    


```python
val_predictions = XGb_model.predict(X_val)
acc = accuracy_score(y_val, val_predictions)
print("Accuracy: {:.4%}".format(acc))

```

    Accuracy: 93.1818%
    


```python
from xgboost import plot_importance
plot_importance(XGb_model)
```




    <AxesSubplot:title={'center':'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](output_28_1.png)
    



```python

df['Team'] = df['Team'].astype('category')
df['Player'] = df['Player'].astype('category')
df['Position'] = df['Position'].astype('category')
df['Champion'] = df['Champion'].astype('category')
df['Opponent'] = df['Opponent'].astype('category')
df['Result'] = df['Result'].astype('category')
print(df.dtypes)
```

    Team                     category
    Player                   category
    Opponent                 category
    Position                 category
    Champion                 category
    Kills                       int64
    Deaths                      int64
    Assists                     int64
    Creep Score                 int64
    Gold Earned                 int64
    Champion Damage Share     float64
    Kill Participation        float64
    Wards Placed                int64
    Wards Destroyed             int64
    Ward Interactions           int64
    Dragons For                 int64
    Barons For                  int64
    Result                   category
    dtype: object
    


```python
df['Team'] = df['Team'].cat.codes
df['Player'] = df['Player'].cat.codes
df['Position'] = df['Position'].cat.codes
df['Champion'] = df['Champion'].cat.codes
df['Opponent'] = df['Opponent'].cat.codes
df['Result'] = df['Result'].cat.codes
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Player</th>
      <th>Opponent</th>
      <th>Position</th>
      <th>Champion</th>
      <th>Kills</th>
      <th>Deaths</th>
      <th>Assists</th>
      <th>Creep Score</th>
      <th>Gold Earned</th>
      <th>Champion Damage Share</th>
      <th>Kill Participation</th>
      <th>Wards Placed</th>
      <th>Wards Destroyed</th>
      <th>Ward Interactions</th>
      <th>Dragons For</th>
      <th>Barons For</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>8</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>188</td>
      <td>11107</td>
      <td>0.17</td>
      <td>0.78</td>
      <td>8</td>
      <td>8</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>17</td>
      <td>9</td>
      <td>4</td>
      <td>17</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>217</td>
      <td>12201</td>
      <td>0.20</td>
      <td>0.52</td>
      <td>10</td>
      <td>7</td>
      <td>17</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>57</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>156</td>
      <td>9048</td>
      <td>0.15</td>
      <td>0.78</td>
      <td>8</td>
      <td>14</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>34</td>
      <td>9</td>
      <td>1</td>
      <td>54</td>
      <td>5</td>
      <td>4</td>
      <td>10</td>
      <td>194</td>
      <td>11234</td>
      <td>0.23</td>
      <td>0.65</td>
      <td>12</td>
      <td>8</td>
      <td>20</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>36</td>
      <td>3</td>
      <td>2</td>
      <td>27</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>216</td>
      <td>9245</td>
      <td>0.29</td>
      <td>0.56</td>
      <td>6</td>
      <td>9</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df.drop('Result', axis = 1)
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (154, 17)
    (154,)
    (66, 17)
    (66,)
    


```python
classifiers = [
    KNeighborsClassifier(n_neighbors = 5),
    SVC(kernel = 'linear', gamma = 'auto', C = 5, probability = True),
    NuSVC(probability = True),
    DecisionTreeClassifier(max_depth = 3),
    RandomForestClassifier(random_state = 1, max_features = 'sqrt', n_jobs = 1, verbose = 1),
    XGBClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
```


```python
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    print("="*30)
    print(name)
    print('****Results****')
    
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    print("\n")
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)
```

    ==============================
    KNeighborsClassifier
    ****Results****
    Accuracy: 65.1515%
    Log Loss: 1.0905065791197928
    
    
    ==============================
    SVC
    ****Results****
    Accuracy: 89.3939%
    Log Loss: 0.29458271459221785
    
    
    ==============================
    NuSVC
    ****Results****
    Accuracy: 75.7576%
    Log Loss: 0.5713124867155898
    
    
    ==============================
    DecisionTreeClassifier
    ****Results****
    Accuracy: 90.9091%
    Log Loss: 1.2488916709594005
    
    
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    0.0s finished
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    0.0s finished
    C:\Users\zl262\anaconda3\lib\site-packages\xgboost\sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    ==============================
    RandomForestClassifier
    ****Results****
    Accuracy: 100.0000%
    Log Loss: 0.20707581808704406
    
    
    [20:23:32] WARNING: ..\src\learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    ==============================
    XGBClassifier
    ****Results****
    Accuracy: 98.4848%
    Log Loss: 0.04210175078575654
    
    
    ==============================
    AdaBoostClassifier
    ****Results****
    Accuracy: 92.4242%
    Log Loss: 0.49708640917854857
    
    
    ==============================
    GradientBoostingClassifier
    ****Results****
    Accuracy: 96.9697%
    Log Loss: 0.09029776913057452
    
    
    ==============================
    GaussianNB
    ****Results****
    Accuracy: 89.3939%
    Log Loss: 0.41449908952022135
    
    
    ==============================
    LinearDiscriminantAnalysis
    ****Results****
    Accuracy: 96.9697%
    Log Loss: 0.04301917965530296
    
    
    ==============================
    QuadraticDiscriminantAnalysis
    ****Results****
    Accuracy: 96.9697%
    Log Loss: 0.5055512041808419
    
    
    ==============================
    

    C:\Users\zl262\anaconda3\lib\site-packages\sklearn\discriminant_analysis.py:808: UserWarning: Variables are collinear
      warnings.warn("Variables are collinear")
    


```python
plt.figure(figsize = (10,6))
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y = 'Classifier', data = log, color = "springgreen")
plt.xlabel('Accuracy %')
plt.title('Accuracy of Classification Model')
plt.show()
sns.set_color_codes("muted")
plt.show()
```


    
![png](output_34_0.png)
    



```python
plt.figure(figsize = (10,6))
sns.barplot(x = 'Log Loss', y = 'Classifier', data = log, color = "orangered")
plt.xlabel('Log Loss')
plt.title('Log Loss of Classification Model')
plt.show()
```


    
![png](output_35_0.png)
    



```python
rf = RandomForestClassifier(random_state = 1, max_features = 'sqrt', n_jobs = 1, verbose = 1)
%time rf.fit(X_train, y_train)
```

    Wall time: 152 ms
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    0.0s finished
    




    RandomForestClassifier(max_features='sqrt', n_jobs=1, random_state=1, verbose=1)




```python
y_pred = rf.predict(X_test)
print(y_pred)
```

    [0 0 1 1 0 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 1 0 0 0 1 1 1 0 1 0
     0 0 1 0 1 0 0 1 0 0 0 0 1 1 0 1 0 0 1 0 0 1 1 1 1 1 0 1 0]
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    0.0s finished
    


```python
matrix = metrics.confusion_matrix(y_test, y_pred)
print(matrix)

plt.figure(figsize = (10,6))
sns.heatmap(matrix, annot = True, fmt = ".0f", cmap = 'RdYlBu')
plt.title("Prediction")
plt.show()
```

    [[32  0]
     [ 0 34]]
    


    
![png](output_38_1.png)
    



```python
report = metrics.classification_report(y_test, y_pred)
print(report)
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        32
               1       1.00      1.00      1.00        34
    
        accuracy                           1.00        66
       macro avg       1.00      1.00      1.00        66
    weighted avg       1.00      1.00      1.00        66
    
    


```python
feature = pd.Series(rf.feature_importances_, index = X_train.columns).sort_values(ascending = False)
print(feature)
```

    Barons For               0.185231
    Assists                  0.172662
    Dragons For              0.161857
    Deaths                   0.143338
    Team                     0.055099
    Kills                    0.053816
    Gold Earned              0.036866
    Opponent                 0.030278
    Kill Participation       0.029197
    Creep Score              0.023980
    Champion Damage Share    0.020510
    Player                   0.018480
    Wards Placed             0.017715
    Champion                 0.017426
    Ward Interactions        0.015683
    Wards Destroyed          0.014755
    Position                 0.003107
    dtype: float64
    


```python
plt.figure(figsize = (10,6))
sns.barplot(x = feature, y = feature.index)
plt.title("Feature Importance")
plt.xlabel('Score')
plt.ylabel('Features')
plt.show()
```


    
![png](output_41_0.png)
    



```python
team_result = df.groupby(['Team', 'Result']).size().reset_index(name = 'Count')
print(team_result)
```

        Team  Result  Count
    0      0       0     15
    1      0       1     10
    2      1       0     10
    3      1       1     15
    4      2       0      5
    5      2       1     20
    6      3       0     10
    7      3       1     10
    8      4       0      5
    9      4       1     15
    10     5       0     20
    11     6       1     20
    12     7       0     10
    13     7       1     10
    14     8       0     15
    15     8       1      5
    16     9       0     20
    17     9       1      5
    


```python
plt.figure(figsize = (10,6))
sns.barplot(x = 'Team', y = 'Count', hue = 'Result', data = team_result, palette = 'Set1')
plt.title("Team ~ Result")
plt.legend(bbox_to_anchor = (1.1,1), borderaxespad = 0)
plt.show()
```


    
![png](output_43_0.png)
    



```python
op_result = df.groupby(['Opponent', 'Result']).size().reset_index(name = 'Count')
print(op_result)
```

        Opponent  Result  Count
    0          0       0     10
    1          0       1     15
    2          1       0     15
    3          1       1     10
    4          2       0     20
    5          2       1      5
    6          3       0     10
    7          3       1     10
    8          4       0     15
    9          4       1      5
    10         5       1     20
    11         6       0     20
    12         7       0     10
    13         7       1     10
    14         8       0      5
    15         8       1     15
    16         9       0      5
    17         9       1     20
    


```python
plt.figure(figsize = (10,6))
sns.barplot(x = 'Opponent', y = 'Count', hue = 'Result', data = op_result)
plt.title("Opponent ~ Result")
plt.legend(bbox_to_anchor = (1.1,1), borderaxespad = 0)
plt.show()
```


    
![png](output_45_0.png)
    



```python
plt.figure(figsize = (10,6))
result = [np.count_nonzero(df['Result'] == '1'),
         np.count_nonzero(df['Result'] == '0')]
activities = ['Loss', 'Win']
plt.pie(result, labels = activities, startangle = 100, autopct = '%.2f%%', shadow = True)
plt.title("Result")
plt.show()
```

    C:\Users\zl262\AppData\Local\Temp/ipykernel_15496/1223124718.py:5: MatplotlibDeprecationWarning: normalize=None does not normalize if the sum is less than 1 but this behavior is deprecated since 3.3 until two minor releases later. After the deprecation period the default value will be normalize=True. To prevent normalization pass normalize=False 
      plt.pie(result, labels = activities, startangle = 100, autopct = '%.2f%%', shadow = True)
    


    
![png](output_46_1.png)
    



```python

```
