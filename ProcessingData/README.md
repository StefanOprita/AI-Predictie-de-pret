# Codul folosit pentru procesarea datelor

## Biblioteci necesare
    1. pandas (pip install pandas) 
    2. pandas_ta (pip install pandas_ta)
    3. fastparquet (pip install fastparquet)

Pentru a consuma mai putin spatiu, am salvat `data-frame`-urile cu `fastparquet`
Pentru a incarca continutul fisierelor `.parquet`, folositi functia `read_parquet`

```python
import pandas as pd
data = pd.read_parquet(path_to_file, engine='fastparquet')
```