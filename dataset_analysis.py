# 0. Instalar bibliotecas abaixo:
# pip install ydata-profiling pandera great-expectations evidently
#             tensorflow-data-validation cleanlab scikit-learn pyod fairlearn

import pandas as pd
from evidently.report import Report
from evidently.metrics import (DataDriftTable, DataBalanceMetric)
import tensorflow_data_validation as tfdv
import cleanlab
from pyod.models.iforest import IForest

# 1. Carrega dados
ref  = pd.read_parquet("train.parquet")
curr = pd.read_parquet("2025-05-prod.parquet")

# 2. Profiling e schema
stats_ref = tfdv.generate_statistics_from_dataframe(ref)
schema    = tfdv.infer_schema(stats_ref)
anomalies = tfdv.validate_statistics(
              statistics=tfdv.generate_statistics_from_dataframe(curr),
              schema=schema)

# 3. Drift + balance report
report = Report(metrics=[DataDriftTable(), DataBalanceMetric()])
report.run(ref, curr)
report.save_html("evidently_report.html")

# 4. Outliers multivariados
iso = IForest().fit(ref.select_dtypes("number"))
curr["outlier_flag"] = iso.predict(curr.select_dtypes("number"))

# 5. Limpeza de rótulos (supervisionados)
cl = cleanlab.CleanLearning(clf).fit(curr[features], curr["label"])
issue_idx = cl.get_label_issues()
