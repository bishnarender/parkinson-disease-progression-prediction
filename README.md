## parkinson's-disease-progression-prediction
## score at 3rd position is achieved.
![parkinson-progression-submission](https://github.com/bishnarender/parkinson-disease-progression-prediction/assets/49610834/0636250f-e2b0-4dea-9175-04b79dfd299e)

### Start 
-----
For better understanding of project, read the files in the following order:
1. eda.ipynb 
2. parkinson-progression-submission.ipynb

<b>Code has been explained in the above files and in the linked files to these.</b>

<b>Final solution is a ensemble of two models: LGB and NN.</b>

### Features
-----
Train/sample DataFrame build up.
First of all, the clinical DataFrame is picked up. And two copies of this are created.

In one copy visit_month and visit_id are renamed as target_month and visit_id_target respectively. 
<code>
clinical_df.rename({"visit_month":"target_month", "visit_id":"visit_id_target"}, axis=1)
</code>

From other copy only three columns ["patient_id", "visit_month", "visit_id"] are picked. 
<code>
clinical_df[["patient_id", "visit_month", "visit_id"]]
</code>


Each row ["visit_id_target","patient_id","target_month","updrs_1","updrs_2","updrs_3","updrs_4","upd23b_clinical_state_on_medication"] of renamed copy is merged with other copy at particular patient_id.
Thus each row is attached with all visit_month and visit_id of present patient_id.
<code>
clinical_df.rename({"visit_month":"target_month", "visit_id":"visit_id_target"}, axis=1).merge(clinical_df[["patient_id", "visit_month", "visit_id"]], how="left", on="patient_id")
</code>

​After merging the newly created train DataFrame is referred to as a "sample".

Feature <b>=></b> sample['horizon'].
<code>
sample["horizon"] = sample["target_month"] - sample["visit_month"]
</code>
Now keep those rows for which ["horizon"] feature is in [0, 6, 12, 24]. This keeps only those visit_month which are less than target_month.
Further, this also keeps those visit_month (for a particular target_month) which are not later than 24 months with a gap of 6 or 12 months​.

Feature <b>=></b> sample['visit_0m']. 
<code>
sample["visit_0m"] = sample.apply(lambda x: (x["patient_id"] in p) and (x["visit_month"] >= 0), axis=1).astype(int)
#- 1 when visit_month values are equal to or greater than 0 (i.e., 0th month).
</code>
Similarly, other related features ['visit_6m', 'visit_12m', ..., 'visit_84m'] are calculated. Indicators whether a patient visit occurred on 6th, 12th, ... and 84th month.

Feature <b>=></b> sample['t_month_eq_0']. 
<code>
sample["t_month_eq_0"] = (sample["target_month"] == 0).astype(int)
#- 1 when target_month values are equal to 0 (i.e., 0th month).
</code>
Similarly, other related features ['t_month_eq_6', 't_month_eq_12', ..., 't_month_eq_84'] are calculated.

Feature <b>=></b> sample['v_month_eq_0']. 
<code>
sample["v_month_eq_0"] = (sample["visit_month"] == 0).astype(int)
#- 1 when visit_month values are equal to 0 (i.e., 0th month).
</code>
Similarly, other related features ['v_month_eq_6', 'v_month_eq_12', ..., 'v_month_eq_84'] are calculated.

Feature <b>=></b> sample['hor_eq_0']. 
<code>
sample["hor_eq_0"] = (sample["horizon"] == 0).astype(int)
#- 1 when horizon values are equal to 0 (i.e., 0th month).
</code>
Similarly, other related features ['hor_eq_6', 'hor_eq_12', 'hor_eq_24'] are calculated.

Feature <b>=></b> sample['target_n_1']. 
<code>
sample["target_i"] = 1 # 1,2,3,4
sample["target_n_1"] = (sample["target_i"] == 1).astype(int)
#- 1 when target_i values are equal to 1 (i.e., updrs_1).
</code>
Similarly, other related features ['target_n_2', 'target_n_3', 'target_n_4'] are calculated.
