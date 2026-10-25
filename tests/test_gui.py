"""Exercise the actual Streamlit script, navigation and prediction form submissions."""
from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def ok(at,label):
    assert not at.exception,[(e.message,e.stack_trace) for e in at.exception]
    assert not at.error,[e.value for e in at.error]
    print('PASS',label)

at=AppTest.from_file(str(ROOT/'app/app.py'),default_timeout=90).run()
ok(at,'Dashboard')
assert 'Data garage' not in at.sidebar.radio[0].options
print('PASS Data Garage removed')
at.sidebar.radio[0].set_value('Predict transmission (Classification)').run()
ok(at,'Prediction page')
assert at.selectbox(key='predict_stage_random_forest').value=='After tuning'
at.button[0].click().run()
ok(at,'Single classification form')
assert any('Predicted transmission:' in s.value for s in at.success)
for name in ['Random Forest','AdaBoost','Gradient Boosting','Bagging','MLP','Logistic Regression','KNN','Gaussian Naive Bayes','Decision Tree','SVC']:
    at.selectbox[0].set_value(name).run()
    at.button[0].click().run()
    ok(at,'Predict '+name)
at.sidebar.radio[0].set_value('Classification benchmarks').run()
ok(at,'Ten-model comparison')
at.radio(key='comparison_view').set_value('After tuning').run()
ok(at,'After-tuning-only comparison')
at.radio(key='comparison_view').set_value('All ten models').run()
ok(at,'All ten new-dataset models')
at.radio(key='comparison_view').set_value('Model diagnostics').run()
ok(at,'Part B tuning comparison')
for slug in ['random_forest','adaboost','gradient_boosting','bagging','mlp']:
    at.selectbox(key='diagnostic_model').set_value(slug).run();ok(at,'Diagnostics '+slug)
at.sidebar.radio[0].set_value('Car clusters').run()
at.radio(key='cluster_view').set_value('Assign a car').run()
for kind in ['kmeans','agglomerative']:
    at.selectbox[0].set_value(kind).run()
    at.button[0].click().run();ok(at,'Cluster assignment '+kind)
    assert any('Assigned segment:' in s.value for s in at.success)
at.radio(key='cluster_view').set_value('Segments').run()
at.multiselect[0].set_value([]).run();ok(at,'Empty cluster filter')
at.radio(key='cluster_view').set_value('After tuning').run();ok(at,'Clustering after tuning')
at.radio(key='cluster_view').set_value('Before vs after').run();ok(at,'Clustering before/after')
at.radio(key='cluster_view').set_value('Diagnostics').run()
for kind in ['kmeans', 'agglomerative']:
    at.selectbox(key='cluster_model').set_value(kind).run()
    at.selectbox(key=f'cluster_diagnostic_{kind}').set_value('Five-fold tuning search').run()
    ok(at, f'Cluster diagnostics five-fold tuning search {kind}')
    df = at.dataframe[0].value
    assert 'linkage' in df.columns, f'linkage missing from {kind}'
at.sidebar.radio[0].set_value('Evaluation metrics').run();ok(at,'Evaluation metrics')
print('ALL GUI CHECKS PASSED')
