"""TORQUEX: Unified Automotive Machine Learning Intelligence Platform.
Integrates Review 1 (10 Regression Models) and Review 2 (10 Classification + 2 Clustering Models).
"""
from pathlib import Path
import sys
import json
import base64
import html
import streamlit as st

try:
    import sklearn._loss._loss
    sys.modules['_loss'] = sklearn._loss._loss
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schema import (
    RAW_FEATURES, PART_B,
    REGRESSION_RAW_FEATURES, REGRESSION_NUMERIC_FEATURES, REGRESSION_CATEGORICAL_FEATURES, REGRESSION_MODELS
)

st.set_page_config(page_title='TORQUEX | Automotive Intelligence', page_icon='🏎️', layout='wide')

@st.cache_data(show_spinner=False)
def text_asset(path, mtime):
    return (ROOT / path).read_text(encoding='utf-8')

def text(path):
    return text_asset(path, (ROOT / path).stat().st_mtime_ns)

@st.cache_data(show_spinner=False)
def json_asset(path, mtime):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def data_json(path):
    return json_asset(path, (ROOT / path).stat().st_mtime_ns)

@st.cache_data(show_spinner=False)
def csv_asset(path, mtime):
    import pandas as pd
    return pd.read_csv(ROOT / path)

def table(path):
    return csv_asset(path, (ROOT / path).stat().st_mtime_ns)

@st.cache_resource(show_spinner='Loading model into memory…', max_entries=24)
def model_asset(filename, mtime):
    import joblib
    return joblib.load(ROOT / 'models' / filename)

def model(filename):
    return model_asset(filename, (ROOT / 'models' / filename).stat().st_mtime_ns)

def probability(fitted, X):
    return fitted.predict_proba(X)[:, list(fitted.classes_).index('Automatic')]

def format_inr(val):
    if val >= 1e7:
        return f"₹ {val / 1e7:.2f} Cr"
    elif val >= 1e5:
        return f"₹ {val / 1e5:.2f} Lakh"
    else:
        return f"₹ {val:,.0f}"

def heading(kicker, title, description):
    st.markdown(
        f'<div class="tx-tag">{kicker}</div><div class="tx-page-title">{title}</div><div class="tx-page-desc">{description}</div>',
        unsafe_allow_html=True
    )

def section(title):
    st.markdown(f'<div class="tx-section">{title}</div>', unsafe_allow_html=True)

def racing_leaderboard(frame, score_column, detail_builder, title, metric_label, variant):
    records = frame.to_dict('records')
    if not records:
        return ''

    leader_score = float(records[0][score_column])
    rows = []
    for rank, record in enumerate(records, start=1):
        score = float(record[score_column])
        gap = leader_score - score
        position_class = f' tx-race-row--p{rank}' if rank <= 3 else ''
        gap_label = 'LEADER' if rank == 1 else f'+{gap:.3f}'
        score_width = max(4, min(100, score * 100))
        rows.append(
            f'<div class="tx-race-row{position_class}" '
            f'style="--row-delay:{rank * 45}ms;--score-width:{score_width:.1f}%">'
            f'<div class="tx-position"><span>{rank:02d}</span></div>'
            f'<div class="tx-race-model"><strong>{html.escape(str(record["Model"]))}</strong>'
            f'<small>{html.escape(str(detail_builder(record)))}</small></div>'
            f'<div class="tx-race-gap">{gap_label}</div>'
            f'<div class="tx-race-score">{score:.3f}</div>'
            f'<div class="tx-race-pace"><i></i></div>'
            '</div>'
        )

    return (
        f'<div class="tx-timing-tower tx-timing-tower--{variant}">'
        '<div class="tx-tower-top">'
        '<div class="tx-start-lights"><i></i><i></i><i></i></div>'
        f'<div class="tx-tower-title"><small>LIVE TIMING</small><strong>{html.escape(title)}</strong></div>'
        f'<div class="tx-entry-count"><b>{len(records):02d}</b><small>STARTERS</small></div>'
        '</div>'
        '<div class="tx-tower-columns">'
        '<span class="tx-col-pos">POS</span><span class="tx-col-model">MACHINE</span>'
        f'<span class="tx-col-gap">GAP</span><span class="tx-col-score">{html.escape(metric_label)}</span>'
        '<span class="tx-col-pace">PACE</span>'
        '</div>'
        + ''.join(rows) +
        '<div class="tx-tower-footer"><span class="tx-checkered"></span>'
        '<span>OFFICIAL CLASSIFICATION</span><em>TORQUEX</em></div>'
        '</div>'
    )

def download(frame, filename, label='Download results'):
    st.download_button(label, frame.to_csv(index=False).encode(), file_name=filename, mime='text/csv')

def figure_bar(frame, x, y, color=None, barmode=None, title='', yrange=None):
    import plotly.express as px
    fig = px.bar(
        frame, x=x, y=y, color=color, barmode=barmode,
        color_discrete_sequence=['#f03745', '#91a8c8', '#f4b76d', '#6ad1a3', '#b388ff'],
        title=title
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text='',
        font_color='#bdc1ce'
    )
    if yrange is not None:
        fig.update_yaxes(range=yrange, gridcolor='#292d37')
    else:
        fig.update_yaxes(gridcolor='#292d37')
    st.plotly_chart(fig, width='stretch')

def metric_charts(frame, metrics, title, key, labels=None, group=None, yrange=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labels = labels or {}
    columns = 2
    rows = (len(metrics) + columns - 1) // columns
    colors = ['#f03745', '#91a8c8', '#f4b76d', '#6ad1a3', '#b388ff']
    groups = [None] if group is None else list(frame[group].drop_duplicates())
    fig = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[labels.get(metric, metric) for metric in metrics]
    )

    for metric_index, metric in enumerate(metrics):
        row = metric_index // columns + 1
        column = metric_index % columns + 1
        for group_index, group_value in enumerate(groups):
            subset = frame if group_value is None else frame[frame[group].eq(group_value)]
            fig.add_trace(
                go.Bar(
                    x=subset['Model'],
                    y=subset[metric],
                    name=labels.get(metric, metric) if group_value is None else str(group_value),
                    legendgroup='' if group_value is None else str(group_value),
                    showlegend=group_value is not None and metric_index == 0,
                    marker_color=colors[(metric_index if group_value is None else group_index) % len(colors)],
                    hovertemplate='%{x}<br>%{y:,.4f}<extra></extra>'
                ),
                row=row,
                col=column
            )
        if yrange is not None:
            fig.update_yaxes(range=yrange, row=row, col=column)
        else:
            fig.update_yaxes(rangemode='tozero', row=row, col=column)

    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=max(380, rows * 300),
        margin=dict(l=20, r=20, t=75, b=40),
        barmode='group',
        legend_title_text=group or '',
        font_color='#bdc1ce'
    )
    fig.update_xaxes(tickangle=-30, automargin=True)
    fig.update_yaxes(gridcolor='#292d37', automargin=True)
    st.plotly_chart(fig, width='stretch', key=key)

st.markdown('<style>' + text('app/assets/style.css') + '</style>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="tx-logo">TORQUE<span>X</span></div><div class="tx-logo-sub">AUTOMOTIVE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="tx-tag">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.radio(
        'Navigation',
        [
            'Dashboard',
            'Predict car price (Regression)',
            'Regression benchmarks',
            'Predict transmission (Classification)',
            'Classification benchmarks',
            'Car clusters',
            'Evaluation metrics'
        ],
        key='navigation',
        label_visibility='collapsed'
    )
    st.markdown(
        '<div class="tx-side-footer">'
        '<span class="tx-status"><i class="tx-dot"></i> 22 ML MODELS ACTIVE</span><br><br>'
        '<b>Review 1:</b> 10 Regression Models (₹)<br>'
        '<b>Review 2:</b> 10 Classification + 2 Clustering<br>'
        '23CSE301 · 2026–27 Capstone'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown(
    f'<div class="tx-topline"><span>TORQUEX <strong> / {html.escape(page.upper())}</strong></span><span>FULL CAPSTONE PIPELINE · REVIEWS 01 & 02</span></div>',
    unsafe_allow_html=True
)

if not (ROOT / 'results/dashboard.json').exists():
    st.error('Prepared results are missing. Run python scripts/export_gui_assets.py from the project root.')
    st.stop()

summary = data_json('results/dashboard.json')
selected = summary.get('selection', {})

# ==============================================================================
# 1. DASHBOARD
# ==============================================================================
if page == 'Dashboard':
    encoded = base64.b64encode(text('app/assets/torquex_gt.svg').encode()).decode()
    st.markdown(f'''<div class="tx-hero"><div><div class="tx-tag">AUTOMOTIVE MACHINE LEARNING</div>
<h1>INTELLIGENCE.<br>IN <em>TOP GEAR.</em></h1>
<p>End-to-end car intelligence platform unifying vehicle price regression, transmission classification, and unsupervised market segmentation.</p>
<div class="tx-rules">
<div><b>10</b>REGRESSION MODELS</div>
<div><b>10</b>CLASSIFIERS</div>
<div><b>02</b>CLUSTERING ALGORITHMS</div>
</div></div>
<img class="tx-car" alt="TORQUEX red GT concept car" src="data:image/svg+xml;base64,{encoded}"></div>''', unsafe_allow_html=True)

    reg_best = summary.get('regression_best_model', {})
    cls_m = summary.get('selected_test_metrics', {})

    cards = [
        ('Price Regression (Review 1)', f"{reg_best.get('R2', 0.939):.3f}", 'R²', f"Best: {reg_best.get('Model', 'Random Forest')} (15,411 listings)"),
        ('Transmission Classification', f"{cls_m.get('F1_macro', 0.868):.3f}", 'Macro F1', f"Selected: {selected.get('name', 'Random Forest')} (2,500 balanced UK)"),
        ('Vehicle Segmentation', '4,009', 'cars', 'K-Means & Hierarchical 5-fold CV'),
        ('Total Evaluated Models', '22', 'models', '10 Regression + 10 Classifiers + 2 Clustering')
    ]
    st.markdown(
        '<div class="tx-kpis">' + ''.join(
            f'<div class="tx-kpi"><div class="label">{a}</div><div class="value">{b}<span class="unit">{c}</span></div><div class="note">{d}</div></div>'
            for a, b, c, d in cards
        ) + '</div>',
        unsafe_allow_html=True
    )

    regression_leaderboard = table('results/regression_models_comparison.csv').sort_values('R2', ascending=False)
    classification_leaderboard = table('results/ten_model_comparison.csv').sort_values('F1_macro', ascending=False)
    clustering_leaderboard = table('results/clustering_comparison.csv')
    clustering_leaderboard = clustering_leaderboard[
        clustering_leaderboard['Stage'] == 'After tuning'
    ].sort_values('CV_silhouette', ascending=False)

    left, right = st.columns([1, 1], gap='large')
    with left:
        st.markdown(
            racing_leaderboard(
                regression_leaderboard,
                'R2',
                lambda record: record['Family'],
                'REGRESSION GRAND PRIX',
                'R²',
                'regression'
            ),
            unsafe_allow_html=True
        )
        st.caption('Ranked by test R² on 3,083 held-out Cardekho listings. Higher is better.')

    with right:
        st.markdown(
            racing_leaderboard(
                classification_leaderboard,
                'F1_macro',
                lambda record: f'{record["Stage"]} · Part {record["Part"]}',
                'CLASSIFICATION SPRINT',
                'F1',
                'classification'
            ),
            unsafe_allow_html=True
        )
        st.caption('Ranked by macro F1 on the 500-row held-out split. Higher is better.')

    cluster_column, overview_column = st.columns([1, 1], gap='large')
    with cluster_column:
        st.markdown(
            racing_leaderboard(
                clustering_leaderboard,
                'CV_silhouette',
                lambda record: f'After tuning · k={int(record["k"])} · {record["linkage"]}',
                'CLUSTERING TIME ATTACK',
                'SIL',
                'clustering'
            ),
            unsafe_allow_html=True
        )
        st.caption('Ranked by mean five-fold validation silhouette after tuning. Higher is better.')

    with overview_column:
        section('MODEL COVERAGE')
        st.markdown(
            '<div class="tx-card"><div class="tx-tag">THREE PILLARS · ONE WORKSPACE</div>'
            '<p><b>1. Regression:</b> Price valuation in INR ₹ across 10 supervised algorithms.<br>'
            '<b>2. Classification:</b> Transmission prediction (Manual / Auto) in GBP £.<br>'
            '<b>3. Clustering:</b> Unsupervised market segments in USD $.<br><br>'
            'Select any tool from the navigation panel to run predictions or inspect benchmarks.</p></div>',
            unsafe_allow_html=True
        )

# ==============================================================================
# 2. PREDICT CAR PRICE (REGRESSION - ALL 10 MODELS)
# ==============================================================================
elif page == 'Predict car price (Regression)':
    heading(
        'REGRESSION / VEHICLE PRICE VALUATION',
        'Estimate selling price with 10 regression models.',
        'Configure any real-world car profile or upload a batch CSV. Choose between all 10 Review 1 regression models, or compare all 10 simultaneously.'
    )

    reg_summary_file = ROOT / 'results/regression_summary.json'
    reg_summary = data_json('results/regression_summary.json') if reg_summary_file.exists() else {}
    reg_inputs_file = ROOT / 'results/regression_inputs.json'
    reg_inputs = data_json('results/regression_inputs.json') if reg_inputs_file.exists() else {}

    model_options = {
        f"{info['Model']} ({info['Family']}) - R²: {info['R2']:.3f}": info['Slug']
        for info in reg_summary.get('models', [])
    }
    if not model_options:
        model_options = {name: slug for slug, name in REGRESSION_MODELS.items()}

    col1, col2 = st.columns([2.2, 1])
    selected_label = col1.selectbox('Selected regression algorithm', list(model_options.keys()), index=0, key='reg_model_select')
    chosen_slug = model_options[selected_label]

    reg_mode = st.radio('Valuation mode', ['Single car valuation', 'Batch CSV valuation'], horizontal=True, key='reg_mode')

    if reg_mode == 'Single car valuation':
        import pandas as pd
        brands = reg_inputs.get('brands', ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Mahindra', 'Tata', 'Volkswagen', 'BMW', 'Audi'])
        models_by_brand = reg_inputs.get('models_by_brand', {})

        b_col1, b_col2 = st.columns([1, 1.5])
        default_brand = 'Maruti' if 'Maruti' in brands else brands[0]
        brand = b_col1.selectbox('Manufacturer / Brand', brands, index=brands.index(default_brand), key='reg_brand')
        avail_models = models_by_brand.get(brand, [brand + ' Car'])
        default_model = avail_models[0]
        if 'Swift' in avail_models: default_model = 'Swift'
        elif 'City' in avail_models: default_model = 'City'
        elif 'i20' in avail_models: default_model = 'i20'

        car_model = b_col2.selectbox('Vehicle Model', avail_models, index=avail_models.index(default_model) if default_model in avail_models else 0, key=f'reg_model_{brand}')

        with st.form('reg_single_form'):
            c1, c2, c3 = st.columns(3)
            year = c1.number_input('Manufacture year', min_value=1995, max_value=2026, value=2019, step=1)
            vehicle_age = 2026 - year
            km_driven = c2.number_input('Kilometers driven (KM)', min_value=100, max_value=1000000, value=45000, step=2500)
            fuel_type = c3.selectbox('Fuel type', reg_inputs.get('fuel_types', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']), index=0)

            transmission_type = c1.selectbox('Transmission', reg_inputs.get('transmissions', ['Manual', 'Automatic']), index=0)
            seller_type = c2.selectbox('Seller type', reg_inputs.get('seller_types', ['Individual', 'Dealer', 'Trustmark Dealer']), index=0)
            seats = c3.selectbox('Seating capacity', reg_inputs.get('seats', [2, 4, 5, 7, 8]), index=2 if 5 in reg_inputs.get('seats', []) else 0)

            mileage = c1.number_input('Fuel economy / Mileage (kmpl)', min_value=5.0, max_value=40.0, value=20.5, step=0.5)
            engine = c2.number_input('Engine displacement (CC)', min_value=600, max_value=6500, value=1197, step=50)
            max_power = c3.number_input('Engine max power (bhp)', min_value=30.0, max_value=650.0, value=82.0, step=1.0)

            compare_all = st.checkbox('Compare estimated price across ALL 10 regression models simultaneously', value=True)
            submit_reg = st.form_submit_button('ESTIMATE SELLING PRICE', type='primary')

        if submit_reg:
            row_dict = {
                'vehicle_age': vehicle_age,
                'km_driven': km_driven,
                'mileage': mileage,
                'engine': engine,
                'max_power': max_power,
                'seats': seats,
                'brand': brand,
                'model': car_model,
                'seller_type': seller_type,
                'fuel_type': fuel_type,
                'transmission_type': transmission_type
            }
            X_input = pd.DataFrame([row_dict])

            fitted = model(f'regression_{chosen_slug}.joblib')
            pred_val = float(fitted.predict(X_input)[0])
            pred_val = max(10000.0, pred_val)

            st.markdown(
                f'<div class="tx-card" style="border-left: 4px solid #f03745; margin-top: 15px;">'
                f'<div class="tx-tag">ESTIMATED MARKET SELLING PRICE ({html.escape(REGRESSION_MODELS.get(chosen_slug, chosen_slug))})</div>'
                f'<div style="font-size: 38px; font-weight: 900; color: #fff; margin: 8px 0 4px;">{format_inr(pred_val)}</div>'
                f'<div style="font-size: 13px; color: #9da3b4;">Exact: ₹ {pred_val:,.0f} &nbsp;·&nbsp; Approx. $ {pred_val / 83.5:,.0f} USD &nbsp;·&nbsp; £ {pred_val / 106.0:,.0f} GBP</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Vehicle Age', f"{vehicle_age} yrs", f"Year {year}")
            m2.metric('Odometer', f"{km_driven:,} km")
            m3.metric('Power-to-Engine', f"{max_power / (engine/1000):.1f} bhp/L", f"{engine} cc")
            m4.metric('Fuel Economy', f"{mileage} kmpl", fuel_type)

            if compare_all:
                section('MULTI-MODEL REGRESSION CONSENSUS (ALL 10 MODELS)')
                all_preds = []
                for s_key, s_name in REGRESSION_MODELS.items():
                    try:
                        m_pipe = model(f'regression_{s_key}.joblib')
                        pv = float(m_pipe.predict(X_input)[0])
                        pv = max(10000.0, pv)
                        all_preds.append({
                            'Model': s_name,
                            'Slug': s_key,
                            'Estimated Price (INR)': pv,
                            'Price (Lakh)': round(pv / 1e5, 2),
                            'Formatted Price': format_inr(pv)
                        })
                    except Exception as err:
                        pass

                if all_preds:
                    comp_preds_df = pd.DataFrame(all_preds).sort_values('Estimated Price (INR)', ascending=False)
                    median_price = comp_preds_df['Estimated Price (INR)'].median()

                    fig = figure_bar(
                        comp_preds_df,
                        x='Price (Lakh)',
                        y='Model',
                        title='Price Predictions Across All 10 Models (in ₹ Lakhs)'
                    )
                    st.dataframe(comp_preds_df[['Model', 'Formatted Price', 'Price (Lakh)']], hide_index=True, width='stretch')
                    st.caption(f'Median consensus price across all 10 models: {format_inr(median_price)}.')

    else:
        st.write('Upload a CSV of car listings conforming to the 10-feature regression schema.')
        template_bytes = (ROOT / 'data/regression_prediction_template.csv').read_bytes() if (ROOT / 'data/regression_prediction_template.csv').exists() else b''
        st.download_button('DOWNLOAD REGRESSION CSV TEMPLATE', template_bytes, file_name='TORQUEX_regression_template.csv', mime='text/csv')

        with st.form('batch_reg_form'):
            reg_upload = st.file_uploader('Upload car listings CSV', type=['csv'], key='reg_batch_upload')
            run_reg_batch = st.form_submit_button('RUN BATCH VALUATION', type='primary')

        if run_reg_batch and reg_upload is None:
            st.info('Please select a CSV file first.')

        if run_reg_batch and reg_upload is not None:
            import pandas as pd
            from src.validation import validate_regression_batch
            try:
                raw_df = pd.read_csv(reg_upload)
                validated_df = validate_regression_batch(raw_df)
                fitted_model = model(f'regression_{chosen_slug}.joblib')

                out_df = validated_df.copy()
                preds = fitted_model.predict(validated_df[REGRESSION_RAW_FEATURES])
                out_df['predicted_price_inr'] = np.round(preds, 2)
                out_df['predicted_price_lakh'] = np.round(preds / 1e5, 2)
                out_df['formatted_price'] = [format_inr(p) for p in preds]

                st.success(f'Successfully calculated valuations for {len(out_df):,} vehicles using {REGRESSION_MODELS.get(chosen_slug, chosen_slug)}.')
                st.dataframe(out_df.head(100), hide_index=True, width='stretch')
                download(out_df, 'TORQUEX_batch_price_valuations.csv', 'DOWNLOAD ALL PREDICTIONS CSV')
            except Exception as e:
                st.error(f'Error processing batch CSV: {e}')

# ==============================================================================
# 3. REGRESSION BENCHMARKS (ALL 10 MODELS COMPARED & DIAGNOSTICS)
# ==============================================================================
elif page == 'Regression benchmarks':
    heading(
        'REGRESSION / MODEL BENCHMARKS & DIAGNOSTICS',
        '10 Car Valuation Models Tested.',
        'Comprehensive comparative evaluation across all 10 Review 1 regression models. Evaluated on 3,083 held-out Cardekho test listings with 5-fold cross-validation.'
    )

    reg_bench_mode = st.radio(
        'Benchmark section',
        ['Consolidated comparison (10 models)', 'Model diagnostics & curves', 'Held-out predictions sample'],
        horizontal=True,
        key='reg_bench_tab'
    )

    comp_csv = ROOT / 'results/regression_models_comparison.csv'
    if not comp_csv.exists():
        st.error('Regression comparison results missing. Run python src/regression.py.')
        st.stop()

    comp_df = table('results/regression_models_comparison.csv')

    if reg_bench_mode == 'Consolidated comparison (10 models)':
        metric_choice = st.selectbox(
            'Evaluation metric for ranking',
            ['R2', 'RMSE', 'MAE', 'CV_R2_mean', 'CV_R2_std', 'Fit_time_s'],
            format_func=lambda m: {
                'R2': 'Test R² Score (Higher is better)',
                'RMSE': 'Root Mean Squared Error (₹, Lower is better)',
                'MAE': 'Mean Absolute Error (₹, Lower is better)',
                'CV_R2_mean': '5-Fold Cross Validation R² (Higher is better)',
                'CV_R2_std': '5-Fold Cross Validation R² Variability (Lower is more stable)',
                'Fit_time_s': 'Training Fit Time (seconds)'
            }.get(m, m),
            key='reg_comp_metric'
        )

        yrange = [0, 1] if metric_choice in ['R2', 'CV_R2_mean'] else None
        figure_bar(
            comp_df,
            x='Model',
            y=metric_choice,
            color='Family',
            title=f'10 Regression Models Ranked by {metric_choice}',
            yrange=yrange
        )

        st.dataframe(comp_df, hide_index=True, width='stretch')
        download(comp_df, 'TORQUEX_10_regression_models.csv', 'DOWNLOAD REGRESSION BENCHMARK CSV')

        st.markdown(
            '<div class="tx-card" style="margin-top: 15px;">'
            '<div class="tx-tag">KEY TAKEAWAYS · REVIEW 1 TRACK</div>'
            '<p><b>Top Performer:</b> Random Forest Regressor achieves the highest test R² (0.939) and lowest MAE (₹99,007), closely followed by Gradient Boosting (0.921) and KNN (0.921).<br>'
            '<b>Linear Baselines:</b> Ordinary Least Squares Linear Regression, Ridge, and Lasso cluster closely around R² = 0.796–0.800.<br>'
            '<b>Regularization Effect:</b> ElasticNet balances L1 and L2 penalties with R² = 0.754. SVR demonstrates linear hyperplane constraints with R² = 0.418.<br>'
            '<b>CV Stability:</b> 5-fold cross-validation confirms consistent generalization with standard deviations below 0.10 across folds.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    elif reg_bench_mode == 'Model diagnostics & curves':
        reg_slug_list = comp_df['Slug'].tolist()
        diag_slug = st.selectbox(
            'Select model for diagnostic deep dive',
            reg_slug_list,
            format_func=lambda s: REGRESSION_MODELS.get(s, s),
            key='reg_diag_slug'
        )

        row = comp_df[comp_df.Slug == diag_slug].iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Test R² Score', f"{row['R2']:.4f}")
        k2.metric('Test RMSE', f"₹ {row['RMSE']:,.0f}")
        k3.metric('Test MAE', f"₹ {row['MAE']:,.0f}")
        k4.metric('5-Fold CV R²', f"{row['CV_R2_mean']:.4f}", f"± {row['CV_R2_std']:.4f}")

        diag_view = st.selectbox(
            'Diagnostic view',
            ['Actual vs Predicted Price', 'Residual Analysis', 'Feature Importance / Coefficients', 'Model Metadata & Parameters'],
            key=f'reg_diag_view_{diag_slug}'
        )

        slug_dir = ROOT / f'results/regression/{diag_slug}'
        if diag_view == 'Actual vs Predicted Price':
            p_img = slug_dir / 'pred_vs_actual.png'
            if p_img.exists(): st.image(str(p_img), width='stretch')
            st.caption('Scatter plot of actual selling prices versus predicted prices on 3,083 test cars with the ideal 45-degree reference line.')
        elif diag_view == 'Residual Analysis':
            r_img = slug_dir / 'residuals.png'
            if r_img.exists(): st.image(str(r_img), width='stretch')
            st.caption('Residual plot showing error distributions across the range of vehicle prices. Zero-line indicates unbiased estimates.')
        elif diag_view == 'Feature Importance / Coefficients':
            f_img = slug_dir / 'feature_importance.png'
            if f_img.exists():
                st.image(str(f_img), width='stretch')
                st.caption('Top features influencing vehicle valuation in this model.')
            else:
                st.info('Feature importances or coefficients not applicable for this model architecture.')
        else:
            meta_file = slug_dir / 'metadata.json'
            if meta_file.exists():
                st.json(data_json(f'results/regression/{diag_slug}/metadata.json'))

    else:
        section('TEST SET PREDICTIONS SAMPLE (3,083 TEST LISTINGS)')
        preds_file = ROOT / 'results/regression_test_predictions.csv'
        if preds_file.exists():
            preds_df = table('results/regression_test_predictions.csv')
            st.dataframe(preds_df.head(100), hide_index=True, width='stretch')
            download(preds_df, 'TORQUEX_regression_predictions_sample.csv', 'DOWNLOAD TEST PREDICTIONS CSV')
            st.caption('Showing actual vehicle selling prices alongside predictions from all 10 regression models on held-out test listings.')

# ==============================================================================
# 4. PREDICT TRANSMISSION (CLASSIFICATION)
# ==============================================================================
elif page == 'Predict transmission (Classification)':
    heading('CLASSIFICATION / VEHICLE CONFIGURATOR', 'Know your transmission.', 'Configure a real-world car profile. All ten classifiers use the balanced UK dataset: GBP prices and distance in miles.')
    options = {r['Model']: r['Slug'] for r in summary['models']}
    a, b = st.columns([2, 1])
    name = a.selectbox('Prediction model', list(options), index=list(options).index(selected['name']), key='predict_model')
    slug = options[name]
    stages = ['Before tuning', 'After tuning'] if slug in PART_B else ['Baseline']
    default = stages.index(selected['stage']) if slug == selected['slug'] and selected['stage'] in stages else len(stages) - 1
    stage = b.selectbox('Model version', stages, index=default, key=f'predict_stage_{slug}')
    filename = f'{slug}_{"after" if stage == "After tuning" else "before"}.joblib'
    mode = st.radio('Input mode', ['One car', 'Batch CSV'], horizontal=True, key='predict_mode')

    if mode == 'One car':
        import pandas as pd
        choices = summary['inputs']
        brand = st.selectbox('Manufacturer', choices['brands'], index=choices['brands'].index('Ford') if 'Ford' in choices['brands'] else 0, key='brand')
        with st.form('vehicle_form'):
            a, b, c = st.columns(3)
            car_model = a.selectbox('Model', choices['models_by_brand'][brand], key=f'car_model_{brand}')
            year = b.number_input('Model year', min_value=1970, max_value=2026, value=2018, step=1)
            fuel = c.selectbox('Fuel type', choices['fuels'], index=choices['fuels'].index('Petrol') if 'Petrol' in choices['fuels'] else 0)
            price = a.number_input('Advertised price (GBP)', min_value=1, value=16000, step=500)
            miles = b.number_input('Mileage (miles)', min_value=0, value=35000, step=1000)
            engine = c.number_input('Engine displacement (litres)', min_value=.1, max_value=10., value=1.5, step=.1)
            tax = a.number_input('Vehicle tax (GBP)', min_value=0, value=150, step=10)
            mpg = b.number_input('Fuel economy (source MPG)', min_value=.1, value=55., step=1.)
            compare = c.checkbox('Compare before and after', value=slug in PART_B, disabled=slug not in PART_B)
            submit = st.form_submit_button('RUN TRANSMISSION CHECK', type='primary')

        if submit:
            X = pd.DataFrame([dict(brand=brand, model=car_model, year=year, price_gbp=price, mileage_miles=miles,
                                 fuel_type=fuel, tax_gbp=tax, mpg_uk=mpg, engine_size_l=engine)])
            with st.spinner('Running the selected car profile…'):
                fitted = model(filename)
                label = fitted.predict(X)[0]
                p = float(probability(fitted, X)[0])
            st.success(f'Predicted transmission: {label}')
            a, b = st.columns(2)
            a.metric('Automatic model probability', f'{p:.1%}')
            b.metric('Manual model probability', f'{1-p:.1%}')
            if compare and slug in PART_B:
                records = []
                for suffix, title in [('before', 'Before tuning'), ('after', 'After tuning')]:
                    fitted = model(f'{slug}_{suffix}.joblib')
                    records.append({'Version': title, 'Prediction': fitted.predict(X)[0], 'Automatic probability': float(probability(fitted, X)[0])})
                st.dataframe(pd.DataFrame(records), hide_index=True, width='stretch')
            if not choices['year_min'] <= year <= choices['year_max']:
                st.warning(f'Training years range from {choices["year_min"]} to {choices["year_max"]}; this input is outside that range.')
        st.caption('Probabilities are uncalibrated model outputs. Results describe this balanced UK sample, not natural frequency in the whole market.')
    else:
        st.write('Use the template schema. You can upload up to 10,000 rows; target column is optional.')
        st.download_button('DOWNLOAD INPUT TEMPLATE', (ROOT / 'data/prediction_template.csv').read_bytes(), file_name='TORQUEX_input_template.csv', mime='text/csv')
        with st.form('batch_form'):
            upload = st.file_uploader('Car profiles CSV', type=['csv'], key='batch_upload')
            run = st.form_submit_button('RUN BATCH PREDICTION', type='primary')
        if run and upload is None: st.info('Choose a CSV file before running prediction.')
        if upload is not None and run:
            import pandas as pd
            from src.validation import validate_batch
            try:
                frame = validate_batch(pd.read_csv(upload))
                fitted = model(filename)
                output = frame.copy()
                output['predicted_transmission'] = fitted.predict(frame[RAW_FEATURES])
                output['probability_automatic'] = probability(fitted, frame[RAW_FEATURES])
                st.success(f'Predicted {len(output):,} car profiles.')
                st.dataframe(output.head(100), hide_index=True, width='stretch')
                download(output, 'TORQUEX_predictions.csv', 'DOWNLOAD ALL PREDICTIONS')
            except Exception as exc:
                st.error(str(exc))

# ==============================================================================
# 5. CLASSIFICATION BENCHMARKS
# ==============================================================================
elif page == 'Classification benchmarks':
    heading('PERFORMANCE / CLASSIFIER BENCHMARK', 'Every change, measured.', 'Baseline, tuned and after-tuning comparisons from the 500-row held-out split. Automatic is the positive class.')
    mode = st.radio('Comparison view', ['Before vs after', 'After tuning', 'All ten models', 'Model diagnostics'], horizontal=True, key='comparison_view')
    metric = st.selectbox(
        'Score',
        [
            'F1_macro', 'F1_weighted', 'Accuracy', 'Precision_weighted', 'Recall_weighted',
            'ROC_AUC', 'Average_precision_Automatic', 'Automatic_recall',
            'Automatic_precision', 'Balanced_accuracy', 'CV_macro_F1', 'CV_macro_F1_std'
        ],
        key='comparison_metric'
    )

    if mode == 'Before vs after':
        frame = table('results/part_b_before_after.csv')
        figure_bar(frame, 'Model', metric, 'Stage', 'group', 'Five Part B models: before and after tuning', yrange=[0, 1])
        st.dataframe(frame, hide_index=True, width='stretch')
        section('TUNING DELTAS / AFTER MINUS BEFORE')
        st.dataframe(table('results/tuning_deltas.csv'), hide_index=True, width='stretch')
        download(frame, 'TORQUEX_before_after.csv')
    elif mode == 'After tuning':
        frame = table('results/after_tuning_comparison.csv')
        figure_bar(frame, 'Model', metric, title='All five Part B models after tuning', yrange=[0, 1])
        st.caption('Every row is the tuned variant, including any whose best parameters match baseline.')
        st.dataframe(frame, hide_index=True, width='stretch')
        download(frame, 'TORQUEX_after_tuning.csv')
    elif mode == 'All ten models':
        frame = table('results/ten_model_comparison.csv')
        figure_bar(frame, 'Model', metric, 'Part', title='Ten-model consolidated comparison', yrange=[0, 1])
        st.dataframe(frame, hide_index=True, width='stretch')
        download(frame, 'TORQUEX_ten_models.csv')
    else:
        slug = st.selectbox('Part B model', list(PART_B), format_func=PART_B.get, key='diagnostic_model')
        st.dataframe(table(f'results/{slug}/comparison.csv'), hide_index=True, width='stretch')
        chart = st.selectbox('Diagnostic', ['Confusion matrices', 'ROC and precision–recall', 'Class reports', 'Search results', 'Model details'], key='diagnostic_chart')
        if chart in ['Confusion matrices', 'ROC and precision–recall']:
            filename = 'confusion_matrices.png' if chart == 'Confusion matrices' else 'roc_pr.png'
            st.image(str(ROOT / f'results/{slug}/{filename}'), width='stretch')
        elif chart == 'Class reports':
            a, b = st.columns(2)
            a.write('Before tuning'); a.dataframe(table(f'results/{slug}/before_class_report.csv'), hide_index=True)
            b.write('After tuning'); b.dataframe(table(f'results/{slug}/after_class_report.csv'), hide_index=True)
        elif chart == 'Search results':
            st.dataframe(table(f'results/{slug}/search_results.csv'), hide_index=True, width='stretch')
        else:
            metadata = data_json(f'results/{slug}/metadata.json')
            st.json(metadata['best_parameters'])
            st.write('Training warnings:', metadata['warnings'] or 'None')
            for filename in ['feature_importance.png', 'training_loss.png']:
                if (ROOT / f'results/{slug}/{filename}').exists():
                    st.image(str(ROOT / f'results/{slug}/{filename}'), width='stretch')

# ==============================================================================
# 6. CAR CLUSTERS
# ==============================================================================
elif page == 'Car clusters':
    heading('CLUSTERING / VEHICLE SEGMENTS', 'Find a car’s company.', 'The original 4,009-row used-car dataset, grouped by age, price and mileage. Prices here are USD; distances are miles.')
    kind = st.selectbox('Clustering model', ['kmeans', 'agglomerative'], format_func=lambda x: 'K-Means' if x == 'kmeans' else 'Agglomerative hierarchical', key='cluster_model')
    mode = st.radio('Cluster view', ['Segments', 'Before vs after', 'After tuning', 'Diagnostics', 'Assign a car'], horizontal=True, key='cluster_view')

    if mode == 'Segments':
        import plotly.express as px
        frame = table(f'results/{kind}/clustered_training_cars.csv')
        frame['Segment'] = 'Cluster ' + frame.cluster.astype(str)
        profiles = table(f'results/{kind}/cluster_profiles.csv')
        choices = sorted(frame.Segment.unique())
        chosen = st.multiselect('Visible segments', choices, default=choices, key=f'segments_{kind}')
        subset = frame[frame.Segment.isin(chosen)]
        if subset.empty:
            st.info('Choose a segment to show its cars.')
        else:
            fig = px.scatter(
                subset, x='PC1', y='PC2', color='Segment', hover_data=['brand', 'model', 'price_usd', 'mileage_miles', 'car_age'],
                color_discrete_sequence=['#ff5765', '#7fa7e3', '#f4bf73', '#9ec9a8', '#c8a7ef', '#c9ced7'], opacity=.7
            )
            fig.update_traces(marker_size=5)
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=5, r=5, t=10, b=5), height=440)
            st.plotly_chart(fig, width='stretch')
        st.dataframe(profiles, hide_index=True, width='stretch')
        with st.expander('Preview selected listings'):
            st.dataframe(subset[['brand', 'model', 'price_usd', 'mileage_miles', 'car_age', 'cluster']].head(100), hide_index=True, width='stretch')
        download(subset.drop(columns='Segment'), 'TORQUEX_clustered_cars.csv', 'DOWNLOAD SELECTED LISTINGS')
    elif mode in ['Before vs after', 'After tuning']:
        frame = table('results/clustering_comparison.csv')
        if mode == 'After tuning': frame = frame[frame.Stage == 'After tuning']
        metric = st.selectbox(
            'Cluster metric',
            [
                'Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz',
                'CV_silhouette', 'CV_silhouette_std', 'Holdout_Silhouette',
                'Holdout_Davies_Bouldin', 'Holdout_Calinski_Harabasz'
            ],
            key='cluster_metric'
        )
        import plotly.express as px
        fig = px.bar(frame, x='Model', y=metric, color='Stage', barmode='group', color_discrete_sequence=['#e53949', '#a0b4cf'])
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')
        st.dataframe(frame, hide_index=True, width='stretch')
        download(frame, 'TORQUEX_clustering_comparison.csv')
    elif mode == 'Diagnostics':
        choices = {'PCA before and after': 'pca_comparison.png', 't-SNE': 'tsne.png',
                   ('Elbow curve' if kind == 'kmeans' else 'Dendrogram'): ('elbow.png' if kind == 'kmeans' else 'dendrogram.png')}
        chart = st.selectbox('Cluster diagnostic', list(choices) + ['Five-fold tuning search'], key=f'cluster_diagnostic_{kind}')
        if chart == 'Five-fold tuning search':
            search_results = table(f'results/{kind}/search_results.csv').drop(columns='linkage', errors='ignore')
            st.dataframe(search_results, hide_index=True, width='stretch')
        else:
            st.image(str(ROOT / f'results/{kind}/{choices[chart]}'), width='stretch')
    else:
        if kind == 'agglomerative': st.info('This assigns the new car to its nearest learned cluster centre; it does not rebuild hierarchy.')
        with st.form('cluster_form'):
            a, b, c = st.columns(3)
            year = a.number_input('Model year', 1970, 2026, 2018, step=1)
            price = b.number_input('Price (USD)', min_value=1, value=25000, step=500)
            miles = c.number_input('Mileage (miles)', min_value=0, value=50000, step=1000)
            submit = st.form_submit_button('FIND SIMILAR SEGMENT', type='primary')
        if submit:
            import pandas as pd
            from src.clustering import predict_new
            bundle = model(f'{kind}_after.joblib')
            label = int(predict_new(bundle, pd.DataFrame([{'car_age': 2026 - year, 'price_usd': price, 'mileage_miles': miles}]))[0])
            st.success(f'Assigned segment: Cluster {label}')
            profiles = table(f'results/{kind}/cluster_profiles.csv')
            st.dataframe(profiles[profiles.cluster == label], hide_index=True, width='stretch')
            cars = table(f'results/{kind}/clustered_training_cars.csv')
            st.write('Example listings in this segment')
            st.dataframe(cars[cars.cluster == label][['brand', 'model', 'price_usd', 'mileage_miles', 'car_age']].head(10), hide_index=True, width='stretch')

# ==============================================================================
# 7. EVALUATION METRICS
# ==============================================================================
elif page == 'Evaluation metrics':
    import pandas as pd

    heading(
        'EVALUATION / ALL MODELS',
        'Every model. Every recorded metric.',
        'Complete held-out, cross-validation and clustering results for all 10 regressors, 10 classifiers and 2 clustering algorithms.'
    )
    st.info(
        'Higher is better for R2, accuracy, precision, recall, F1, ROC-AUC, average precision, silhouette and Calinski-Harabasz. '
        'Lower is better for RMSE, MAE and Davies-Bouldin. Lower standard deviation indicates more stable cross-validation.'
    )

    regression_tab, classification_tab, clustering_tab = st.tabs(
        ['Regression - 10 models', 'Classification - 10 models', 'Clustering - 2 models']
    )

    with regression_tab:
        regression = table('results/regression_models_comparison.csv').copy()
        section('HELD-OUT TEST AND 5-FOLD CROSS-VALIDATION METRICS')
        regression_display = regression[
            ['Model', 'Family', 'R2', 'RMSE', 'MAE', 'CV_R2_mean', 'CV_R2_std', 'Fit_time_s']
        ].rename(columns={
            'R2': 'Test R2',
            'RMSE': 'Test RMSE (INR)',
            'MAE': 'Test MAE (INR)',
            'CV_R2_mean': 'CV R2 mean',
            'CV_R2_std': 'CV R2 std',
            'Fit_time_s': 'Fit time (s)'
        }).round(4)
        st.dataframe(regression_display, hide_index=True, width='stretch', height=420)
        section('ALL REGRESSION METRIC GRAPHS')
        metric_charts(
            regression,
            ['R2', 'RMSE', 'MAE', 'CV_R2_mean', 'CV_R2_std', 'Fit_time_s'],
            'Regression metrics across all 10 models',
            'all_regression_metric_charts',
            labels={
                'R2': 'Test R2',
                'RMSE': 'Test RMSE (INR)',
                'MAE': 'Test MAE (INR)',
                'CV_R2_mean': 'CV R2 mean',
                'CV_R2_std': 'CV R2 standard deviation',
                'Fit_time_s': 'Fit time (seconds)'
            }
        )
        st.caption('Test metrics use the 3,083-row held-out set. CV values summarize five training folds.')
        download(regression, 'TORQUEX_all_regression_metrics.csv', 'DOWNLOAD ALL REGRESSION METRICS')

    with classification_tab:
        classification = table('results/ten_model_comparison.csv').copy()

        section('HELD-OUT TEST METRICS')
        held_out = classification[
            [
                'Model', 'Part', 'Stage', 'Accuracy', 'Precision_weighted',
                'Recall_weighted', 'F1_weighted', 'F1_macro', 'Balanced_accuracy',
                'Automatic_precision', 'Automatic_recall', 'ROC_AUC',
                'Average_precision_Automatic'
            ]
        ].rename(columns={
            'Precision_weighted': 'Weighted precision',
            'Recall_weighted': 'Weighted recall',
            'F1_weighted': 'Weighted F1',
            'F1_macro': 'Macro F1',
            'Balanced_accuracy': 'Balanced accuracy',
            'Automatic_precision': 'Automatic precision',
            'Automatic_recall': 'Automatic recall',
            'ROC_AUC': 'ROC-AUC',
            'Average_precision_Automatic': 'Automatic average precision'
        }).round(4)
        st.dataframe(held_out, hide_index=True, width='stretch', height=420)
        section('ALL HELD-OUT CLASSIFICATION METRIC GRAPHS')
        metric_charts(
            classification,
            [
                'Accuracy', 'Precision_weighted', 'Recall_weighted', 'F1_weighted',
                'F1_macro', 'Balanced_accuracy', 'Automatic_precision',
                'Automatic_recall', 'ROC_AUC', 'Average_precision_Automatic'
            ],
            'Held-out classification metrics across all 10 models',
            'all_classification_heldout_metric_charts',
            labels={
                'Precision_weighted': 'Weighted precision',
                'Recall_weighted': 'Weighted recall',
                'F1_weighted': 'Weighted F1',
                'F1_macro': 'Macro F1',
                'Balanced_accuracy': 'Balanced accuracy',
                'Automatic_precision': 'Automatic precision',
                'Automatic_recall': 'Automatic recall',
                'ROC_AUC': 'ROC-AUC',
                'Average_precision_Automatic': 'Automatic average precision'
            },
            yrange=[0, 1]
        )

        class_rows = []
        validation_rows = []
        for result in classification.to_dict('records'):
            slug = result['Slug']
            stage = result['Stage']
            report_name = 'after_class_report.csv' if stage == 'After tuning' else 'before_class_report.csv'
            report = table(f'results/{slug}/{report_name}').copy()
            report = report.set_index(report.columns[0])
            automatic = report.loc['Automatic']
            manual = report.loc['Manual']
            class_rows.append({
                'Model': result['Model'],
                'Stage': stage,
                'Automatic precision': automatic['precision'],
                'Automatic recall': automatic['recall'],
                'Automatic F1': automatic['f1-score'],
                'Automatic support': int(automatic['support']),
                'Manual precision': manual['precision'],
                'Manual recall': manual['recall'],
                'Manual F1': manual['f1-score'],
                'Manual support': int(manual['support'])
            })

            if stage == 'After tuning':
                search = table(f'results/{slug}/search_results.csv')
                best = search.sort_values(
                    ['rank_test_f1_macro', 'mean_test_f1_macro'],
                    ascending=[True, False]
                ).iloc[0]
                weighted_mean = best['mean_test_f1_weighted']
                weighted_std = best['std_test_f1_weighted']
                balanced_mean = best['mean_test_balanced_accuracy']
                balanced_std = best['std_test_balanced_accuracy']
            else:
                folds = table(f'results/{slug}/baseline_cv_folds.csv')
                weighted_mean = folds['test_f1_weighted'].mean()
                weighted_std = folds['test_f1_weighted'].std(ddof=0)
                balanced_mean = folds['test_balanced_accuracy'].mean()
                balanced_std = folds['test_balanced_accuracy'].std(ddof=0)

            validation_rows.append({
                'Model': result['Model'],
                'Stage': stage,
                'CV macro F1 mean': result['CV_macro_F1'],
                'CV macro F1 std': result['CV_macro_F1_std'],
                'CV weighted F1 mean': weighted_mean,
                'CV weighted F1 std': weighted_std,
                'CV balanced accuracy mean': balanced_mean,
                'CV balanced accuracy std': balanced_std
            })

        section('CLASS-LEVEL PRECISION, RECALL, F1 AND SUPPORT')
        class_metrics = pd.DataFrame(class_rows).round(4)
        st.dataframe(class_metrics, hide_index=True, width='stretch', height=420)
        section('ALL CLASS-LEVEL METRIC GRAPHS')
        metric_charts(
            class_metrics,
            [
                'Automatic precision', 'Automatic recall', 'Automatic F1',
                'Manual precision', 'Manual recall', 'Manual F1'
            ],
            'Automatic and Manual class metrics across all 10 models',
            'all_class_level_metric_charts',
            yrange=[0, 1]
        )

        section('5-FOLD CROSS-VALIDATION METRICS')
        validation_metrics = pd.DataFrame(validation_rows).round(4)
        st.dataframe(validation_metrics, hide_index=True, width='stretch', height=420)
        section('ALL CLASSIFICATION CROSS-VALIDATION GRAPHS')
        metric_charts(
            validation_metrics,
            [
                'CV macro F1 mean', 'CV macro F1 std',
                'CV weighted F1 mean', 'CV weighted F1 std',
                'CV balanced accuracy mean', 'CV balanced accuracy std'
            ],
            'Five-fold classification metrics across all 10 models',
            'all_classification_cv_metric_charts'
        )
        st.caption('Part A rows use baseline folds. Part B rows use the stage selected for the final ten-model comparison.')
        download(classification, 'TORQUEX_all_classification_metrics.csv', 'DOWNLOAD ALL CLASSIFICATION METRICS')
        download(class_metrics, 'TORQUEX_class_level_metrics.csv', 'DOWNLOAD CLASS-LEVEL METRICS')
        download(validation_metrics, 'TORQUEX_classification_cv_metrics.csv', 'DOWNLOAD CLASSIFICATION CV METRICS')

    with clustering_tab:
        clustering = table('results/clustering_comparison.csv').copy()
        cv_davies_bouldin = []
        cv_calinski_harabasz = []

        for result in clustering.to_dict('records'):
            algorithm_slug = 'kmeans' if result['Model'] == 'K-Means' else 'agglomerative'
            search = table(f'results/{algorithm_slug}/search_results.csv')
            selected = search[search['k'].eq(result['k'])]
            if algorithm_slug == 'agglomerative':
                selected = selected[selected['linkage'].eq(result['linkage'])]
            selected = selected.iloc[0]
            cv_davies_bouldin.append(selected['CV_Davies_Bouldin'])
            cv_calinski_harabasz.append(selected['CV_Calinski_Harabasz'])

        clustering['CV_Davies_Bouldin'] = cv_davies_bouldin
        clustering['CV_Calinski_Harabasz'] = cv_calinski_harabasz

        section('FITTED TRAINING-PARTITION METRICS')
        fitted_metrics = clustering[
            ['Model', 'Stage', 'k', 'linkage', 'Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz']
        ].rename(columns={
            'Davies_Bouldin': 'Davies-Bouldin',
            'Calinski_Harabasz': 'Calinski-Harabasz'
        }).round(4)
        st.dataframe(fitted_metrics, hide_index=True, width='stretch')
        section('ALL FITTED CLUSTERING METRIC GRAPHS')
        metric_charts(
            clustering,
            ['Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz'],
            'Training-partition clustering metrics',
            'all_fitted_clustering_metric_charts',
            labels={
                'Davies_Bouldin': 'Davies-Bouldin',
                'Calinski_Harabasz': 'Calinski-Harabasz'
            },
            group='Stage'
        )

        section('5-FOLD CROSS-VALIDATION METRICS')
        clustering_cv = clustering[
            [
                'Model', 'Stage', 'CV_silhouette', 'CV_silhouette_std',
                'CV_Davies_Bouldin', 'CV_Calinski_Harabasz'
            ]
        ].rename(columns={
            'CV_silhouette': 'CV silhouette mean',
            'CV_silhouette_std': 'CV silhouette std',
            'CV_Davies_Bouldin': 'CV Davies-Bouldin',
            'CV_Calinski_Harabasz': 'CV Calinski-Harabasz'
        }).round(4)
        st.dataframe(clustering_cv, hide_index=True, width='stretch')
        section('ALL CLUSTERING CROSS-VALIDATION GRAPHS')
        metric_charts(
            clustering,
            [
                'CV_silhouette', 'CV_silhouette_std',
                'CV_Davies_Bouldin', 'CV_Calinski_Harabasz'
            ],
            'Five-fold clustering metrics',
            'all_clustering_cv_metric_charts',
            labels={
                'CV_silhouette': 'CV silhouette mean',
                'CV_silhouette_std': 'CV silhouette standard deviation',
                'CV_Davies_Bouldin': 'CV Davies-Bouldin',
                'CV_Calinski_Harabasz': 'CV Calinski-Harabasz'
            },
            group='Stage'
        )

        section('HELD-OUT CLUSTERING METRICS')
        clustering_holdout = clustering[
            [
                'Model', 'Stage', 'Holdout_Silhouette',
                'Holdout_Davies_Bouldin', 'Holdout_Calinski_Harabasz'
            ]
        ].rename(columns={
            'Holdout_Silhouette': 'Holdout silhouette',
            'Holdout_Davies_Bouldin': 'Holdout Davies-Bouldin',
            'Holdout_Calinski_Harabasz': 'Holdout Calinski-Harabasz'
        }).round(4)
        st.dataframe(clustering_holdout, hide_index=True, width='stretch')
        section('ALL HELD-OUT CLUSTERING METRIC GRAPHS')
        metric_charts(
            clustering,
            ['Holdout_Silhouette', 'Holdout_Davies_Bouldin', 'Holdout_Calinski_Harabasz'],
            'Held-out clustering metrics',
            'all_clustering_holdout_metric_charts',
            labels={
                'Holdout_Silhouette': 'Holdout silhouette',
                'Holdout_Davies_Bouldin': 'Holdout Davies-Bouldin',
                'Holdout_Calinski_Harabasz': 'Holdout Calinski-Harabasz'
            },
            group='Stage'
        )
        st.caption('Rows cover both before- and after-tuning configurations for K-Means and Agglomerative clustering.')
        download(clustering, 'TORQUEX_all_clustering_metrics.csv', 'DOWNLOAD ALL CLUSTERING METRICS')

st.markdown('<div class="tx-footer">TORQUEX / AUTOMOTIVE INTELLIGENCE &nbsp; · &nbsp; REVIEWS 01 & 02 UNIFIED &nbsp; · &nbsp; 22 MODELS</div>', unsafe_allow_html=True)
