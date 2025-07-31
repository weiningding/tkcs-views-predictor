import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re

@st.cache_data
def load_and_train_model():
    df = pd.read_excel('cleaned_with_links_added.xlsx')
    df = df.dropna(subset=['video views', 'length'])

    # Normalize casing for consistency
    df['channel'] = df['channel'].astype(str).str.strip().str.title()
    df['content'] = df['content'].astype(str).str.strip().str.title()
    df['time'] = df['time'].astype(str).str.strip().str.lower()

    df['Video Views'] = np.log(df['video views'] + 1)
    df['Length_log'] = np.log(df['length'] + 1)
    df['Length_log_sq'] = df['Length_log'] ** 2

    dummies = pd.get_dummies(df[['content', 'channel', 'time']], drop_first=True)
    X = pd.concat([df[['Length_log', 'Length_log_sq']], dummies], axis=1)

    for col in [c for c in dummies.columns if c.startswith('Channel_')]:
        X[f'Length_log*{col}'] = df['Length_log'] * dummies[col]

    X = sm.add_constant(X)
    y = df['Video Views']
    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = sm.OLS(y_train, X_train).fit()

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return model, dummies.columns.tolist(), mae, mse, rmse, r2, df

def get_thumbnail(url):
    if "youtube.com" in url or "youtu.be" in url:
        match = re.search(r'(?:v=|be/)([a-zA-Z0-9_-]{11})', url)
        if match:
            return f"https://img.youtube.com/vi/{match.group(1)}/0.jpg"
    elif "tiktok.com" in url:
        return "https://via.placeholder.com/320x180.png?text=TikTok+Preview"
    elif "facebook.com" in url:
        return "https://via.placeholder.com/320x180.png?text=Facebook+Preview"
    return "https://via.placeholder.com/320x180.png?text=No+Preview"

model, dummy_columns, mae, mse, rmse, r2, df_full = load_and_train_model()

print("Dummy columns used in model:", dummy_columns)  # Debug print for dummy columns

st.title("🎬 The Kelly Clarkson Show: Video Insights Tool")

if 'page' not in st.session_state:
    st.session_state.page = 'home'

col1, col2 = st.columns(2)
with col1:
    if st.button("🎯 Video Views Predictor"):
        st.session_state.page = 'predictor'
with col2:
    if st.button("🔍 Optimal Characteristics"):
        st.session_state.page = 'characteristics'

def run_predictor():
    # Use unique normalized values for dropdowns:
    content_options = sorted(df_full['content'].dropna().unique())
    if "Edit" not in content_options:
        content_options = ["Edit"] + content_options
    channel_options = sorted(df_full['channel'].dropna().unique())
    if "Facebook" not in channel_options:
        channel_options = ["Facebook"] + channel_options
    time_options = sorted(df_full['time'].dropna().unique())
    if "afternoon" not in time_options:
        time_options = ["afternoon"] + time_options

    length = st.number_input("Video Length (seconds):", min_value=1, value=60)
    content = st.selectbox("Content Type:", content_options)
    channel = st.selectbox("Channel:", channel_options)
    time_of_day = st.selectbox("Time of Day Posted:", time_options)

    def predict_views(length, content, channel, time_of_day):
        input_dict = {
            'Length_log': [np.log(length + 1)],
            'Length_log_sq': [np.log(length + 1) ** 2]
        }
        # Reset all dummy vars to 0
        for col in dummy_columns:
            input_dict[col] = [0]

        if content != "Edit" and f'content_{content}' in dummy_columns:
            input_dict[f'content_{content}'] = [1]
        if channel != "facebook" and f'channel_{channel}' in dummy_columns:
            input_dict[f'channel_{channel}'] = [1]
            length_log = np.log(length + 1)
            interaction_col = f'Length_log*channel_{channel}'
            if interaction_col in dummy_columns:
                input_dict[interaction_col] = [length_log]
        if time_of_day != "afternoon" and f'time_{time_of_day}' in dummy_columns:
            input_dict[f'time_{time_of_day}'] = [1]

        print("Input dictionary for prediction:")
        print(input_dict)  # Debug print for prediction input dictionary

        # Flatten input_dict values from lists to scalars for DataFrame creation
        flat_input_dict = {k: v[0] if isinstance(v, list) else v for k, v in input_dict.items()}
        input_df = pd.DataFrame([flat_input_dict])  # one-row DataFrame

        input_df = sm.add_constant(input_df, has_constant='add')
        input_df = input_df.reindex(columns=model.model.exog_names, fill_value=0)

        # Ensure all columns are float type to prevent dtype=object error
        input_df = input_df.astype(float)

        prediction = model.get_prediction(input_df)
        pred_summary = prediction.summary_frame(alpha=0.05)

        log_pred = pred_summary['mean'].iloc[0]
        log_ci_lower = pred_summary['mean_ci_lower'].iloc[0]
        log_ci_upper = pred_summary['mean_ci_upper'].iloc[0]
        log_pi_lower = pred_summary['obs_ci_lower'].iloc[0]
        log_pi_upper = pred_summary['obs_ci_upper'].iloc[0]

        # Bias correction using residual variance
        residual_var = model.mse_resid
        correction = np.exp(residual_var / 2)

        pred_views = (np.exp(log_pred) - 1) * correction
        ci_lower = (np.exp(log_ci_lower) - 1) * correction
        ci_upper = (np.exp(log_ci_upper) - 1) * correction
        pi_lower = (np.exp(log_pi_lower) - 1) * correction
        pi_upper = (np.exp(log_pi_upper) - 1) * correction

        return pred_views, ci_lower, ci_upper, pi_lower, pi_upper

    if st.button("🔮 Predict Video Views"):
        predicted_views, ci_lower, ci_upper, pi_lower, pi_upper = predict_views(length, content, channel, time_of_day)

        st.success(f"### Predicted Views: {predicted_views:,.0f}")
        st.write(f"**95% Confidence Interval (mean):** {ci_lower:,.0f} – {ci_upper:,.0f}")

        df_full['length_diff'] = np.abs(df_full['length'] - length)

        similar = df_full.copy()
        if channel in similar['channel'].values:
            similar = similar[similar['channel'] == channel]
        if content in similar['content'].values:
            similar = similar[similar['content'] == content]

        similar = similar.sort_values(by='length_diff')
        similar = pd.concat([
            similar[similar['time'] == time_of_day],
            similar[similar['time'] != time_of_day]
        ])

        similar_videos = similar.head(3)

        st.subheader("🎬 Similar Videos")
        cols = st.columns(3)
        for i, (_, row) in enumerate(similar_videos.iterrows()):
            with cols[i]:
                st.image(get_thumbnail(row['post link']), caption=row['channel'], use_container_width=True)
                st.markdown(f"**Views:** {int(row['video views']):,}")
                st.markdown(f"**Length:** {int(row['length'])} sec")
                st.markdown(f"**Time of Day:** {row['time'].capitalize()}")
                st.markdown(f"**Channel:** {row['channel']}")
                st.markdown(f"**Content Type:** {row['content']}")
                st.markdown(f"[▶ Watch Video]({row['post link']})", unsafe_allow_html=True)

    with st.expander("📈 Model Performance (Log Scale)"):
        st.write(f"**MAE:** {mae:.3f}")
        st.write(f"**MSE:** {mse:.3f}")
        st.write(f"**RMSE:** {rmse:.3f}")
        st.write(f"**R²:** {r2:.3f}")

def show_optimal_characteristics():
    st.header("🔍 Characteristics That Maximize Views")

    coefs = model.params.drop("const")

    baseline_views = 10000
    def coef_to_extra_views(coef, baseline=baseline_views):
        return baseline * (np.exp(coef) - 1)

    coef_df = pd.DataFrame({
        'Feature': coefs.index,
        'Coefficient': coefs.values,
        'Extra Views': coef_to_extra_views(coefs.values)
    })
    coef_df = coef_df.sort_values(by='Extra Views', ascending=False)

    st.markdown("### Top Positive Drivers of Video Views")

    # Bullet points for length and time of day
    length_coef = coefs.get('Length_log', None)
    length_sq_coef = coefs.get('Length_log_sq', None)
    time_coefs = {k: v for k, v in coefs.items() if k.startswith('time_')}

    if time_coefs:
        best_time, best_time_coef = max(time_coefs.items(), key=lambda x: coef_to_extra_views(x[1]))
        extra_views_time = coef_to_extra_views(best_time_coef)
        time_name = best_time.replace("time_", "")
        st.markdown(f"- **Time of day posted:** Videos posted during **{time_name}** tend to get approximately **{int(extra_views_time):,} more views** than those posted in the afternoon.")

   explanations = []
excluded_feats = {'Length_log', 'Length_log_sq'}.union(time_coefs.keys())
    for feat, extra_views in zip(coef_df['Feature'], coef_df['Extra Views']):
        if feat in excluded_feats:
            continue
        if feat.startswith("Length_log*Channel_"):
            channel_name = feat.replace("Length_log*Channel_", "").replace("_", " ").title()
            explanations.append(f"{channel_name}")
        elif feat.startswith("Channel_"):
            channel_name = feat.replace("Channel_", "").replace("_", " ").title()
            explanations.append(f"{channel_name}")
        elif feat.startswith("Content_"):
            content_name = feat.replace("Content_", "").replace("_", " ").title()
            explanations.append(f"{content_name}")
        else:
            explanations.append(f"{feat}")


    for explanation in explanations[:5]:
        st.markdown(f"- {explanation}")

    top_features = coef_df[~coef_df['Feature'].isin(excluded_feats)].head(3)['Feature'].tolist()
    filtered = df_full.copy()
    for feat in top_features:
        if feat in filtered.columns:
            median_val = filtered[feat].median()
            filtered = filtered[filtered[feat] > median_val]

    top_examples = filtered.sort_values('video views', ascending=False).head(3)

    st.subheader("🎥 Example High-Performing Videos")
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_examples.iterrows()):
        with cols[i]:
            st.image(get_thumbnail(row['post link']), caption=row['channel'], use_container_width=True)
            st.markdown(f"**Views:** {int(row['video views']):,}")
            st.markdown(f"**Length:** {int(row['length'])} sec")
            st.markdown(f"**Time of Day:** {row['time'].capitalize()}")
            st.markdown(f"**Channel:** {row['channel']}")
            st.markdown(f"**Content Type:** {row['content']}")
            st.markdown(f"[▶ Watch Video]({row['post link']})", unsafe_allow_html=True)

if st.session_state.page == 'predictor':
    run_predictor()
elif st.session_state.page == 'characteristics':
    show_optimal_characteristics()
else:
    st.write("Welcome! Use the buttons above to start.")
