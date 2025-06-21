import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import sklearn
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(
    page_title=(("Taxi Fare Predictor"))",
    layout="centered",
    page_icon="🚕",
)

st.title("Taxi Fare Prediction App 🚕")

## Step 01 - Setup
st.sidebar.title("Explore different taxi fares🚕")
page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊", "Automated Report 📑","Prediction"])

st.write("   ")
st.write("   ")
st.write("   ")
df = pd.read_csv("updated_taxidata.csv")
# Make column names unique if there are duplicates
df.columns = ['Vendor ID',
    'Pickup time',
    'Dropoff time',
    'Passenger count',
    'Trip distance',
    'Payment type',
    'Fare amount',
    'Extra',
    'MTA tax',
    'Tip amount',
    'Tolls amount',
    'Improvement surcharge',
    'Total amount',
    'Congestion surcharge',]


## Step 02 - Load dataset
if page == "Introduction 📘":

    st.image("/workspaces/world-gni/nyc-yellow-taxi-in-times-square-hero.jpg.webp", use_container_width=True)
    st.subheader("01 Introduction 📘")

    st.markdown("##### Overview")
    st.write("This app allows you to explore and visualize data relating to New York City taxi fares. It compiles factors such as the distance traveled, time of day, number of passengers, etc to predict total costs.")
    st.write("Note: The data shown here is a sample of a much larger dataset, which was cut from over 1 million rows to 20,000 because of storage limitations.")

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    st.success("✅ No missing values found")

elif page == "Visualization 📊":

    st.subheader("02 Data Viz")

    # Arrange image and scorecards side by side with smallest allowed gap
    img_col, metrics_col = st.columns([5, 5], gap="small")
    with img_col:
        st.image("TaxiNYC1.jpg", use_container_width=True)
    with metrics_col:
        # Calculate metrics
        avg_price = df['Total amount'].mean()
        # Calculate average tipping percentage
        avg_tip_pct = (df['Tip amount'] / (df['Total amount'] - df['Tip amount'])).replace([np.inf, -np.inf], np.nan).dropna().mean() * 100
        avg_tip_pct = np.round(avg_tip_pct, 2) if not np.isnan(avg_tip_pct) else 0
        df['Pickup day'] = pd.to_datetime(df['Pickup time']).dt.day_name()
        busi")est_day = df['Pickup day'].mode()[0]
        avg_distance = df['Trip distance'].mean()
        # Scorecards in a 2x2 grid
        row1_col1, row1_col2 = st.columns(2)
        row1_col1.metric("Average Price", f"${avg_price:,.2f}")
        row1_col2.metric("Average Distance", f"{avg_distance:,.2f} mi")
        # Add vertical space before the bottom two scorecards
        st.write("")
        st.write("")
        row2_col1, row2_col2 = st.columns(2)
        row2_col1.metric("Average Tip %", f"{avg_tip_pct:.2f}%")
        # Show actual busiest date, formatted as 'Jan. 1' (no year)
        busiest_date = pd.to_datetime(df['Pickup time']).dt.date.value_counts().idxmax()
        formatted_busiest_date = pd.to_datetime(str(busiest_date)).strftime('%b. %-d')
        row2_col2.metric("Busiest Date", formatted_busiest_date)

    fig, ax = plt.subplots()
    ax.scatter(df['Trip distance'], df['Total amount'], alpha=0.5)
    ax.set_xlabel('Trip Distance')
    ax.set_ylabel('Total Fare')
    ax.set_title('Total Fare vs Trip Distance')
    st.pyplot(fig)

    # --- Additional Visualizations (Pie and Line Chart side-by-side) ---
    st.markdown('---')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Passenger Count Distribution**')
        # Convert passenger counts to float for pie chart
        passenger_counts = df['Passenger count'].value_counts().sort_index().astype(float)
        fig_pie, ax_pie = plt.subplots()
        # Show pie chart percentages as whole numbers (no decimal points)
        ax_pie.pie(passenger_counts, labels=passenger_counts.index, autopct='%d%%', startangle=90)
        ax_pie.set_title('Share of Rides by Passenger Count')
        st.pyplot(fig_pie)
    with col2:
        st.markdown('**Popularity Throughout the Night (by Minute)**')
        # Extract minute of day from pickup time
        df['Pickup minute'] = pd.to_datetime(df['Pickup time']).dt.hour * 60 + pd.to_datetime(df['Pickup time']).dt.minute
        # Night: 6pm (1080) to 2am (120) next day, so combine 18:00-23:59 and 0:00-1:59
        night_minutes = df[(df['Pickup minute'] >= 0) & (df['Pickup minute'] < 120)]
        minute_counts = night_minutes['Pickup minute'].value_counts().sort_index()
        fig_line, ax_line = plt.subplots()
        ax_line.plot(minute_counts.index, minute_counts.values, marker='o', linestyle='-')
        ax_line.set_xlabel('Minute of Night (0=Midnight, 120=2am)')
        ax_line.set_ylabel('Number of Rides')
        ax_line.set_title('Ride Popularity Throughout the Night (Midnight-2am, by Minute)')
        st.pyplot(fig_line)
    
    col_x = st.selectbox("Select X-axis variable",df.columns,index=0)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for Y-axis. Please check your data or select a different dataset.")
        col_y = None
    else:
        col_y = st.selectbox("Select Y-axis variable (numeric only)", numeric_cols, index=0)
       
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Bar Chart 📊","Line Chart 📈","Correlation Heatmap 🔥", "Histogram 📊", "Scatter Plot ⚡"])

    with tab1:
        st.subheader("Bar Chart")
        if col_y:
            try:
                if col_x in df.columns and col_y in df.columns:
                    plot_df = df[[col_x, col_y]].dropna()
                    if not plot_df.empty and isinstance(plot_df[col_y], pd.Series):
                        plot_df = plot_df[pd.to_numeric(plot_df[col_y], errors='coerce').notnull()]
                        if not np.issubdtype(plot_df[col_x].dtype, np.number):
                            plot_df[col_x] = plot_df[col_x].astype(str)
                        st.bar_chart(plot_df, x=col_x, y=col_y, use_container_width=True)
                    else:
                        st.info("No valid data to plot. Please check your column selections.")
                else:
                    st.info("Selected columns are not valid. Please select different columns.")
            except Exception as e:
                st.warning(f"Could not generate bar chart: {e}")
        else:
            st.info("Please select a numeric column for the Y-axis.")

    with tab2:
        st.subheader("Line Chart")
        if col_y:
            try:
                if col_x in df.columns and col_y in df.columns:
                    plot_df = df[[col_x, col_y]].dropna()
                    if not plot_df.empty and isinstance(plot_df[col_y], pd.Series):
                        plot_df = plot_df[pd.to_numeric(plot_df[col_y], errors='coerce').notnull()]
                        if not np.issubdtype(plot_df[col_x].dtype, np.number):
                            plot_df[col_x] = plot_df[col_x].astype(str)
                        st.line_chart(plot_df, x=col_x, y=col_y, use_container_width=True)
                    else:
                        st.info("No valid data to plot. Please check your column selections.")
                else:
                    st.info("Selected columns are not valid. Please select different columns.")
            except Exception as e:
                st.warning(f"Could not generate line chart: {e}")
        else:
            st.info("Please select a numeric column for the Y-axis.")

    with tab3:
        st.subheader("Correlation Matrix")
        df_numeric = df.select_dtypes(include=np.number)
        fig_corr, ax_corr = plt.subplots(figsize=(40,30))
        sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
        st.pyplot(fig_corr)

    with tab4:
        st.subheader("Histogram of Numeric Columns")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_hist_col = st.selectbox("Select column for histogram", numeric_cols, key="hist_col")
        if selected_hist_col:
            fig, ax = plt.subplots()
            df[selected_hist_col].dropna().hist(bins=30, ax=ax)
            ax.set_title(f"Histogram of {selected_hist_col}")
            ax.set_xlabel(selected_hist_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    with tab5:
        st.subheader("Scatter Plot")
        scatter_x = st.selectbox("Select X-axis for scatter plot", df.columns, index=0, key="scatter_x")
        scatter_y = st.selectbox("Select Y-axis for scatter plot (numeric only)", numeric_cols, index=0, key="scatter_y")
        if scatter_x and scatter_y:
            fig, ax = plt.subplots()
            ax.scatter(df[scatter_x], df[scatter_y], alpha=0.5)
            ax.set_xlabel(scatter_x)
            ax.set_ylabel(scatter_y)
            ax.set_title(f"{scatter_y} vs {scatter_x}")
            st.pyplot(fig)
        else:
            st.info("Please select columns for the scatter plot.")




elif page == "Automated Report 📑":
    st.subheader("03 Automated Report")
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            profile = ProfileReport(df,title="Taxi Fare Report",explorative=True,minimal=True)
            st_profile_report(profile)

        export = profile.to_html()
        st.download_button(label="📥 Download full Report",data=export,file_name="california_housing_report.html",mime='text/html')
elif page == "Prediction":
    st.subheader("04 Prediction with Linear Regression")
    df2 = pd.read_csv("updated_taxidata.csv")
    # Rename columns to match pretty names
    df2.columns = ['Vendor ID', 'Pickup time', 'Dropoff time', 'Passenger count', 'Trip distance', 'Payment type', 'Fare amount', 'Extra', 'MTA tax', 'Tip amount', 'Tolls amount', 'Improvement surcharge', 'Total amount', 'Congestion surcharge']

    ### removing missing values 
    df2 = df2.dropna()

    ### i) X and y
    x = df2[['Passenger count','Trip distance','Extra','MTA tax','Tip amount','Tolls amount','Improvement surcharge','Congestion surcharge',]]
    y = df2['Total amount']

    
    col_pred1, col_pred2 = st.columns([3, 1])
    with col_pred1:
        st.markdown('**Feature Table**')
        st.dataframe(x, use_container_width=True, hide_index=True, height=400)
    with col_pred2:
        st.markdown('**Total Amount**')
        st.dataframe(y.to_frame('Total amount'), use_container_width=True, hide_index=True, height=400)


    ### ii) train_test_split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


    ## Model 

    ### i) Definition model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()

    ### ii) Training model
    model.fit(x_train,y_train)

    ### iii) Prediction
    predictions = model.predict(x_test)

    # Let user select metrics to display
    selected_metrics = st.multiselect(
        "Select metrics to display:",
        ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"],
        default=["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"]
    )

    ### iv) Evaluation 
    from sklearn import metrics 
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE** {mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE** {mae:,.2f}")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R2** {r2:,.3f}")

    st.success(f"My model performance is of ${np.round(mae,2)}")

    # --- Prediction Visualization ---
    st.subheader("Prediction Visualization")
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    
    # Optional: Residuals histogram
    residuals = y_test - predictions
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=30)
    ax.set_title("Histogram of Residuals (Actual - Predicted)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # --- User Prediction Form ---
    st.subheader("Try Your Own Prediction")
    with st.form("predict_form"):
        user_distance = st.number_input("Trip distance (miles)", min_value=0.0, value=1.0)
        user_passenger = st.number_input("Passenger count", min_value=1, value=1)
        tip_pct = st.slider("Tip percentage (%)", min_value=0, max_value=40, value=0, step=1)
        submitted = st.form_submit_button("Predict Fare")

    if submitted:
        # Predict base fare (without tip)
        base_input = pd.DataFrame([{
            'Passenger count': user_passenger,
            'Trip distance': user_distance,
            'Extra': 0,
            'MTA tax': 0.5,
            'Tip amount': 0,
            'Tolls amount': 0,
            'Improvement surcharge': 0.3,
            'Congestion surcharge': 2.5
        }])
        base_pred = model.predict(base_input)[0]
        # Calculate tip amount as a percentage of predicted base fare
        tip_amount = base_pred * (tip_pct / 100)
        user_input = base_input.copy()
        user_input['Tip amount'] = tip_amount
        user_pred = model.predict(user_input)[0]
        st.success(f"Predicted Total Fare (with {tip_pct}% tip): ${user_pred:.2f}")

