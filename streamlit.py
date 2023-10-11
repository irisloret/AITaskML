import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

st.set_page_config(page_title="Machine Learning Algorithms", page_icon="🤖", layout="wide")

header = st.container()
EDA = st.container()
MLA = st.container()
matrix = st.container()
with header:
    st.title("Benchmarking machine learning algorithms")

with EDA:
    st.header("Performing exploratory data analysis")
    st.write("The dataset I chose, talks about different levels of obesity based on eating habits, physical condition if he/she smokes...")
    st.write("This dataset includes data from people from Mexico, Peru and Colombia.")
    with st.expander(":blue[Shape]", expanded=False):
        st.text("I will start by showing the number of rows and columns.")
        df = pd.read_csv('../Tasks/ObesityDataSet_raw_and_data_sinthetic.csv')
        st.write(df.shape)
        st.text("This number means that there are 2111 instances and 17 attributes.")
    with st.expander(":blue[First and last 5 lines of code]", expanded=False):
        st.write(df.head())
        st.write(df.tail())
    with st.expander(":blue[Count the null values]", expanded=False):
        st.write(df.isnull().sum())
        st.text("Since there are no null values we don't have to handle them.")
    with st.expander(":blue[Amount of different obesity levels]", expanded=False):
        obesity_counts = df['NObeyesdad'].value_counts()
        st.bar_chart(obesity_counts)
        st.text("We can see that obesity type 1 is most common in this dataset")
    with st.expander(":blue[Change categorical values]", expanded=False):
        st.text("In the data we see a lot of categorical values. We don't want that so we are going to change those into numerical values.")
        st.text("The following values are what we are going to change.")

        feature_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        X = df[feature_cols]
        st.write(X.head())
        y=df[['NObeyesdad']]
        ord = ce.OrdinalEncoder(cols=feature_cols)
        st.text("After changing it, it looks like this:")
        X_cat = ord.fit_transform(X)
        st.write(X_cat.head())

with MLA:
    st.header("Different machine learning algorithms")
    algorithm = st.radio("Select the algorithm you would like to see the result of:", ["Random Forest", "AdaBoost", "KNeighbors"])
    col1, col2 = st.columns(2)
    X_train, X_test, y_train, y_test = train_test_split(X_cat, y, test_size=0.3) 
    if algorithm == "Random Forest":
        number = col1.slider("Number of trees", min_value=50, max_value=200, value=100, step=10)
        rfc = RandomForestClassifier(n_estimators=number)
    elif algorithm == "AdaBoost":
        number = col1.slider("Number of estimators", min_value=1, max_value=50, value=10, step=1)
        learn = col1.slider("Learning rate", min_value=0.0, max_value=1.5, value=0.9, step=0.1)
        rfc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=number, learning_rate=learn)
    elif algorithm == "KNeighbors":
        number = col1.slider("Number of neighbors", min_value=1, max_value=25, value=9, step=2)

        rfc = KNeighborsClassifier(n_neighbors=number, metric='euclidean')

    rfc = rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_test)
    st.write(f"**Accuracy:** {metrics.accuracy_score(y_test, y_pred_rfc):.2%}")
    
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_rfc), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    col3, col4 = st.columns(2)
    col3.pyplot(fig)
st.header("Conclusion")
st.write("To conclude we can see that in most cases AdaBoost is the worst algorithm and Random Forest is the best.")
st.write("We can also see that none of these algorithms give a good result, maybe it is hard to classify this data.")



footer="""<style>

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}


</style>
<div class="footer">
<p>Iris Loret - 2023</p>
</div>


"""
st.markdown(footer,unsafe_allow_html=True)
    
