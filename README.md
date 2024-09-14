# CryptoClustering

<h2>Unsupervised Machine Learning</h2>

<h3>Overview</h3>
<div>
In this assignment, I utilized Python and unsupervised machine learning to predict if cryptocurrencies are affected by 24-hour or 7-day price changes.
</div>
<hr/>
<div>
Goal:
<ul>
<li>Create a DataFrame with the scaled data.</li>
<li>Find the best value for k Using the original scaled dataframe </li>
<li>Draw scatterplots with K-means using the original scaled Data </li>
<li>Optimize clusters with principal component analysis</li>
<li>Find the best value for k Using the PCA data</li>
<li>Draw scatterplots with K-means using the PCA data</li>
</ul>

<h3>Tools and Techniques</h3>
<hr/>

<ul>
<li>Python</li>
<li>Pandas</li>
<li>NumPy</li>
<li>scikit-learn</li>
<li>hvPlot</li>
</ul>

</div>

<h3>Project Structure</h3>
<hr/>


<h4>Part 1: Transform the original dataset to scaled data</h4>

```python
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
market_data_scaled = StandardScaler().fit_transform(
    df_market_data[['price_change_percentage_24h', 'price_change_percentage_7d',
       'price_change_percentage_14d', 'price_change_percentage_30d',
       'price_change_percentage_60d', 'price_change_percentage_200d',
       'price_change_percentage_1y']]
)
# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(market_data_scaled,
             columns = ['price_change_percentage_24h', 'price_change_percentage_7d',
       'price_change_percentage_14d', 'price_change_percentage_30d',
       'price_change_percentage_60d', 'price_change_percentage_200d',
       'price_change_percentage_1y']
            )

# Copy the crypto names from the original data
df_market_data_scaled["coinid"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coinid")

# Display sample data
df_market_data_scaled.head()
```

<h4>Part 2: Find the best value for k & Cluster Cryptocurrencies with K-means with the Original Scaled Data</h4>

```python
# Create a list with the number of k-values from 1 to 11
k = list(range(1,11))
# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list

for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_market_data_scaled)
    inertia.append(model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k_original":k, "inertia":inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_original = pd.DataFrame(elbow_data)
df_elbow_original.head()

# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_plot_original = df_elbow_original.hvplot.line(
    x="k_original", 
    y="inertia", 
    title="Elbow Curve Original", 
    xticks=k
)

elbow_plot_original

# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=1)
# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)
# Predict the clusters to group the cryptocurrencies using the scaled data
k_4 = model.predict(df_market_data_scaled)

# Print the resulting array of cluster values.
k_4
# Create a copy of the DataFrame
market_data_predictions_df = df_market_data_scaled.copy()
# Add a new column to the DataFrame with the predicted clusters
market_data_predictions_df['class'] = k_4
# make the 'class' column data type to string
market_data_predictions_df['class'] = market_data_predictions_df['class'].astype(str)

# Display sample data
market_data_predictions_df.head()
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
market_scaled_plot_original = market_data_predictions_df.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    c="class",
    hover_cols=["coinid"]
)
market_scaled_plot_original
```

<h4>Part 3: Optimize Clusters with Principal Component Analysis and find the best value for k & Cluster Cryptocurrencies with K-means with the PCA Data</h4>


```python
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)

# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
market_pca = pca.fit_transform(df_market_data_scaled)
# View the first five rows of the DataFrame. 
market_pca[0:5]

# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_

# Create a new DataFrame with the PCA data.
market_pca_df = pd.DataFrame(
    market_pca,
    columns=["PC1", "PC2", "PC3"]
)

# Copy the crypto names from the original data
market_pca_df["coinid"] = df_market_data_scaled.index

# Set the coinid column as index
market_pca_df = market_pca_df.set_index("coinid")

# Display sample data
market_pca_df.head()

# Find the Best Value for k Using the PCA Data

# Create a list with the number of k-values from 1 to 11
k = list(range(1,11))

# Create an empty list to store the inertia values
inertia_pca = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, random_state=0)
    k_model.fit(market_pca_df)
    inertia_pca.append(k_model.inertia_)


# Create a dictionary with the data to plot the Elbow curve
elbow_data_pca = {"k_pca": k, "inertia": inertia_pca}
# Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(elbow_data_pca)

# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_pca_plot = df_elbow_pca.hvplot.line(
    x="k_pca", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)

elbow_pca_plot

# Cluster Cryptocurrencies with K-means Using the PCA Data

# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=2, random_state=0)

# Fit the K-Means model using the PCA data
model.fit(market_pca_df)

# Predict the clusters to group the cryptocurrencies using the PCA data
k_2_pca = model.predict(market_pca_df)
# Print the resulting array of cluster values.
k_2_pca

# Create a copy of the DataFrame with the PCA data
market_pca_predictions_df = market_pca_df.copy()

# Add a new column to the DataFrame with the predicted clusters
market_pca_predictions_df["class"] = k_2_pca

market_pca_predictions_df["class"] = market_pca_predictions_df["class"].astype(str)

# Display sample data
market_pca_predictions_df.head()

# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
market_scaled_plot_pca = market_pca_predictions_df.hvplot.scatter(
    x="PC1",
    y="PC2",
    by="class",
    hover_cols=["coinid"]
)
market_scaled_plot_pca

```
<h3>Q & A:</h3>
<hr/>
<p>
<b>Q1:</b>

What is the best value for k?

<b>Answer:</b>

After calculating the percentage change in inertia for different values of 𝐾, I observe that the percentage reduction in inertia is substantial up to K=4, with the values in the table (df_elbow_original) above.

At K=4, the percentage change is still relatively high (45.74%), but after K=4, the percentage change drops significantly to 16.20% and continues to decrease more gradually.

Thus, K=4 is the optimal number of clusters, as adding more clusters beyond this point provides diminishing returns in terms of reduced inertia. This means that while adding more clusters improves the model slightly, the improvement becomes much less significant after K=4.
</p>

<p>
<b>Q2:</b>

What is the best value for k when using the PCA data?

<b>Answer:</b>

About 89.5% of the total variance is condensed into the 3 PCA variables.
</p>

<p>
<b>Q3:</b>

What is the total explained variance of the three principal components?

<b>Answer:</b>

After calculating the percentage change in inertia for different values of K, we observe that the percentage reduction in inertia is substantial up to K=4, with the values in the table (df_elbow_pca) above.
At K=4, the percentage change is still relatively high (57.15%), but after K=4, the percentage change drops significantly to 16.73% and continues to decrease more gradually.

However, the initial drop at k=2 (34.28%) already indicates that adding a second cluster significantly improves the clustering.

Thus, K=2 is the optimal number of clusters, as adding more clusters beyond this point provides diminishing returns in terms of reduced inertia. This means that while adding more clusters improves the model slightly, the improvement becomes much less significant after K=4.
</p>

<p>
<b>Q4:</b>

Does it differ from the best k value found using the original data?

<b>Answer:</b>
no
</p>

<p>
<b>Q5:</b>

After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

<b>Answer:</b>

The analysis of the original dataset ("market_scaled_plot_original") using the model determined an optimal k-value of 4, indicating that four clusters were identified. In the graph, two distinct points, specifically "ethlend" and "celsius-degree-token," appear isolated as separate clusters. The boundaries between these clusters are not clearly defined in this visualization. In contrast, applying Principal Component Analysis (PCA) with three components (n_components=3) revealed that the elbow method suggests an optimal k-value of 2. This approach provides a clearer delineation of the clusters and better defines the thresholds for cluster separation.


</p>

<hr/>
<h3>Conclusion</h3>

<p>
The project examines how reducing the number of features affects clustering performance using K-means. By comparing the clustering results of the original dataset with those derived from PCA-reduced data, we gain insights into the impact of dimensionality reduction on the clustering process.
</p>