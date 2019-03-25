# load libraries
library(tidyverse) # group of packages to wrangle and visualize data
library(cluster) # cluster analysis
library(factoextra) # visualize clusters and principal components
library(dendextend) # visualize dendrograms
library(here) # create a file directory
library(ggrepel) # repel overlapping text labels
library(clustree) # visualize clusters
library(FactoMineR) # explore multivariate data
library(ggcorrplot) # visualize correlations
library(clValid) # compute cluster metrics
library(broom) # tidy algorithm outputs
library(umap) # dimension reduction
library(tidyquant) # in this case theme and color for clusters visualization


# load the file
pharmaceuticals <- read_csv(here::here("pharmaceuticals.csv"))

# explore the data
glimpse(pharmaceuticals)



# create correlation matrix
pharmaceuticals_cor <- pharmaceuticals %>%
  select_if(is.numeric) %>%
  cor()

# visualize correlations
ggcorrplot(pharmaceuticals_cor, 
           outline.color = "grey50", 
           lab = TRUE,
           hc.order = TRUE,
           type = "full") 



# use a data frame only with numeric values and scale the variables because they were measured in different scales
pharmaceuticals_tbl <- na.omit(pharmaceuticals) %>%
dplyr::select(-c(1, 12, 13, 14)) %>% 
column_to_rownames(var = "Name") %>%
scale(.) %>% # standardize the values 
as.data.frame() # convert to data frame

## use PCA to check how many dimensions we have
# PCA of our dataframe
new_pca <- PCA(pharmaceuticals_tbl)

# check eigenvalues and percentage of variance
new_pca$eig


# visualization of how much variance each dimension explains
fviz_screeplot(new_pca, addlabels = TRUE)


# get each variable PCA results
var <- get_pca_var(new_pca)

# each variable contribution to PC1 - top 5
fviz_contrib(new_pca, choice = "var", axes = 1, top = 5)

# each variable contribution to PC2 - top 5
fviz_contrib(new_pca, choice = "var", axes = 2, top = 5)

# each variable contribution to PC3 - top 5
fviz_contrib(new_pca, choice = "var", axes = 3, top = 5)

# each variable contribution to PC - top 5
fviz_contrib(new_pca, choice = "var", axes = 4, top = 5)


# visualization of the first two components and the contributions of each variable
fviz_pca_var(new_pca, col.var="contrib",
gradient.cols = c("red", "green", "blue"),
repel = TRUE 
) + 
labs( title = "Variables - PCA")


## Hierarchical Cluster Analysis

# compute distance measure
dt <- dist(pharmaceuticals_tbl, method = "euclidean")


# visualize distance
fviz_dist(dt, gradient = list(low = "red", mid = "white", high = "blue"))


m <- c("average", "single", "complete", "ward")
names(m) <- c("average", "single", "complete", "ward")


#function to check the best (means higher value) linkage method 
ac <- function(x) {
  agnes(dt, method = x)$ac
}

map_dbl(m, ac)


# hierarchical clustering 
set.seed(88)
hclust_1 <- hclust(dt, method = "ward.D2") # ward.D2 corresponds to the ward                                                method in the hclust function

# plot hierarchical clustering
plot(hclust_1, cex = 0.6)


# elbow method
fviz_nbclust(pharmaceuticals_tbl, FUNcluster = hcut, method = "wss")

# sillhouette method
fviz_nbclust(pharmaceuticals_tbl, FUNcluster = hcut, method = "silhouette")



# cutree function
cl_1 <- cutree(hclust_1, k = 2)

# table function check the number of pharmaceutical companies in each cluster
table(cl_1)


plot(hclust_1, cex = 0.6)
rect.hclust(hclust_1, k = 2, border= 2:5)



# fviz_cluster function to visualize the clusters
fviz_cluster(list(data = pharmaceuticals_tbl, cluster = cl_1, repel = TRUE)) +
theme_minimal()



# create cluster variable
pharmaceuticals_tbl$cluster <- cl_1

# aggregate by cluster our variables
pharmaceuticals_tbl %>%
group_by(cluster) %>%
summarise_all(mean)


## K-Means Cluster Analysis


# use a data frame only with numeric values and scale the variables because they were measured in different scales
pharmaceuticals_tbl <- na.omit(pharmaceuticals) %>%
dplyr::select(-c(1, 12, 13, 14)) %>% 
column_to_rownames(var = "Name") %>%
scale(.) %>% # standardize the values 
as.data.frame() # convert to data frame

# elbow method
fviz_nbclust(pharmaceuticals_tbl, FUNcluster = kmeans, method = "wss")

# sillhouette method
fviz_nbclust(pharmaceuticals_tbl, FUNcluster = kmeans, method = "silhouette")

# build algorithm
set.seed(88)
k_cluster2 <- kmeans(pharmaceuticals_tbl, centers = 2, nstart = 50,
iter.max = 10) # k equals 2 clusters

table(k_cluster2$cluster)


# check total within and between sum of squares
glance(k_cluster2)

# dunn index
dunn_k2 <- dunn(clusters = k_cluster2$cluster, Data = pharmaceuticals_tbl)
dunn_k2




set.seed(88)
k_cluster3 <- kmeans(pharmaceuticals_tbl, centers = 3, nstart = 50,
iter.max = 10) # centers equals 3 clusters



table(k_cluster3$cluster)


# check wSS and BSS
glance(k_cluster3)
tidy(k_cluster3)

# check dunn index
dunn_k3 <- dunn(clusters = k_cluster3$cluster, Data = pharmaceuticals_tbl)
dunn_k3


# umap our data frame
umap_pharma <- pharmaceuticals_tbl %>%
umap()

# create umap dataframe
umap_obj <- umap_pharma$layout %>%
as.data.frame() %>%
rownames_to_column(var = "Pharma")

umap_obj


# visualize umap dataframe
umap_obj %>%
ggplot(aes(V1, V2)) +
geom_point() +
geom_label_repel(aes(label = Pharma))


# use augment to assign the clusters to our pharmaceutical companies
kmeans_tbl <- augment(k_cluster3, pharmaceuticals_tbl) %>% 
dplyr::select(pharma = .rownames, .cluster)

# join the kmeans data frame with the umap object
kmeans_umap <- kmeans_tbl %>% 
left_join(umap_obj, by = c("pharma" = "Pharma"))

kmeans_umap


kmeans_umap %>% 
mutate(label_pharma = str_glue("Company: {pharma}
Cluster:{.cluster}")) %>%
ggplot(aes(V1, V2, color = .cluster)) +
geom_point() +
geom_label_repel(aes(label = label_pharma), size = 2.5) +
guides(color = FALSE) +
theme_minimal() +
scale_color_tq() +
labs(title = "Pharmaceutical Companies Segmentation",
subtitle = "K-Means Cluster Algorithm with UMAP Projection")


k_cluster3 %>%
augment(pharmaceuticals_tbl) %>%
dplyr::select(-.rownames) %>%
group_by(.cluster) %>% 
summarise_all(mean)



# create tibble withthe characteristics of the 3 cluster
cluster_tibble <- tibble::tribble(~.cluster, ~cluster.label,
                                  1, "Non Profitable/High Risk Investment/Underpriced Stocks",
                                  2, "Non Profitable/High Risk Investment/Overpriced Stocks",
                                  3, "Profitable/Low Risk Investment")


# make .cluster variable a factor
cluster_tibble <- cluster_tibble %>%
  mutate(.cluster = as.factor(as.character(.cluster)))



# clusters visualization
kmeans_umap %>%
  left_join(cluster_tibble) %>%
  mutate(label_pharma = str_glue("Company: {pharma}
                                 Cluster:{.cluster}
                                 {cluster.label}")) %>%
  ggplot(aes(V1, V2, color = .cluster)) +
  geom_point() +
  geom_label_repel(aes(label = label_pharma), size = 2) +
  guides(color = FALSE) +
  theme_tq() +
  scale_color_tq() +
  labs(title = "Pharmaceutical Companies Segmentation",
       subtitle = "UMAP 2D Projection with the K-Means Cluster Algorithm")
