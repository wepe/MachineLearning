
# Reading in data
ds = read.csv('Mall_Customers.csv')
X = ds[,4:5]

# Creating dendrogram to choose k
hc = hclust(dist(X, method = "euclidean"), method = "ward.D")

plot(hc, labels = FALSE, hang = 0.03, 
     main = paste("Cluster Dendrogram"), 
     xlab = 'Customers', 
     ylab = "Euclidean distance")
    
# Clustering
y_hc = cutree(hc, 5)

plot(X, col = y_hc)
