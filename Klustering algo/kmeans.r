
# Reading in data
ds = read.csv('Mall_Customers.csv')
X = ds[4:5]

# Finding k
wcss = vector()
for (i in 1:10) 
    wcss[i] =sum(kmeans(X, i)$withinss)

plot(1:10, wcss, type = 'b', main=paste("Elbow method"), xlab = 'number clusters' )

# Clustering
kmeans = kmeans(X, 5)
y_kmeans = kmeans$cluster

# Visualising the clusters
plot(X, col = y_kmeans)
points(kmeans$center,col=1:2,pch=8,cex=1)
