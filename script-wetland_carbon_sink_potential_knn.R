# Preamble ----
# This program uses environmental parameters to predict the potential of 
# wetland environments in the Pacific Northwest as carbon sinks using KNN
# classification.
#
# Author: Pat McCornack
# Date: 12/12/2022
#
# Data sourced from:
# Boone Kauffman, J., et al. Dataset: Carbon Stocks in 
# Seagrass Meadows, Emergent Marshes, and Forested Tidal Swamps of the Pacific
# Northwest. 2, Smithsonian Environmental Research Center, 11 Aug. 2020,
# doi:10.25573/serc.12640172.v2. 
# ----

# Load in Libraries
library(tidyverse)
library(dplyr)
library(zoo)
library(ggplot2)
library(class)
library(lattice)
library(caret)

## User Defined Functions ----
# min.max.norm used to normalize data using min-max normalization
min.max.norm <- function(x) {
  y <- (x - min(x)) / (max(x) - min(x))   
  return(y)
}

## Data Pre-Processing ---- 
# Shape the data into a dataframe that will be used to create classifications

# Read in data
setwd('./data/')
depth.series <- read_csv("kauffman_et_al_2020_depthseries.csv", show_col_types = F)
cores <- read_csv("kauffman_et_al_2020_cores.csv", show_col_types = F)

# Subset each dataframe to relevant variables
depth.series <- select(depth.series, -'depth_min', -'depth_max',
                       -'depth_interval_notes', -'representative_depth_max')
cores <- select(cores, 3, 9, 13, 16, 19:20)


# Remove observations with NAs from core dataframe
cores <- mutate(cores, core_elevation = na.approx(core_elevation))  # Interpolate where possible
cores <- na.omit(cores)  # Remove the rest


# Merge datasets and create factor variables
df <- merge(cores, depth.series, by = 'core_id')
df$vegetation_class <- as.factor(df$vegetation_class)
df$salinity_class <- as.factor(df$salinity_class)

## Create classes out of fraction_carbon - this will be response
fc.summary <- summary(df$fraction_carbon)
df$carbon_class <- cut(df$fraction_carbon,
                       breaks = c(0, fc.summary[2], fc.summary[3], fc.summary[5], 1),
                       labels = c('Low', 'Med', 'High', 'Very High'))

# Normalize the data using min-max normalization
df <- df %>% mutate_at(min.max.norm,
                       .vars = vars(core_elevation, porewater_salinity, pH,
                                    dry_bulk_density, representative_depth_min))


## Plot the class distribution

ggplot(data = df, aes(x = carbon_class)) +
  geom_bar(color = "black", fill = "#377eb8") +
  labs(title = "Distribution of Carbon Sequestration ",
       x = 'Carbon Sequestration Potential',
       y = 'Classification Count') +
  theme_bw() 

ggsave('./plots/true_class_distribution.jpg',  width = 4.5, height = 3.00)

## CV of multiple models ----
# The number of neighbors is chosen using cross-validation

knn.labels <- select(df, carbon_class)

# Continous variables only:
knn.cont <- select(df, c(core_elevation, porewater_salinity, pH,
                         dry_bulk_density, representative_depth_min))

# With categorical variables as dummies
knn.d.predictors <- select(df, c(-core_id, -study_id, -site_id, -carbon_class, -fraction_carbon))

# Encode the dummy variables
dummies <- dummyVars(~., data=knn.d.predictors)
knn.d.predictors <- data.frame(predict(dummies, knn.d.predictors))


# Cross-validation
k.fold <- 5  # Number of folds
k.neighbor <- c(1:50)
ncv <- ceiling(nrow(knn.cont) / k.fold)  # Number of observations per fold
cv.ind <- rep(1:k.fold, ncv)  # Used to assign observations to folds
set.seed(1234)
cv.ind.rand <- sample(cv.ind, nrow(knn.cont)) # Randomize the folds

cv.accuracy <- c() # Tracks results of each round of cross-validation
k.accuracy <- c() # Tracks results of KNN with each # neighbors 

cv.d.accuracy <- c() # Cross-validation with dummy variables
k.d.accuracy <- c() # Number neighbors with dummy variables

# CV Loop
for(i in k.neighbor){  # For each number of neighbors 
  for(j in 1:k.fold){  # For each fold
    set.seed(1234)
    # Continuous model
    knn.train.pred <- knn.cont[cv.ind.rand != j, ]  # Training dataset
    knn.test.pred <- knn.cont[cv.ind.rand == j, ]  # Testing dataset
    
    # With categorical model
    knn.d.train.pred <- knn.d.predictors[cv.ind.rand != j, ]  # Training dataset
    knn.d.test.pred <- knn.d.predictors[cv.ind.rand == j, ]  # Testing dataset
   
    knn.train.lab <- knn.labels[cv.ind.rand != j, ]
    knn.test.lab <- knn.labels[cv.ind.rand == j, ]
    
    # Continuous KNN model
    label.pred <- knn(knn.train.pred, knn.test.pred, knn.train.lab, k = i) 
    
    # Categorical KNN model
    label.d.pred <- knn(knn.d.train.pred, knn.d.test.pred, knn.train.lab, k = i)
    
    # Number correct over total number observations
    accuracy <- (sum(diag(table(knn.test.lab, label.pred)))) / length(knn.test.lab)
    accuracy.d <- (sum(diag(table(knn.test.lab, label.d.pred)))) / length(knn.test.lab)
    cv.accuracy[j] <- accuracy
    cv.d.accuracy[j] <- accuracy.d
  }
  k.accuracy[i] <- mean(cv.accuracy) 
  k.d.accuracy[i] <- mean(cv.d.accuracy)
}

# Gather results
results <- data.frame(rbind(max(k.accuracy), which.max(k.accuracy)), row.names = c('max.accuracy','max.neighbors'))
colnames(results)[1] <- 'KNN.continuous'
results$KNN.all <- c(max(k.d.accuracy), which.max(k.d.accuracy))

# Label results using optimal model
label.pred <- knn(knn.train.pred, knn.test.pred, knn.train.lab, k = which.max(k.accuracy)) 
knn.cont.labels <- data.frame(actual = knn.test.lab, predicted = label.pred)
knn.cont.labels <- pivot_longer(knn.cont.labels, cols = c('actual', 'predicted'), names_to = 'source', values_to = 'class')


label.d.pred <- knn(knn.d.train.pred, knn.d.test.pred, knn.train.lab, k = which.max(k.d.accuracy))
knn.all.labels <- data.frame(actual = knn.test.lab, predicted = label.d.pred)
knn.all.labels <- pivot_longer(knn.all.labels, cols = c('actual', 'predicted'), names_to = 'source', values_to = 'class')

error <- data.frame(k.neighbor, k.accuracy)
ggplot(data = error, aes(x=k.neighbor, y = k.accuracy)) +
  geom_point(color = 'dodgerblue3') +
  geom_line(color = 'dodgerblue3') + 
  labs(title = 'Number of Neighbors vs. KNN Accuracy',
       subtitle = 'Continuous wetland predictors',
       x = 'Number of Neighbors',
       y = 'Accuracy Rate') +
  theme_bw()
ggsave('./plots/num_neighbors/continous_KNN.jpg', width = 4.5, height = 3.00)

error.d <- data.frame(k.neighbor, k.d.accuracy)
ggplot(data = error.d, aes(x=k.neighbor, y = k.accuracy)) +
  geom_point(color = 'dodgerblue3') +
  geom_line(color = 'dodgerblue3') + 
  labs(title = 'Number of Neighbors vs. KNN Accuracy',
       subtitle = 'Continuous and categorical wetland predictors',
       x = 'Number of Neighbors',
       y = 'Accuracy Rate') +
  theme_bw()
ggsave('./plots/num_neighbors/all_KNN.jpg', width = 4.5, height = 3.00)

## CV of continuous data with PCA ----
# The number of neighbors is chosen using cross-validation

knn.labels <- select(df, carbon_class)

# Perform PCA on the training data
# Over 95% of variance is explained by first 3 PCs 
knn.cont <- select(df, c(core_elevation, porewater_salinity, pH,
                         dry_bulk_density, representative_depth_min))
knn.cont <- prcomp(knn.cont, center = FALSE, scale = FALSE)
summary(knn.cont)
knn.cont <- knn.cont$x[,1:3]

# Cross-validation
k.fold <- 5  # Number of folds
k.neighbor <- c(1:50)
ncv <- ceiling(nrow(knn.cont) / k.fold)  # Number of observations per fold
cv.ind <- rep(1:k.fold, ncv)  # Used to assign observations to folds
set.seed(1234)
cv.ind.rand <- sample(cv.ind, nrow(knn.cont)) # Randomize the folds

cv.accuracy <- c() # Tracks results of each round of cross-validation
k.accuracy <- c() # Tracks results of KNN with each # neighbors 

# CV Loop
for(i in k.neighbor){  # For each number of neighbors 
  for(j in 1:k.fold){  # For each fold
    set.seed(1234)
    # Continuous model
    knn.train.pred <- knn.cont[cv.ind.rand != j, ]  # Training dataset
    knn.test.pred <- knn.cont[cv.ind.rand == j, ]  # Testing dataset
    
    knn.train.lab <- knn.labels[cv.ind.rand != j, ]
    knn.test.lab <- knn.labels[cv.ind.rand == j, ]
    
    # Continuous KNN model
    label.pred <- knn(knn.train.pred, knn.test.pred, knn.train.lab, k = i) 

    # Number correct over total number observations
    accuracy <- (sum(diag(table(knn.test.lab, label.pred)))) / length(knn.test.lab)
    cv.accuracy[j] <- accuracy

  }
  k.accuracy[i] <- mean(cv.accuracy) 
}

error.pca <- data.frame(k.neighbor, k.accuracy)
ggplot(data = error.pca, aes(x=k.neighbor, y = k.accuracy)) +
  geom_point(color = 'dodgerblue3') +
  geom_line(color = 'dodgerblue3') + 
  labs(title = 'Number of Neighbors vs. KNN Accuracy',
       subtitle = 'Continuous wetland predictors with PCA',
       x = 'Number of Neighbors',
       y = 'Accuracy Rate') +
  theme_bw()
ggsave('./plots/num_neighbors/pca_KNN.jpg', width = 4.5, height = 3.00)

# Gather Results
results$KNN.PCA.cont <- c(max(k.accuracy), which.max(k.accuracy))

label.pred <- knn(knn.train.pred, knn.test.pred, knn.train.lab, k = which.max(k.accuracy)) 
knn.pca.labels <- data.frame(actual = knn.test.lab, predicted = label.pred)
knn.pca.labels <- pivot_longer(knn.pca.labels, cols = c('actual', 'predicted'), names_to = 'source', values_to = 'class')


## Visually analyze all results ----
ggplot(data = knn.cont.labels, aes(x = class, fill = source)) +
  geom_bar(position = 'dodge', color = "black") +
  scale_fill_brewer(palette = 'Set1') +
  labs(title = "Actual vs. Predicted Classifications",
       subtitle = "Using only continuous predictors",
       x = 'Carbon Sequestration Potential',
       y = 'Classification Count') +
  theme_bw() +
  guides(fill=guide_legend(""))
ggsave('./plots/predictions/continous_predictions.jpg', width = 4.5, height = 3.00)

ggplot(data = knn.all.labels, aes(x = class, fill = source)) +
  geom_bar(position = 'dodge', color = "black") +
  scale_fill_brewer(palette = 'Set1') +
  labs(title = "Actual vs. Predicted Classifications",
       subtitle = "Using categorical and continuous predictors",
       x = 'Carbon Sequestration Potential',
       y = 'Classification Count') +
  theme_bw() +
  guides(fill=guide_legend(""))
ggsave('./plots/predictions/all_predictions.jpg', width = 4.5, height = 3.00)

ggplot(data = knn.pca.labels, aes(x = class, fill = source)) +
  geom_bar(position = 'dodge', color = "black") +
  scale_fill_brewer(palette = 'Set1') +
  labs(title = "Actual vs. Predicted Classifications",
       subtitle = "Using PCA of continuous predictors",
       x = 'Carbon Sequestration Potential',
       y = 'Classification Count') +
  theme_bw()
ggsave('./plots/predictions/pca_predictions.jpg', width = 4.5, height = 3.00)

# Export results 
write.csv(results, './results.csv', row.names = T)

