# R code to resample the skeletons
options(rgl.useNULL = TRUE)
library(nat)

skeletons_folder <- "/Users/ramyarajalakshmi/Documents/graph_comparison/skeletons_folder/"
result_folder <- "/Users/ramyarajalakshmi/Documents/graph_comparison/resampled/"

stepsize <- 0.5

skeletons <- read.neurons(skeletons_folder)
resampled <- lapply(skeletons, nat::resample, stepsize = stepsize)
write.neurons(resampled, result_folder, format = "swc", files = names(resampled))