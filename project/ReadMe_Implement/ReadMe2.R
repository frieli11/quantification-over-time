library(readme)
library(tensorflow)

# 定义一个函数
readme2 <- function(data_t, seed_) {
# readme2 <- function() {
  
  data_t = read.csv("0.csv")
  seed_ = 1
  
  data_t[["TEXT"]] <- gsub("\\0", "", data_t[["TEXT"]])

  wordVec_summaries = undergrad(documentText = cleanme(data_t$TEXT), wordVecs = NULL)
  
  # Estimate category proportions
  set.seed(seed_) # Set a seed if you choose
  readme.estimates <- readme(dfm = wordVec_summaries , labeledIndicator = data_t$TRAININGSET, categoryVec = data_t$TRUTH)
  
  # Output proportions estimate
  qua_prev = readme.estimates$point_readme
  
  prev_df <- as.data.frame(t(qua_prev))
  print(prev_df)
  write.csv(prev_df, file = "prev.csv", row.names = FALSE)

  # return(qua_prev)
}


# # 使用函数
# dataname = 'global_covid19_tweets'
# file_name = paste0(format(0), ".csv")  # 构建文件名
# relative_path = file.path("data", dataname, file_name)
# 
# datat = read.csv("0.csv")
# seed = 1
readme2(datat, seed)
# # prev = readme2(datat, seed)
# # print(prev)


