library(readme)
library(tensorflow)


datasets = list(
  'global_covid19_tweets',
  'nepali_dataset_eng',              
  'Apple-Twitter-Sentiment-DFE'
  )

seeds = list(
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10
  )


for (s in seeds) {
  print(paste0("++++++++++++++++++++++++++++++++++++++++++++++seed", s, "++++++++++++++++++++++++++++++++++++++++++++++"))
  for (dataname in datasets) {
    print(paste0("----------------------------", dataname, "----------------------------"))
    seed_path = file.path("data", dataname, paste0("seed", s))
    if (!file.exists(seed_path)) {
      dir.create(seed_path)
    }
    
    val_preds <- data.frame(matrix(ncol = 3, nrow = 0))
    colnames(val_preds) <- c("-1", "0", "1")
    i = 0
    repeat {
      print(paste("...................val", i, "..................."))
      file_name = paste0(format(i), ".csv")  # Build file name
      
      relative_path <- file.path("data", dataname, "val", file_name)
      
      if(!file.exists(relative_path)) {
        break
      }
      data_t = read.csv(relative_path)
      # a = data_t["TEXT"]
      # print(a)
      data_t[["TEXT"]] <- gsub("\\0", "", data_t[["TEXT"]])
      
      # Estimate category proportions
      wordVec_summaries = undergrad(documentText = cleanme(data_t$TEXT), wordVecs = NULL)
      set.seed(s) # Set a seed if you choose
      readme.estimates <- readme(dfm = wordVec_summaries , labeledIndicator = data_t$TRAININGSET, categoryVec = data_t$TRUTH)
      
      # Output proportions estimate
      val_pred = readme.estimates$point_readme
      val_preds <- rbind(val_preds, val_pred)  # Adding new rows to a DataFrame
      # print(val_pred)
      
      # # Compare to the truth
      # true = table(data_t$TRUTH[data_t$TRAININGSET == 0])/sum(table((data_t$TRUTH[data_t$TRAININGSET == 0])))
      # print(true)
      
      i = i+1
    }
    write.csv(val_preds, file.path(seed_path, "val_preds.csv"), row.names = FALSE)
    
    
    test_preds <- data.frame(matrix(ncol = 3, nrow = 0))
    colnames(test_preds) <- c("-1", "0", "1")
    j = 0
    repeat {
      print(paste("...................test", j, "..................."))
      file_name = paste0(format(j), ".csv")  # Build file name
      
      relative_path <- file.path("data", dataname, "test", file_name)
      
      if(!file.exists(relative_path)) {
        break
      }
      data_t = read.csv(relative_path)
      # a = data_t["TEXT"]
      # print(a)
      data_t[["TEXT"]] <- gsub("\\0", "", data_t[["TEXT"]])
      
      # Estimate category proportions
      wordVec_summaries = undergrad(documentText = cleanme(data_t$TEXT), wordVecs = NULL)
      set.seed(s) # Set a seed if you choose
      readme.estimates <- readme(dfm = wordVec_summaries , labeledIndicator = data_t$TRAININGSET, categoryVec = data_t$TRUTH)
      
      # Output proportions estimate
      test_pred = readme.estimates$point_readme
      test_preds <- rbind(test_preds, test_pred)  # Adding new rows to a DataFrame
      # print(test_preds)
      
      # # Compare to the truth
      # true = table(data_t$TRUTH[data_t$TRAININGSET == 0])/sum(table((data_t$TRUTH[data_t$TRAININGSET == 0])))
      # print(true)
      
      j = j+1
    } 
    write.csv(test_preds, file.path(seed_path, "test_preds.csv"), row.names = FALSE)
  }
  
}



